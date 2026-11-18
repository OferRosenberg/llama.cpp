#include "qairt-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace ggml::qairt {
namespace {
constexpr size_t staging_limit = 256 * 1024 * 1024;
constexpr size_t cache_limit = 512;
constexpr uint64_t static_cache_limit = uint64_t(1) * 1024 * 1024 * 1024;

qairt_error conversion_error(const char * text) { return host_error(error_stage::numerical_check, "float preparation", text); }

result<std::vector<uint8_t>> materialize(const ggml_tensor * t, enum ggml_type type,
                                         const std::vector<pending_output> & pending) {
    using bytes_result = result<std::vector<uint8_t>>;
    if (!readable_tensor(t) || dense_bytes(t) > staging_limit) {
        return bytes_result::failure(conversion_error("Unsupported source buffer or staging size"));
    }
    const size_t source_element = t->type == GGML_TYPE_F32 ? 4 : 2;
    const size_t target_element = type == GGML_TYPE_F32 ? 4 : 2;
    const size_t count = size_t(ggml_nelements(t));
    if (count > staging_limit / target_element) { return bytes_result::failure(conversion_error("Conversion exceeds staging size")); }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    const auto address = reinterpret_cast<uintptr_t>(t->data);
    // Overlay uncommitted predecessors, including views into their contiguous destinations.
    for (const auto & output : pending) {
        const auto base = reinterpret_cast<uintptr_t>(output.tensor->data);
        if (base >= address) {
            const size_t offset = base - address;
            if (offset < raw.size()) { std::memcpy(raw.data() + offset, output.bytes.data(), std::min(output.bytes.size(), raw.size() - offset)); }
        } else {
            const size_t offset = address - base;
            if (offset < output.bytes.size()) { std::memcpy(raw.data(), output.bytes.data() + offset, std::min(raw.size(), output.bytes.size() - offset)); }
        }
    }
    std::vector<uint8_t> packed(count * target_element);
    if (t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q6_K) {
        const auto * traits = ggml_get_type_traits(t->type);
        if (!traits->to_float || type != GGML_TYPE_F16) { return bytes_result::failure(conversion_error("Unsupported quantized conversion")); }
        std::vector<float> row(size_t(t->ne[0]));
        for (size_t r = 0; r < size_t(t->ne[1]); ++r) {
            const size_t offset = r * t->nb[1];
            if (offset > raw.size() || t->nb[1] > raw.size() - offset) { return bytes_result::failure(conversion_error("Quantized row exceeds allocation")); }
            traits->to_float(raw.data() + offset, row.data(), t->ne[0]);
            for (size_t col = 0; col < row.size(); ++col) {
                if (!std::isfinite(row[col]) || std::fabs(row[col]) > 65504.0f) { return bytes_result::failure(conversion_error("Invalid quantized FP16 weight")); }
                const auto half = ggml_fp32_to_fp16(row[col]);
                std::memcpy(packed.data() + (r * row.size() + col) * 2, &half, 2);
            }
        }
        return bytes_result::success(std::move(packed));
    }
    for (size_t row = 0; row < count / size_t(t->ne[0]); ++row) {
        for (size_t col = 0; col < size_t(t->ne[0]); ++col) {
            const size_t r1 = row % size_t(t->ne[1]), outer = row / size_t(t->ne[1]);
            const size_t offset = col * t->nb[0] + r1 * t->nb[1] + (outer % size_t(t->ne[2])) * t->nb[2] + (outer / size_t(t->ne[2])) * t->nb[3];
            const size_t out = (row * size_t(t->ne[0]) + col) * target_element;
            if (offset > raw.size() || source_element > raw.size() - offset) { return bytes_result::failure(conversion_error("Source stride exceeds allocation")); }
            if (t->type == type) { std::memcpy(packed.data() + out, raw.data() + offset, source_element); continue; }
            float value;
            if (t->type == GGML_TYPE_F32) { std::memcpy(&value, raw.data() + offset, 4); }
            else if (t->type == GGML_TYPE_BF16) {
                ggml_bf16_t value_bf16; std::memcpy(&value_bf16, raw.data() + offset, sizeof(value_bf16));
                value = ggml_bf16_to_fp32(value_bf16);
            }
            else { ggml_fp16_t half; std::memcpy(&half, raw.data() + offset, 2); value = ggml_fp16_to_fp32(half); }
            if (!std::isfinite(value) || (type == GGML_TYPE_F16 && std::fabs(value) > 65504.0f)) {
                return bytes_result::failure(conversion_error("Nonfinite or overflowing FP16 boundary value"));
            }
            if (type == GGML_TYPE_F32) { std::memcpy(packed.data() + out, &value, 4); }
            else { const auto half = ggml_fp32_to_fp16(value); std::memcpy(packed.data() + out, &half, 2); }
        }
    }
    return bytes_result::success(std::move(packed));
}

std::vector<uint32_t> qnn_shape(const ggml_tensor * tensor, bool rows) {
    const uint32_t count = uint32_t(ggml_nelements(tensor) / tensor->ne[0]), channels = uint32_t(tensor->ne[0]);
    if (!rows) { return {count, channels}; }
    // Factor independent rows across spatial dimensions for HTP normalization tiling.
    const uint32_t height = row_height(count);
    return {1, height, count / height, channels};
}

struct softmax_boundary {
    std::vector<fp16_bits> logits, mask;
};
result<softmax_boundary> pack_softmax(const ggml_tensor * operation, const std::vector<pending_output> & pending) {
    using packed_result = result<softmax_boundary>;
    const auto * tensor = operation->src[0]; const auto * mask = operation->src[1];
    float scale; std::memcpy(&scale, operation->op_params, sizeof(scale));
    auto bytes = materialize(tensor, GGML_TYPE_F32, pending);
    if (!bytes.ok()) { return packed_result::failure(bytes.error()); }
    const size_t columns = size_t(tensor->ne[0]), queries = size_t(tensor->ne[1]), count = size_t(ggml_nelements(tensor));
    softmax_boundary packed; packed.logits.resize(count);
    std::vector<float> mask_values;
    if (mask) {
        // Read original bits so unused padding and valid -infinity masks need no global float conversion.
        auto raw = materialize(mask, mask->type, pending);
        if (!raw.ok()) { return packed_result::failure(raw.error()); }
        mask_values.resize(columns * queries * size_t(mask->ne[2]) * size_t(mask->ne[3]));
        packed.mask.resize(mask_values.size());
        for (size_t row = 0; row < queries * size_t(mask->ne[2]) * size_t(mask->ne[3]); ++row) {
            const size_t source_row = row / queries * size_t(mask->ne[1]) + row % queries;
            for (size_t col = 0; col < columns; ++col) {
                const size_t source = source_row * columns + col, destination = row * columns + col;
                float value;
                if (mask->type == GGML_TYPE_F32) { std::memcpy(&value, raw.value().data() + source * 4, 4); }
                else { ggml_fp16_t half; std::memcpy(&half, raw.value().data() + source * 2, 2); value = ggml_fp16_to_fp32(half); }
                const bool excluded = value == -std::numeric_limits<float>::infinity();
                if (!excluded && (!std::isfinite(value) || std::fabs(value) > 65504.0f)) {
                    return packed_result::failure(conversion_error("Invalid or overflowing FP16 Softmax mask"));
                }
                mask_values[destination] = value;
                packed.mask[destination].bits = ggml_fp32_to_fp16(excluded ? -65504.0f : value);
            }
        }
    }
    for (size_t start = 0; start < count; start += columns) {
        const size_t row = start / columns, query = row % queries;
        const size_t head = row / queries % size_t(tensor->ne[2]), batch = row / (queries * size_t(tensor->ne[2]));
        const size_t mask_start = mask ? ((batch % size_t(mask->ne[3]) * size_t(mask->ne[2]) + head % size_t(mask->ne[2])) * queries + query) * columns : 0;
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t col = 0; col < columns; ++col) {
            float value; std::memcpy(&value, bytes.value().data() + (start + col) * sizeof(float), sizeof(value)); value *= scale;
            if (!std::isfinite(value)) { return packed_result::failure(conversion_error("Nonfinite scaled Softmax input")); }
            const double additive = mask ? mask_values[mask_start + col] : 0.0;
            maximum = std::max(maximum, double(value) + additive);
            std::memcpy(bytes.value().data() + (start + col) * sizeof(float), &value, sizeof(value));
        }
        if (!std::isfinite(maximum)) { return packed_result::failure(conversion_error("Softmax row has no unmasked finite value")); }
        // Centre before conversion; QNN performs mask addition, exponentials and normalization.
        for (size_t col = 0; col < columns; ++col) {
            float value; std::memcpy(&value, bytes.value().data() + (start + col) * sizeof(float), sizeof(value));
            const bool excluded = mask && mask_values[mask_start + col] == -std::numeric_limits<float>::infinity();
            const double centered = excluded ? 0.0 : double(value) - maximum;
            if (mask && std::fabs(centered) > 65504.0) {
                return packed_result::failure(conversion_error("Masked Softmax centered input exceeds FP16 range"));
            }
            packed.logits[start + col].bits = ggml_fp32_to_fp16(float(std::max(-65504.0, centered)));
        }
    }
    return packed_result::success(std::move(packed));
}
bool static_source(const ggml_tensor * t, const std::vector<pending_output> & pending) {
    auto b = t->view_src ? t->view_src->buffer : t->buffer;
    if (!qairt_buffer(b) || ggml_backend_buffer_get_usage(b) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) { return false; }
    const auto ptr = reinterpret_cast<uintptr_t>(t->data);
    for (const auto & output : pending) {
        const auto out = reinterpret_cast<uintptr_t>(output.tensor->data);
        if ((out >= ptr && out - ptr < ggml_nbytes(t)) || (out < ptr && ptr - out < output.bytes.size())) { return false; }
    }
    return true;
}

std::vector<uint64_t> structural_key(const ggml_tensor * t, bool constant) {
    std::vector<uint64_t> key { uint64_t(t->op), uint64_t(constant) };
    const ggml_tensor * operands[] = {t, t->src[0], t->src[1]};
    for (const auto * value : operands) {
        key.push_back(value != nullptr);
        if (!value) { continue; }
        key.push_back(value->type);
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) { key.push_back(value->ne[i]); key.push_back(value->nb[i]); }
    }
    if (rowwise_type(t->op)) {
        uint32_t parameter; std::memcpy(&parameter, t->op_params, sizeof(parameter)); key.push_back(parameter);
        if (t->op == GGML_OP_SOFT_MAX) {
            std::memcpy(&parameter, reinterpret_cast<const uint8_t *>(t->op_params) + sizeof(float), sizeof(parameter)); key.push_back(parameter);
        }
    }
    if (silu_op(t)) { key.push_back(uint64_t(ggml_get_unary_op(t))); }
    if (constant) {
        const auto * source = t->src[0];
        auto b = source->view_src ? source->view_src->buffer : source->buffer;
        const auto * allocation = static_cast<const buffer_context *>(b->context);
        key.push_back(allocation->identity); key.push_back(allocation->generation);
        key.push_back(reinterpret_cast<uintptr_t>(source->data) - reinterpret_cast<uintptr_t>(allocation->storage));
    }
    return key;
}
}

result<std::vector<fp16_bits>> backend_context::pack(const ggml_tensor * t, const std::vector<pending_output> & pending, input_preparation preparation) const {
    using packed_result = result<std::vector<fp16_bits>>;
    if (preparation == input_preparation::bf16 && !runtime_->config().direct_fp16) {
        if (t->type != GGML_TYPE_F32) { return packed_result::failure(conversion_error("Invalid BF16 activation boundary")); }
        auto bytes = materialize(t, GGML_TYPE_F32, pending);
        if (!bytes.ok()) { return packed_result::failure(bytes.error()); }
        std::vector<fp16_bits> values(bytes.value().size() / sizeof(float));
        for (size_t i = 0; i < values.size(); ++i) {
            float value; std::memcpy(&value, bytes.value().data() + i * sizeof(float), sizeof(float));
            if (!std::isfinite(value)) { return packed_result::failure(conversion_error("Nonfinite BF16 activation")); }
            // Match CPU BF16 MatMul activation rounding before the FP16 QNN boundary.
            value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
            if (!std::isfinite(value) || std::fabs(value) > 65504.0f) { return packed_result::failure(conversion_error("Overflowing FP16 activation after BF16 rounding")); }
            values[i].bits = ggml_fp32_to_fp16(value);
        }
        return packed_result::success(std::move(values));
    }
    if (preparation == input_preparation::q8_k && !runtime_->config().direct_fp16) {
        if (t->type != GGML_TYPE_F32 || t->ne[0] % ggml_blck_size(GGML_TYPE_Q8_K)) { return packed_result::failure(conversion_error("Invalid Q8_K activation boundary")); }
        auto bytes = materialize(t, GGML_TYPE_F32, pending);
        if (!bytes.ok()) { return packed_result::failure(bytes.error()); }
        std::vector<fp16_bits> values(size_t(t->ne[0]) * size_t(t->ne[1]));
        std::vector<float> row(size_t(t->ne[0]));
        std::vector<block_q8_K> quantized(size_t(t->ne[0]) / size_t(ggml_blck_size(GGML_TYPE_Q8_K)));
        for (size_t r = 0; r < size_t(t->ne[1]); ++r) {
            std::memcpy(row.data(), bytes.value().data() + r * row.size() * 4, row.size() * 4);
            for (const float value : row) {
                if (!std::isfinite(value) || std::fabs(value) > 65504.0f) { return packed_result::failure(conversion_error("Invalid quantized activation")); }
            }
            const size_t block_size = size_t(ggml_blck_size(GGML_TYPE_Q8_K));
            for (size_t start = 0; start < row.size(); start += block_size) {
                float maximum = 0.0f;
                for (size_t col = start; col < start + block_size; ++col) { maximum = std::max(maximum, std::fabs(row[col])); }
                if (maximum != 0.0f && !std::isfinite(127.0f / maximum)) {
                    return packed_result::failure(conversion_error("Q8_K activation scale overflows"));
                }
            }
            // Match GGML's quantized MatMul activation preparation before the FP16 QNN boundary.
            quantize_row_q8_K_ref(row.data(), quantized.data(), t->ne[0]);
            dequantize_row_q8_K(quantized.data(), row.data(), t->ne[0]);
            for (size_t col = 0; col < row.size(); ++col) {
                const auto half = ggml_fp32_to_fp16(row[col]);
                if (!std::isfinite(ggml_fp16_to_fp32(half))) { return packed_result::failure(conversion_error("Nonfinite reconstructed FP16 activation")); }
                values[r * row.size() + col].bits = half;
            }
        }
        return packed_result::success(std::move(values));
    }
    auto bytes = materialize(t, GGML_TYPE_F16, pending);
    if (!bytes.ok()) { return packed_result::failure(bytes.error()); }
    std::vector<fp16_bits> values(bytes.value().size() / 2);
    std::memcpy(values.data(), bytes.value().data(), bytes.value().size());
    for (const auto half : values) {
        if (!std::isfinite(ggml_fp16_to_fp32(half.bits))) { return packed_result::failure(conversion_error("Nonfinite FP16 input")); }
    }
    return packed_result::success(std::move(values));
}

result<prepared_run *> backend_context::prepare_run(const std::vector<ggml_tensor *> & nodes, const std::vector<pending_output> & pending) {
    using prepared_result = result<prepared_run *>;
    struct source_record { const ggml_tensor * tensor; size_t node; size_t source; bool constant; input_preparation preparation; tensor_id id; };
    std::vector<source_record> sources;
    std::vector<std::array<int, 2>> topology(nodes.size());
    std::vector<uint64_t> key {nodes.size()};
    const bool rows = rowwise_type(nodes[0]->op) != nullptr || silu_op(nodes[0]);
    auto shape = [&](const ggml_tensor * tensor) {
        // Preserve every logical dimension for residual broadcasting.
        const bool masked_softmax = nodes[0]->op == GGML_OP_SOFT_MAX && nodes[0]->src[1];
        if (nodes[0]->op == GGML_OP_ADD || masked_softmax) {
            const uint32_t queries = uint32_t(masked_softmax && tensor == nodes[0]->src[1] ? nodes[0]->ne[1] : tensor->ne[1]);
            return std::vector<uint32_t>{uint32_t(tensor->ne[3]), uint32_t(tensor->ne[2]), queries, uint32_t(tensor->ne[0])};
        }
        return qnn_shape(tensor, rows);
    };
    size_t static_bytes = (nodes[0]->op == GGML_OP_NORM || nodes[0]->op == GGML_OP_RMS_NORM) ? size_t(nodes[0]->ne[0]) * 4 + sizeof(uint32_t) : 0;
    const auto dtype = QNN_DATATYPE_FLOAT_16;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto * t = nodes[i];
        const bool constant = t->op == GGML_OP_MUL_MAT && static_source(t->src[0], pending);
        auto part = structural_key(t, constant); key.insert(key.end(), part.begin(), part.end());
        for (size_t j = 0; j < 2; ++j) {
            const auto * source = t->src[j]; if (!source) { continue; } int internal = -1;
            for (size_t previous = 0; previous < i; ++previous) { if (nodes[previous] == source) { internal = int(previous); } }
            if (internal >= 0) { topology[i][j] = -internal - 1; key.push_back(uint64_t(-internal - 1)); continue; }
            const bool is_static = constant && j == 0;
            input_preparation preparation = input_preparation::none;
            if (!runtime_->config().direct_fp16 && j == 1 && t->op == GGML_OP_MUL_MAT && source->type == GGML_TYPE_F32) {
                if (ggml_is_quantized(t->src[0]->type)) { preparation = input_preparation::q8_k; }
                else if (t->src[0]->type == GGML_TYPE_BF16) { preparation = input_preparation::bf16; }
            }
            size_t external = 0;
            for (; external < sources.size(); ++external) {
                if (sources[external].tensor == source && sources[external].constant == is_static && sources[external].preparation == preparation &&
                    (t->op != GGML_OP_SOFT_MAX || sources[external].source == j)) { break; }
            }
            if (external == sources.size()) {
                sources.push_back({source, i, j, is_static, preparation, {}});
                if (is_static) { static_bytes += size_t(source->ne[0]) * size_t(source->ne[1]) * 2; }
            }
            topology[i][j] = int(external); key.push_back(external); key.push_back(uint64_t(preparation));
        }
    }
    for (size_t i = 0; i < cache_.size(); ++i) {
        if (cache_[i]->key == key) {
            auto entry = std::move(cache_[i]); cache_.erase(cache_.begin() + i); cache_.push_back(std::move(entry));
            ++stats_.cache_hits; return prepared_result::success(cache_.back().get());
        }
    }
    uint64_t retained = 0;
    for (const auto & value : cache_) { retained += value->static_bytes; }
    while (!cache_.empty() && (cache_.size() >= cache_limit || retained + static_bytes > static_cache_limit)) {
        retained -= cache_.front()->static_bytes; cache_.erase(cache_.begin());
    }
    auto context = context_owner::create(runtime_);
    if (!context.ok()) { return prepared_result::failure(context.error()); }
    auto graph = graph_owner::create(context.value(), "dense_run");
    if (!graph.ok()) { return prepared_result::failure(graph.error()); }
    auto entry = std::make_unique<prepared_run>();
    entry->key = std::move(key); entry->graph = graph.value(); entry->static_bytes = static_bytes;
    for (size_t i = 0; i < sources.size(); ++i) {
        auto & source = sources[i];
        tensor_spec spec {"input" + std::to_string(i), source.constant ? QNN_TENSOR_TYPE_STATIC : QNN_TENSOR_TYPE_APP_WRITE,
                          dtype, shape(source.tensor)};
        result<tensor_id> registered = result<tensor_id>::failure(conversion_error("Tensor not registered"));
        if (source.constant) {
            auto values = pack(source.tensor, pending);
            if (!values.ok()) { return prepared_result::failure(values.error()); }
            registered = graph.value()->register_tensor(std::move(spec), {values.value().data(), values.value().size() * 2});
        } else { registered = graph.value()->register_tensor(std::move(spec)); }
        if (!registered.ok()) { return prepared_result::failure(registered.error()); }
        source.id = registered.value();
        if (!source.constant) { entry->inputs.push_back({source.node, source.source, source.id, source.preparation}); }
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto id = graph.value()->register_tensor({"output" + std::to_string(i), QNN_TENSOR_TYPE_APP_READ, dtype,
                                                 shape(nodes[i])});
        if (!id.ok()) { return prepared_result::failure(id.error()); }
        entry->outputs.push_back(id.value());
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        tensor_id ids[2] {};
        for (size_t j = 0; j < 2; ++j) {
            if (!nodes[i]->src[j]) { continue; }
            const int index = topology[i][j]; ids[j] = index < 0 ? entry->outputs[size_t(-index - 1)] : sources[size_t(index)].id;
        }
        op_spec spec;
        if (silu_op(nodes[i])) {
            auto sigmoid = graph.value()->register_tensor({"sigmoid" + std::to_string(i), QNN_TENSOR_TYPE_NATIVE,
                                                          QNN_DATATYPE_FLOAT_16, qnn_shape(nodes[i], true)});
            if (!sigmoid.ok()) { return prepared_result::failure(sigmoid.error()); }
            auto activation = op_spec::sigmoid(ids[0], sigmoid.value());
            activation.name = "node" + std::to_string(i) + "_sigmoid";
            auto added = graph.value()->add_node(activation);
            if (!added.ok()) { return prepared_result::failure(added.error()); }
            spec = {"silu", "qti.aisw", "ElementWiseMultiply", {ids[0], sigmoid.value()}, {entry->outputs[i]}, {}, {}};
        }
        else if (rowwise_type(nodes[i]->op)) {
            float parameter; std::memcpy(&parameter, nodes[i]->op_params, sizeof(parameter));
            if (nodes[i]->op == GGML_OP_SOFT_MAX) {
                auto logits = ids[0];
                if (nodes[i]->src[1]) {
                    auto masked = graph.value()->register_tensor({"masked_logits" + std::to_string(i), QNN_TENSOR_TYPE_NATIVE, dtype, shape(nodes[i])});
                    if (!masked.ok()) { return prepared_result::failure(masked.error()); }
                    op_spec add {"node" + std::to_string(i) + "_mask", "qti.aisw", "ElementWiseAdd", {ids[0], ids[1]}, {masked.value()}, {}, {}};
                    auto added = graph.value()->add_node(add);
                    if (!added.ok()) { return prepared_result::failure(added.error()); }
                    logits = masked.value();
                }
                spec = op_spec::softmax(logits, entry->outputs[i], 1.0f, 3);
            }
            else {
                const uint32_t channels = uint32_t(nodes[i]->ne[0]);
                std::vector<fp16_bits> gamma(channels), beta(channels);
                for (auto & value : gamma) { value.bits = ggml_fp32_to_fp16(1.0f); }
                auto g = graph.value()->register_tensor({"gamma", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, {channels}},
                                                        {gamma.data(), gamma.size() * sizeof(fp16_bits)});
                auto b = graph.value()->register_tensor({"beta", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, {channels}},
                                                        {beta.data(), beta.size() * sizeof(fp16_bits)});
                if (!g.ok()) { return prepared_result::failure(g.error()); }
                if (!b.ok()) { return prepared_result::failure(b.error()); }
                spec = op_spec::normalization(ids[0], g.value(), b.value(), entry->outputs[i], nodes[i]->op == GGML_OP_RMS_NORM, parameter, 3);
            }
        }
        else if (nodes[i]->op == GGML_OP_MUL_MAT) { spec = op_spec::matmul(ids[1], ids[0], entry->outputs[i]); }
        else { spec = {"elementwise", "qti.aisw", elementwise_type(nodes[i]->op), {ids[0], ids[1]}, {entry->outputs[i]}, {}}; }
        spec.name = "node" + std::to_string(i);
        auto added = graph.value()->add_node(spec);
        if (!added.ok()) { return prepared_result::failure(added.error()); }
    }
    auto finalized = graph.value()->finalize();
    if (!finalized.ok()) { return prepared_result::failure(finalized.error()); }
    cache_.push_back(std::move(entry)); ++stats_.preparations;
    return prepared_result::success(cache_.back().get());
}
bool backend_context::reserve_cache(size_t additional, const prepared_run * keep) {
    if (additional > static_cache_limit) { return false; }
    uint64_t retained = 0;
    for (const auto & entry : cache_) { retained += entry->static_bytes; }
    while ((!keep && cache_.size() >= cache_limit) || retained > static_cache_limit - additional) {
        auto victim = std::find_if(cache_.begin(), cache_.end(), [keep](const auto & entry) { return entry.get() != keep; });
        if (victim == cache_.end()) { return false; }
        retained -= (*victim)->static_bytes; cache_.erase(victim);
    }
    return true;
}

result<prepared_run *> backend_context::prepare_rope(const ggml_tensor * t, const std::vector<float> & factors, size_t rows) {
    using prepared_result = result<prepared_run *>;
    constexpr size_t table_limit = 64 * 1024 * 1024;
    const auto params = rope_params(t);
    const size_t pairs = size_t(params.dimensions / 2);
    if (rows > table_limit / (pairs * 2 * sizeof(fp16_bits))) {
        return prepared_result::failure(conversion_error("RoPE table exceeds 64 MiB cache limit"));
    }
    auto key = structural_key(t, false);
    for (size_t i = 0; i < sizeof(t->op_params) / sizeof(uint32_t); ++i) {
        uint32_t word; std::memcpy(&word, reinterpret_cast<const uint8_t *>(t->op_params) + i * sizeof(word), sizeof(word)); key.push_back(word);
    }
    key.push_back(factors.size());
    for (float factor : factors) { uint32_t word; std::memcpy(&word, &factor, sizeof(word)); key.push_back(word); }
    prepared_run * cached = nullptr;
    for (size_t i = 0; i < cache_.size(); ++i) {
        if (cache_[i]->key == key) {
            auto entry = std::move(cache_[i]); cache_.erase(cache_.begin() + i); cache_.push_back(std::move(entry));
            cached = cache_.back().get(); ++stats_.cache_hits; break;
        }
    }
    std::unique_ptr<prepared_run> fresh;
    if (!cached) {
        fresh = std::make_unique<prepared_run>(); fresh->key = std::move(key); fresh->rope = std::make_unique<rope_tables>();
        cached = fresh.get();
    }
    auto & table = *cached->rope;
    if (rows > table.rows) {
        const size_t bytes = rows * pairs * 2 * sizeof(fp16_bits);
        if (!reserve_cache(bytes - cached->static_bytes, fresh ? nullptr : cached)) {
            return prepared_result::failure(conversion_error("RoPE cache exceeds retained payload limit"));
        }
        table.cosine.resize(rows * pairs); table.sine.resize(rows * pairs);
        const float theta_scale = std::pow(params.base, -2.0f / params.dimensions);
        float correction[2] {};
        if (params.extension != 0.0f) {
            ggml_rope_yarn_corr_dims(params.dimensions, params.context, params.base,
                                    params.beta_fast, params.beta_slow, params.truncate, correction);
        }
        // Match the GGML CPU reference's float recurrence and YaRN scaling.
        for (size_t position = table.rows; position < rows; ++position) {
            float theta = float(position);
            for (size_t pair = 0; pair < pairs; ++pair) {
                const float extrapolated = theta / (factors.empty() ? 1.0f : factors[pair]);
                const float interpolated = params.scale * extrapolated;
                float angle = interpolated, magnitude = params.attention;
                if (params.extension != 0.0f) {
                    const float ramp = 1.0f - std::min(1.0f, std::max(0.0f, (float(pair) - correction[0]) / std::max(0.001f, correction[1] - correction[0])));
                    const float mix = ramp * params.extension;
                    angle = interpolated * (1.0f - mix) + extrapolated * mix;
                    magnitude *= 1.0f + 0.1f * std::log(1.0f / params.scale);
                }
                const float cosine = std::cos(angle) * magnitude, sine = std::sin(angle) * magnitude;
                if (!std::isfinite(cosine) || !std::isfinite(sine) || std::fabs(cosine) > 65504.0f || std::fabs(sine) > 65504.0f) {
                    return prepared_result::failure(conversion_error("Invalid precomputed FP16 RoPE value"));
                }
                table.cosine[position * pairs + pair].bits = ggml_fp32_to_fp16(cosine);
                table.sine[position * pairs + pair].bits = ggml_fp32_to_fp16(sine);
                theta *= theta_scale;
            }
        }
        table.rows = rows; cached->static_bytes = bytes;
    }
    if (!fresh) { return prepared_result::success(cached); }
    auto context = context_owner::create(runtime_);
    if (!context.ok()) { return prepared_result::failure(context.error()); }
    auto graph = graph_owner::create(context.value(), "rope_run");
    if (!graph.ok()) { return prepared_result::failure(graph.error()); }
    cached->graph = graph.value();
    const std::vector<uint32_t> shape {uint32_t(t->ne[3]), uint32_t(t->ne[1]), uint32_t(t->ne[2]), uint32_t(t->ne[0])};
    const std::vector<uint32_t> selected {shape[0], shape[2], uint32_t(pairs)};
    auto x = graph.value()->register_tensor({"x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, shape});
    auto cosine = graph.value()->register_tensor({"cosine", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, selected});
    auto sine = graph.value()->register_tensor({"sine", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, selected});
    auto y = graph.value()->register_tensor({"y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, shape});
    for (const auto * id : {&x, &cosine, &sine, &y}) { if (!id->ok()) { return prepared_result::failure(id->error()); } }
    table.cosine_input = cosine.value(); table.sine_input = sine.value();
    cached->inputs.push_back({0, 0, x.value(), input_preparation::none}); cached->outputs.push_back(y.value());
    auto added = graph.value()->add_node(op_spec::rotary_embedding(x.value(), cosine.value(), sine.value(), y.value(),
                                                                  uint32_t(params.dimensions), params.mode == GGML_ROPE_TYPE_NORMAL));
    if (!added.ok()) { return prepared_result::failure(added.error()); }
    auto finalized = graph.value()->finalize();
    if (!finalized.ok()) { return prepared_result::failure(finalized.error()); }
    cache_.push_back(std::move(fresh)); ++stats_.preparations;
    return prepared_result::success(cache_.back().get());
}

status backend_context::run_rope(ggml_tensor * t, std::vector<pending_output> & pending) {
    const auto params = rope_params(t);
    const size_t dimensions = size_t(t->ne[0]), heads = size_t(t->ne[1]), tokens = size_t(t->ne[2]), batches = size_t(t->ne[3]);
    const size_t pairs = size_t(params.dimensions / 2);
    std::vector<int32_t> positions(tokens);
    ggml_backend_tensor_get(t->src[1], positions.data(), 0, positions.size() * sizeof(int32_t));
    size_t rows = 1024;
    for (int32_t position : positions) {
        if (position < 0 || position >= (1 << 20)) { return status::failure(conversion_error("RoPE position outside supported cache range")); }
        while (rows <= size_t(position)) { rows *= 2; }
    }
    std::vector<float> factors;
    if (t->src[2]) {
        auto bytes = materialize(t->src[2], GGML_TYPE_F32, pending);
        if (!bytes.ok()) { return status::failure(bytes.error()); }
        factors.resize(pairs); std::memcpy(factors.data(), bytes.value().data(), pairs * sizeof(float));
        for (float factor : factors) {
            if (!std::isfinite(factor) || factor <= 0.0f) { return status::failure(conversion_error("Invalid RoPE frequency factor")); }
        }
    }
    auto activation = pack(t->src[0], pending);
    if (!activation.ok()) { return status::failure(activation.error()); }
    auto passthrough = params.dimensions < t->ne[0] ? materialize(t->src[0], t->type, pending) :
                       result<std::vector<uint8_t>>::success({});
    if (!passthrough.ok()) { return status::failure(passthrough.error()); }
    auto prepared = prepare_rope(t, factors, rows);
    if (!prepared.ok()) { return status::failure(prepared.error()); }
    auto & entry = *prepared.value(); auto & table = *entry.rope;
    std::vector<fp16_bits> x(activation.value().size()), y(x.size()), cosine(batches * tokens * pairs), sine(cosine.size());
    // GGML stores [B,T,H,D]; QNN RotaryEmbedding consumes [B,H,T,D].
    for (size_t b = 0; b < batches; ++b) {
        for (size_t token = 0; token < tokens; ++token) {
            const size_t source = size_t(positions[token]) * pairs, destination = (b * tokens + token) * pairs;
            std::copy_n(table.cosine.data() + source, pairs, cosine.data() + destination);
            std::copy_n(table.sine.data() + source, pairs, sine.data() + destination);
            for (size_t h = 0; h < heads; ++h) {
                std::copy_n(activation.value().data() + ((b * tokens + token) * heads + h) * dimensions, dimensions,
                            x.data() + ((b * heads + h) * tokens + token) * dimensions);
            }
        }
    }
    auto binding = execution_binding::create(entry.graph);
    if (!binding.ok()) { return status::failure(binding.error()); }
    auto bound = binding.value()->bind_input(entry.inputs[0].id, {x.data(), x.size() * sizeof(fp16_bits)});
    if (!bound.ok()) { return bound; }
    bound = binding.value()->bind_input(table.cosine_input, {cosine.data(), cosine.size() * sizeof(fp16_bits)});
    if (!bound.ok()) { return bound; }
    bound = binding.value()->bind_input(table.sine_input, {sine.data(), sine.size() * sizeof(fp16_bits)});
    if (!bound.ok()) { return bound; }
    bound = binding.value()->bind_output(entry.outputs[0], {y.data(), y.size() * sizeof(fp16_bits)});
    if (!bound.ok()) { return bound; }
    auto executed = entry.graph->execute(*binding.value());
    if (!executed.ok()) { return executed; }
    ++stats_.executions;
    std::vector<uint8_t> bytes(dense_bytes(t));
    const size_t element = t->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
    for (size_t b = 0; b < batches; ++b) {
        for (size_t token = 0; token < tokens; ++token) {
            for (size_t h = 0; h < heads; ++h) {
                for (size_t d = 0; d < dimensions; ++d) {
                    const auto half = y[((b * heads + h) * tokens + token) * dimensions + d].bits;
                    const float value = ggml_fp16_to_fp32(half);
                    if (!std::isfinite(value)) { return status::failure(conversion_error("Nonfinite QNN RoPE output")); }
                    const size_t destination = (((b * tokens + token) * heads + h) * dimensions + d) * element;
                    if (d >= size_t(params.dimensions)) {
                        // Partial RoPE leaves the remaining GGML channels bit-exact.
                        std::memcpy(bytes.data() + destination, passthrough.value().data() + destination, element);
                    } else if (t->type == GGML_TYPE_F32) { std::memcpy(bytes.data() + destination, &value, sizeof(value)); }
                    else { std::memcpy(bytes.data() + destination, &half, sizeof(half)); }
                }
            }
        }
    }
    pending.push_back({t, std::move(bytes)});
    return status::success();
}

status backend_context::run_attention_decomposed(ggml_tensor * t, std::vector<pending_output> & pending) {
    const auto * q = t->src[0]; const auto * k = t->src[1]; const auto * v = t->src[2];
    const size_t width = size_t(q->ne[0]), queries = size_t(q->ne[1]), heads = size_t(q->ne[2]), kv_heads = size_t(k->ne[2]), capacity = size_t(k->ne[1]);
    auto raw_mask = materialize(t->src[3], GGML_TYPE_F16, pending); if (!raw_mask.ok()) { return status::failure(raw_mask.error()); }
    auto mask_at = [&](size_t row, size_t col) { ggml_fp16_t value; std::memcpy(&value, raw_mask.value().data() + (row * capacity + col) * 2, 2); return ggml_fp16_to_fp32(value); };
    size_t total = 0; while (total < capacity && mask_at(queries - 1, total) == 0.0f) { ++total; }
    if (total < queries) { return status::failure(conversion_error("GQA requires a populated causal KV prefix")); }
    const size_t past = total - queries;
    for (size_t row = 0; row < queries; ++row) { for (size_t col = 0; col < capacity; ++col) {
        const float value = mask_at(row, col);
        if (col <= past + row ? value != 0.0f : !(std::isinf(value) && value < 0.0f)) { return status::failure(conversion_error("GQA mask must match the causal KV prefix")); }
    } }
    auto key = structural_key(t, false); key.push_back(UINT64_MAX); key.push_back(total);
    for (size_t source = 2; source < 4; ++source) { for (size_t axis = 0; axis < GGML_MAX_DIMS; ++axis) { key.push_back(t->src[source]->ne[axis]); key.push_back(t->src[source]->nb[axis]); } }
    uint32_t scale_bits; std::memcpy(&scale_bits, t->op_params, sizeof(scale_bits)); key.push_back(scale_bits);
    prepared_run * entry = nullptr;
    for (size_t i = 0; i < cache_.size(); ++i) {
        if (cache_[i]->key == key) { auto hit = std::move(cache_[i]); cache_.erase(cache_.begin() + i); cache_.push_back(std::move(hit)); entry = cache_.back().get(); ++stats_.cache_hits; break; }
    }
    if (!entry) {
        if (!reserve_cache(0)) { return status::failure(conversion_error("GQA cache limit")); }
        auto context = context_owner::create(runtime_); if (!context.ok()) { return status::failure(context.error()); }
        auto graph = graph_owner::create(context.value(), "grouped_attention_decomposed"); if (!graph.ok()) { return status::failure(graph.error()); }
        auto prepared = std::make_unique<prepared_run>(); prepared->key = std::move(key); prepared->graph = graph.value();
        const std::vector<std::vector<uint32_t>> dimensions {{uint32_t(queries), uint32_t(width)}, {uint32_t(total), uint32_t(width)},
            {uint32_t(width), uint32_t(total)}, {uint32_t(queries), uint32_t(total)}, {uint32_t(queries), uint32_t(total)},
            {uint32_t(queries), uint32_t(total)}, {uint32_t(queries), uint32_t(total)}, {uint32_t(queries), uint32_t(width)}};
        std::vector<tensor_id> ids;
        for (size_t i = 0; i < dimensions.size(); ++i) {
            auto id = graph.value()->register_tensor({"tensor" + std::to_string(i), i < 4 ? QNN_TENSOR_TYPE_APP_WRITE : i == 7 ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                QNN_DATATYPE_FLOAT_16, dimensions[i]});
            if (!id.ok()) { return status::failure(id.error()); } ids.push_back(id.value());
            if (i < 4) { prepared->inputs.push_back({0, i, id.value()}); }
        }
        const std::vector<op_spec> ops {op_spec::matmul(ids[0], ids[1], ids[4]),
            {"mask_add", "qti.aisw", "ElementWiseAdd", {ids[4], ids[3]}, {ids[5]}, {}, {}},
            op_spec::softmax(ids[5], ids[6], 1.0f, 1), op_spec::matmul(ids[6], ids[2], ids[7])};
        for (size_t i = 0; i < ops.size(); ++i) { auto op = ops[i]; op.name += std::to_string(i); auto added = graph.value()->add_node(op); if (!added.ok()) { return added; } }
        auto finalized = graph.value()->finalize(); if (!finalized.ok()) { return finalized; }
        prepared->outputs.push_back(ids[7]); cache_.push_back(std::move(prepared)); entry = cache_.back().get(); ++stats_.preparations;
    }
    auto packed_q = pack(q, pending); auto packed_k = pack(k, pending); auto packed_v = pack(v, pending);
    if (!packed_q.ok()) { return status::failure(packed_q.error()); } if (!packed_k.ok()) { return status::failure(packed_k.error()); } if (!packed_v.ok()) { return status::failure(packed_v.error()); }
    float scale; std::memcpy(&scale, t->op_params, sizeof(scale));
    std::vector<fp16_bits> query(queries * width), keys(total * width), values(width * total), mask(queries * total), output(queries * width);
    for (size_t row = 0; row < queries; ++row) { for (size_t col = 0; col < total; ++col) { mask[row * total + col].bits = ggml_fp32_to_fp16(col <= past + row ? 0.0f : -65504.0f); } }
    std::vector<uint8_t> bytes(dense_bytes(t));
    for (size_t head = 0; head < heads; ++head) {
        const size_t kv = head / (heads / kv_heads);
        for (size_t i = 0; i < query.size(); ++i) {
            const float value = ggml_fp16_to_fp32(packed_q.value()[head * query.size() + i].bits) * scale;
            if (!std::isfinite(value) || std::fabs(value) > 65504.0f) { return status::failure(conversion_error("GQA scaled query overflows FP16")); }
            query[i].bits = ggml_fp32_to_fp16(value);
        }
        std::copy_n(packed_k.value().data() + kv * capacity * width, keys.size(), keys.data());
        for (size_t row = 0; row < total; ++row) { for (size_t col = 0; col < width; ++col) { values[col * total + row] = packed_v.value()[(kv * capacity + row) * width + col]; } }
        auto binding = execution_binding::create(entry->graph); if (!binding.ok()) { return status::failure(binding.error()); }
        const std::vector<fp16_bits> * inputs[] = {&query, &keys, &values, &mask};
        for (size_t i = 0; i < 4; ++i) { auto bound = binding.value()->bind_input(entry->inputs[i].id, {inputs[i]->data(), inputs[i]->size() * 2}); if (!bound.ok()) { return bound; } }
        auto bound = binding.value()->bind_output(entry->outputs[0], {output.data(), output.size() * 2}); if (!bound.ok()) { return bound; }
        auto executed = entry->graph->execute(*binding.value()); if (!executed.ok()) { return executed; } ++stats_.executions;
        for (size_t row = 0; row < queries; ++row) { for (size_t col = 0; col < width; ++col) {
            const float value = ggml_fp16_to_fp32(output[row * width + col].bits); if (!std::isfinite(value)) { return status::failure(conversion_error("Nonfinite GQA output")); }
            std::memcpy(bytes.data() + ((row * heads + head) * width + col) * sizeof(float), &value, sizeof(value));
        } }
    }
    pending.push_back({t, std::move(bytes)}); return status::success();
}

status backend_context::run_attention(ggml_tensor * t, std::vector<pending_output> & pending) {
    const auto * q = t->src[0]; const auto * k = t->src[1]; const auto * v = t->src[2]; const auto * mask = t->src[3];
    const size_t width = size_t(q->ne[0]), queries = size_t(q->ne[1]), heads = size_t(q->ne[2]), kv_heads = size_t(k->ne[2]), capacity = size_t(k->ne[1]);
    auto mask_bytes = materialize(mask, GGML_TYPE_F16, pending);
    if (!mask_bytes.ok()) { return status::failure(mask_bytes.error()); }
    auto mask_at = [&](size_t row, size_t col) {
        ggml_fp16_t half; std::memcpy(&half, mask_bytes.value().data() + (row * capacity + col) * sizeof(half), sizeof(half));
        return ggml_fp16_to_fp32(half);
    };
    size_t total = 0;
    while (total < capacity && mask_at(queries - 1, total) == 0.0f) { ++total; }
    if (total < queries) { return status::failure(conversion_error("GQA requires a populated causal KV prefix")); }
    const size_t past = total - queries;
    for (size_t row = 0; row < queries; ++row) { for (size_t col = 0; col < capacity; ++col) {
        const float value = mask_at(row, col);
        if (col <= past + row ? value != 0.0f : !(std::isinf(value) && value < 0.0f)) {
            return status::failure(conversion_error("GQA mask must match the causal KV prefix"));
        }
    } }
    auto key = structural_key(t, false); key.push_back(total);
    for (size_t source = 2; source < 4; ++source) { for (size_t axis = 0; axis < GGML_MAX_DIMS; ++axis) { key.push_back(t->src[source]->ne[axis]); key.push_back(t->src[source]->nb[axis]); } }
    uint32_t scale_bits; std::memcpy(&scale_bits, t->op_params, sizeof(scale_bits)); key.push_back(scale_bits);
    prepared_run * entry = nullptr;
    for (size_t i = 0; i < cache_.size(); ++i) {
        if (cache_[i]->key == key) {
            auto hit = std::move(cache_[i]); cache_.erase(cache_.begin() + i); cache_.push_back(std::move(hit)); entry = cache_.back().get(); ++stats_.cache_hits; break;
        }
    }
    if (!entry) {
        if (!reserve_cache(0)) { return status::failure(conversion_error("GQA cache limit")); }
        auto context = context_owner::create(runtime_); if (!context.ok()) { return status::failure(context.error()); }
        auto graph = graph_owner::create(context.value(), "grouped_attention"); if (!graph.ok()) { return status::failure(graph.error()); }
        auto prepared = std::make_unique<prepared_run>(); prepared->key = std::move(key); prepared->graph = graph.value();
        std::vector<tensor_id> ids(10, null_tensor_id);
        const std::vector<std::vector<uint32_t>> dims {{1, uint32_t(queries), uint32_t(heads * width)}, {1}, {},
            {1, uint32_t(queries), uint32_t(kv_heads * width)}, {1, uint32_t(queries), uint32_t(kv_heads * width)},
            {1, uint32_t(kv_heads), uint32_t(past), uint32_t(width)}, {1, uint32_t(kv_heads), uint32_t(past), uint32_t(width)}};
        for (size_t i = 0; i < 7; ++i) {
            if (i >= 5 && !past) { continue; }
            auto id = graph.value()->register_tensor({"input" + std::to_string(i), QNN_TENSOR_TYPE_APP_WRITE,
                i == 1 || i == 2 ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_FLOAT_16, dims[i]});
            if (!id.ok()) { return status::failure(id.error()); } ids[i] = id.value(); prepared->inputs.push_back({0, i, id.value()});
        }
        for (size_t i = 0; i < 3; ++i) {
            auto id = graph.value()->register_tensor({"output" + std::to_string(i), QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
                i == 0 ? dims[0] : std::vector<uint32_t>{1, uint32_t(kv_heads), uint32_t(total), uint32_t(width)}});
            if (!id.ok()) { return status::failure(id.error()); } prepared->outputs.push_back(id.value());
        }
        float scale; std::memcpy(&scale, t->op_params, sizeof(scale));
        auto added = graph.value()->add_node(op_spec::group_query_attention(ids, prepared->outputs, uint32_t(heads), uint32_t(kv_heads), scale));
        if (!added.ok()) { return added; } auto finalized = graph.value()->finalize(); if (!finalized.ok()) { return finalized; }
        cache_.push_back(std::move(prepared)); entry = cache_.back().get(); ++stats_.preparations;
    }
    auto packed_q = pack(q, pending); auto packed_k = pack(k, pending); auto packed_v = pack(v, pending);
    if (!packed_q.ok()) { return status::failure(packed_q.error()); } if (!packed_k.ok()) { return status::failure(packed_k.error()); } if (!packed_v.ok()) { return status::failure(packed_v.error()); }
    std::vector<std::vector<fp16_bits>> inputs(7);
    inputs[0].resize(queries * heads * width);
    for (size_t row = 0; row < queries; ++row) { for (size_t head = 0; head < heads; ++head) {
        std::copy_n(packed_q.value().data() + (head * queries + row) * width, width, inputs[0].data() + (row * heads + head) * width);
    } }
    for (size_t i : {size_t(3), size_t(4)}) {
        const auto & packed = i == 3 ? packed_k.value() : packed_v.value(); inputs[i].resize(queries * kv_heads * width); inputs[i + 2].resize(past * kv_heads * width);
        for (size_t head = 0; head < kv_heads; ++head) {
            if (past) { std::copy_n(packed.data() + head * capacity * width, past * width, inputs[i + 2].data() + head * past * width); }
            for (size_t row = 0; row < queries; ++row) {
                std::copy_n(packed.data() + (head * capacity + past + row) * width, width, inputs[i].data() + (row * kv_heads + head) * width);
            }
        }
    }
    int32_t sequence = int32_t(total - 1), maximum = int32_t(total);
    auto binding = execution_binding::create(entry->graph); if (!binding.ok()) { return status::failure(binding.error()); }
    for (const auto & input : entry->inputs) {
        const size_t index = input.source;
        const buffer_view_const bytes = index == 1 ? buffer_view_const{&sequence, sizeof(sequence)} : index == 2 ? buffer_view_const{&maximum, sizeof(maximum)} :
            buffer_view_const{inputs[index].data(), inputs[index].size() * sizeof(fp16_bits)};
        auto bound = binding.value()->bind_input(input.id, bytes); if (!bound.ok()) { return bound; }
    }
    std::vector<std::vector<fp16_bits>> outputs(3); outputs[0].resize(queries * heads * width); outputs[1].resize(total * kv_heads * width); outputs[2].resize(total * kv_heads * width);
    for (size_t i = 0; i < 3; ++i) { auto bound = binding.value()->bind_output(entry->outputs[i], {outputs[i].data(), outputs[i].size() * sizeof(fp16_bits)}); if (!bound.ok()) { return bound; } }
    auto executed = entry->graph->execute(*binding.value()); if (!executed.ok()) { return executed; } ++stats_.executions;
    // GGML owns KV. Verify QNN cache outputs before publishing attention.
    for (size_t i = 1; i < 3; ++i) {
        const auto & expected = i == 1 ? packed_k.value() : packed_v.value();
        for (size_t head = 0; head < kv_heads; ++head) { for (size_t row = 0; row < total; ++row) { for (size_t col = 0; col < width; ++col) {
            if (outputs[i][(head * total + row) * width + col].bits != expected[(head * capacity + row) * width + col].bits) {
                return status::failure(conversion_error("GQA present cache differs from populated GGML KV"));
            }
        } } }
    }
    std::vector<uint8_t> bytes(dense_bytes(t));
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        const float value = ggml_fp16_to_fp32(outputs[0][i].bits); if (!std::isfinite(value)) { return status::failure(conversion_error("Nonfinite GQA output")); }
        std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(value));
    }
    pending.push_back({t, std::move(bytes)}); return status::success();
}

status backend_context::run_set_rows(ggml_tensor * t, std::vector<pending_output> & pending) {
    const auto * updates = t->src[0]; const auto * indices = t->src[1]; const auto * cache = t->src[2];
    const size_t rows = size_t(cache->ne[1]), count = size_t(indices->ne[0]), width = size_t(cache->ne[0]);
    std::vector<uint8_t> raw(count * ggml_type_size(indices->type)); ggml_backend_tensor_get(indices, raw.data(), 0, raw.size());
    std::vector<int32_t> positions(count); std::vector<bool> seen(rows, false);
    for (size_t i = 0; i < count; ++i) {
        int64_t index;
        if (indices->type == GGML_TYPE_I64) { std::memcpy(&index, raw.data() + i * 8, 8); }
        else { int32_t narrow; std::memcpy(&narrow, raw.data() + i * 4, 4); index = narrow; }
        if (index < 0 || uint64_t(index) >= rows || seen[size_t(index)]) { return status::failure(conversion_error("KV scatter index out of range or duplicate")); }
        positions[i] = int32_t(index); seen[size_t(index)] = true;
    }
    auto key = structural_key(t, false); key.push_back(cache->type); for (size_t i = 0; i < GGML_MAX_DIMS; ++i) { key.push_back(cache->ne[i]); key.push_back(cache->nb[i]); }
    prepared_run * entry = nullptr;
    for (size_t i = 0; i < cache_.size(); ++i) { if (cache_[i]->key == key) { auto hit = std::move(cache_[i]); cache_.erase(cache_.begin() + i); cache_.push_back(std::move(hit)); entry = cache_.back().get(); ++stats_.cache_hits; break; } }
    if (!entry) {
        if (!reserve_cache(0)) { return status::failure(conversion_error("KV scatter cache limit")); }
        auto context = context_owner::create(runtime_); if (!context.ok()) { return status::failure(context.error()); }
        auto graph = graph_owner::create(context.value(), "kv_scatter"); if (!graph.ok()) { return status::failure(graph.error()); }
        auto prepared = std::make_unique<prepared_run>(); prepared->key = std::move(key); prepared->graph = graph.value(); std::vector<tensor_id> ids;
        const std::vector<std::vector<uint32_t>> dims {{uint32_t(rows), uint32_t(width)}, {uint32_t(count), 1}, {uint32_t(count), uint32_t(width)}, {uint32_t(rows), uint32_t(width)}};
        for (size_t i = 0; i < 4; ++i) {
            auto id = graph.value()->register_tensor({"scatter" + std::to_string(i), i == 3 ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_APP_WRITE,
                i == 1 ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_FLOAT_16, dims[i]});
            if (!id.ok()) { return status::failure(id.error()); } ids.push_back(id.value()); if (i < 3) { prepared->inputs.push_back({0, i, id.value()}); }
        }
        auto added = graph.value()->add_node(op_spec::scatter_rows(ids[0], ids[1], ids[2], ids[3])); if (!added.ok()) { return added; }
        auto finalized = graph.value()->finalize(); if (!finalized.ok()) { return finalized; } prepared->outputs.push_back(ids[3]); cache_.push_back(std::move(prepared)); entry = cache_.back().get(); ++stats_.preparations;
    }
    auto data = pack(cache, pending); auto values = pack(updates, pending);
    if (!data.ok()) { return status::failure(data.error()); } if (!values.ok()) { return status::failure(values.error()); }
    auto binding = execution_binding::create(entry->graph); if (!binding.ok()) { return status::failure(binding.error()); }
    const buffer_view_const input[] {{data.value().data(), data.value().size() * 2}, {positions.data(), positions.size() * 4}, {values.value().data(), values.value().size() * 2}};
    for (size_t i = 0; i < 3; ++i) { auto bound = binding.value()->bind_input(entry->inputs[i].id, input[i]); if (!bound.ok()) { return bound; } }
    std::vector<fp16_bits> output(rows * width); auto bound = binding.value()->bind_output(entry->outputs[0], {output.data(), output.size() * 2}); if (!bound.ok()) { return bound; }
    auto executed = entry->graph->execute(*binding.value()); if (!executed.ok()) { return executed; } ++stats_.executions;
    for (size_t row = 0; row < rows; ++row) { if (!seen[row]) { for (size_t col = 0; col < width; ++col) {
        if (output[row * width + col].bits != data.value()[row * width + col].bits) { return status::failure(conversion_error("KV scatter modified an untouched row")); }
    } } }
    std::vector<uint8_t> bytes(output.size() * 2); std::memcpy(bytes.data(), output.data(), bytes.size()); pending.push_back({t, std::move(bytes)}); return status::success();
}

enum ggml_status backend_context::compute(ggml_cgraph * graph) {
    auto fail = [&](const qairt_error & e) {
        GGML_LOG_ERROR("qairt: %s: %s\n", e.operation.c_str(), e.message.c_str());
        cache_.clear();
        return GGML_STATUS_FAILED;
    };
    if (!graph || ggml_graph_n_nodes(graph) > 128) { return fail(conversion_error("Partition exceeds M2 node limit")); }
    const int count = ggml_graph_n_nodes(graph);
    auto ** nodes = ggml_graph_nodes(graph);
    size_t budget = 0, peak_workspace = 0;
    for (int i = 0; i < count; ++i) {
        const auto * t = nodes[i];
        if (!primitive_supported(t) || !readable_tensor(t) ||
            (t->op == GGML_OP_SOFT_MAX && !runtime_->config().experimental_softmax) ||
            (t->op == GGML_OP_ADD && !runtime_->config().experimental_add) ||
            (t->op == GGML_OP_ROPE && !runtime_->config().experimental_rope) || ((t->op == GGML_OP_FLASH_ATTN_EXT || t->op == GGML_OP_SET_ROWS) && !runtime_->config().experimental_attention)) {
            if (runtime_->config().verbose_diagnostics) {
                GGML_LOG_WARN("qairt: rejected operation %d, type %d during partition admission\n", int(t->op), int(t->type));
            }
            GGML_LOG_ERROR("qairt: admission failed for %s, op=%s, supported=%d, readable=%d\n", ggml_get_name(t), ggml_op_name(t->op), int(primitive_supported(t)), int(readable_tensor(t)));
            return fail(conversion_error("Unsupported operation, layout or output buffer"));
        }
        if (metadata_op(t->op)) { continue; }
        size_t bytes = dense_bytes(t) * 3;
        for (const auto * source : {t->src[0], t->src[1], (t->op == GGML_OP_ROPE || t->op == GGML_OP_FLASH_ATTN_EXT || t->op == GGML_OP_SET_ROWS) ? t->src[2] : nullptr, t->op == GGML_OP_FLASH_ATTN_EXT ? t->src[3] : nullptr}) {
            if (!source) { continue; }
            if (!readable_tensor(source)) { return fail(conversion_error("Unsupported source allocation")); }
            if (dense_bytes(source) > staging_limit / 4) { return fail(conversion_error("Source exceeds M2 staging budget")); }
            bytes += dense_bytes(source) * 4;
        }
        peak_workspace = std::max(peak_workspace, bytes);
        if (peak_workspace > staging_limit || dense_bytes(t) > staging_limit - peak_workspace || budget > staging_limit - peak_workspace - dense_bytes(t)) { return fail(conversion_error("Partition exceeds M2 staging budget")); }
        budget += dense_bytes(t);
    }
    std::vector<pending_output> pending;
    pending.reserve(count);
    for (int i = 0; i < count; ++i) {
        auto * t = nodes[i];
        if (metadata_op(t->op)) { continue; }
        if (t->op == GGML_OP_CONT || t->op == GGML_OP_DUP || t->op == GGML_OP_CPY) {
            auto bytes = materialize(t->src[0], t->type, pending);
            if (!bytes.ok()) { return fail(bytes.error()); }
            pending.push_back({t, std::move(bytes).value()});
            continue;
        }
        if (t->op == GGML_OP_ROPE) {
            auto rotated = run_rope(t, pending);
            if (!rotated.ok()) { return fail(rotated.error()); }
            continue;
        }
        if (t->op == GGML_OP_SET_ROWS) {
            auto executed = run_set_rows(t, pending); if (!executed.ok()) { return fail(executed.error()); } continue;
        }
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            auto executed = native_gqa_unavailable_ ? run_attention_decomposed(t, pending) : run_attention(t, pending);
            if (!executed.ok() && executed.error().stage == error_stage::op_validation && executed.error().qnn_error == QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND) {
                native_gqa_unavailable_ = true;
                GGML_LOG_WARN("qairt: native GroupQueryAttention unavailable; using QNN decomposition\n");
                executed = run_attention_decomposed(t, pending);
            }
            if (!executed.ok()) { return fail(executed.error()); }
            continue;
        }
        std::vector<ggml_tensor *> run {t};
        auto workspace = [&](const std::vector<ggml_tensor *> & group) {
            size_t peak = 0, dynamic = 0, outputs = 0;
            for (size_t n = 0; n < group.size(); ++n) {
                outputs += dense_bytes(group[n]) * 3;
                for (size_t j = 0; j < 2; ++j) {
                    const auto * source = group[n]->src[j]; if (!source) { continue; } bool internal = false;
                    for (size_t previous = 0; previous < n; ++previous) { internal |= group[previous] == source; }
                    if (internal) { continue; }
                    peak = std::max(peak, dense_bytes(source) * 4);
                    if (!(j == 0 && group[n]->op == GGML_OP_MUL_MAT && static_source(source, pending))) { dynamic += dense_bytes(source) * 2; }
                }
            }
            return peak + dynamic + outputs;
        };
        auto numerical = [](const ggml_tensor * node) { return node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL; };
        while (numerical(t) && run.size() < 16 && i + 1 < count && numerical(nodes[i + 1])) {
            auto * candidate = nodes[i + 1]; bool alias = false;
            for (const auto * source : {candidate->src[0], candidate->src[1]}) {
                for (const auto * previous : run) {
                    if (source == previous) {
                        if (candidate->op == GGML_OP_MUL_MAT && source == candidate->src[1] && (ggml_is_quantized(candidate->src[0]->type) || candidate->src[0]->type == GGML_TYPE_BF16) && source->type == GGML_TYPE_F32) { alias = true; }
                        continue;
                    }
                    const auto a = reinterpret_cast<uintptr_t>(source->data), b = reinterpret_cast<uintptr_t>(previous->data);
                    if ((a >= b && a - b < ggml_nbytes(previous)) || (b > a && b - a < ggml_nbytes(source))) { alias = true; }
                }
            }
            if (alias) { break; }
            run.push_back(candidate);
            if (workspace(run) > staging_limit - budget) { run.pop_back(); break; }
            ++i;
        }
        auto prepared = prepare_run(run, pending);
        if (!prepared.ok()) { return fail(prepared.error()); }
        auto * entry = prepared.value();
        std::vector<std::vector<fp16_bits>> inputs; inputs.reserve(entry->inputs.size());
        if (run[0]->op == GGML_OP_SOFT_MAX) {
            auto boundary = pack_softmax(run[0], pending);
            if (!boundary.ok()) { return fail(boundary.error()); }
            for (const auto & source : entry->inputs) {
                inputs.push_back(source.source == 0 ? std::move(boundary.value().logits) : std::move(boundary.value().mask));
            }
        } else {
            for (const auto & source : entry->inputs) {
                auto values = pack(run[source.node]->src[source.source], pending, source.preparation);
                if (!values.ok()) { return fail(values.error()); }
                inputs.push_back(std::move(values).value());
            }
        }
        std::vector<std::vector<fp16_bits>> outputs; outputs.reserve(run.size());
        for (const auto * node : run) { outputs.emplace_back(size_t(ggml_nelements(node))); }
        auto binding = execution_binding::create(entry->graph);
        if (!binding.ok()) { return fail(binding.error()); }
        for (size_t j = 0; j < inputs.size(); ++j) {
            auto bound = binding.value()->bind_input(entry->inputs[j].id, {inputs[j].data(), inputs[j].size() * sizeof(fp16_bits)});
            if (!bound.ok()) { return fail(bound.error()); }
        }
        for (size_t j = 0; j < outputs.size(); ++j) {
            auto bound = binding.value()->bind_output(entry->outputs[j], {outputs[j].data(), outputs[j].size() * sizeof(fp16_bits)});
            if (!bound.ok()) { return fail(bound.error()); }
        }
        auto executed = entry->graph->execute(*binding.value());
        if (!executed.ok()) {
            GGML_LOG_ERROR("qairt: failed dense run starting at %s, nodes=%zu, preparations=%llu, executions=%llu\n", ggml_get_name(t), run.size(),
                           (unsigned long long) stats_.preparations, (unsigned long long) stats_.executions);
            return fail(executed.error());
        }
        ++stats_.executions;
        for (size_t j = 0; j < run.size(); ++j) {
            auto * node = run[j]; std::vector<uint8_t> bytes(dense_bytes(node));
            for (size_t k = 0; k < size_t(ggml_nelements(node)); ++k) {
                const float value = ggml_fp16_to_fp32(outputs[j][k].bits);
                if (!std::isfinite(value)) { return fail(conversion_error("Nonfinite QNN output")); }
                if (node->type == GGML_TYPE_F32) { std::memcpy(bytes.data() + k * 4, &value, 4); }
                else { const auto half = ggml_fp32_to_fp16(value); std::memcpy(bytes.data() + k * 2, &half, 2); }
            }
            pending.push_back({node, std::move(bytes)});
        }
    }
    for (const auto & output : pending) { ggml_backend_tensor_set(output.tensor, output.bytes.data(), 0, output.bytes.size()); }
    if (runtime_->config().enable_profile) {
        auto events = runtime_->profile_event_count();
        if (events.ok()) { GGML_LOG_INFO("qairt: basic profile events: %u\n", events.value()); }
    }
    return GGML_STATUS_SUCCESS;
}

}
