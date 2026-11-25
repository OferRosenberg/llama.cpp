#include "ggml-cpu.h"
#include "qairt-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace ggml::qairt {
namespace {
constexpr size_t staging_limit = 256 * 1024 * 1024;
constexpr size_t cache_limit = 512;
constexpr uint64_t static_cache_limit = uint64_t(4) * 1024 * 1024 * 1024;

constexpr size_t partition_tensor_limit = 1024;
size_t partition_tensor_cost(const ggml_tensor * t) {
    if (metadata_op(t->op)) { return 0; }
    // Each decomposed head needs two slices with range tensors and a matmul output.
    if (t->op == GGML_OP_MUL_MAT) {
        const auto * input = t->src[1];
        const size_t heads = size_t(input->ne[2]);
        const size_t batches = size_t(input->ne[3]);
        if (heads > (partition_tensor_limit - 16) / 5 / batches) { return partition_tensor_limit + 1; }
        return 16 + 5 * heads * batches;
    }
    if (t->op == GGML_OP_FLASH_ATTN_EXT) { return 32 + 10 * size_t(t->src[0]->ne[2]); }
    // Include external inputs, view transformations and operation parameter tensors.
    if (t->op == GGML_OP_ROPE || t->op == GGML_OP_RMS_NORM || t->op == GGML_OP_NORM ||
        t->op == GGML_OP_SET_ROWS || t->op == GGML_OP_GET_ROWS) { return 16; }
    return 8;
}
qairt_error conversion_error(const char * text, const ggml_tensor * tensor = nullptr) {
    auto error = host_error(error_stage::numerical_check, "float preparation", text);
    if (tensor) { error.message += std::string(" (tensor ") + ggml_get_name(tensor) + ")"; }
    return error;
}

const ggml_tensor * storage_root(const ggml_tensor * t) {
    for (size_t depth = 0; t && t->view_src && depth < 32; ++depth) { t = t->view_src; }
    return t;
}
const pending_output * retained_storage(const ggml_tensor * t, const std::vector<pending_output> & pending) {
    for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
        const auto * producer = it->tensor;
        if (storage_root(t) != storage_root(producer) || t->type != producer->type || !ggml_is_contiguous(producer)) { continue; }
        const auto a = reinterpret_cast<uintptr_t>(t->data), b = reinterpret_cast<uintptr_t>(producer->data);
        if (a >= b && a - b <= ggml_nbytes(producer) && ggml_nbytes(t) <= ggml_nbytes(producer) - (a - b)) { return &*it; }
    }
    return nullptr;
}
std::vector<uint8_t> output_bytes(const pending_output & output) {
    std::vector<uint8_t> bytes(dense_bytes(output.tensor));
    if (output.tensor->type == GGML_TYPE_F16) { std::memcpy(bytes.data(), output.values->data(), bytes.size()); }
    else {
        for (size_t i = 0; i < output.values->size(); ++i) {
            const float value = ggml_fp16_to_fp32((*output.values)[i].bits);
            std::memcpy(bytes.data() + i * 4, &value, 4);
        }
    }
    return bytes;
}
void commit_outputs(const std::vector<pending_output> & pending, ggml_backend_qairt_stats & stats) {
    for (const auto & output : pending) {
        auto bytes = output_bytes(output);
        ggml_backend_tensor_set(output.tensor, bytes.data(), 0, bytes.size());
        ++stats.materialized_outputs;
    }
}
void mark_observable(const ggml_tensor * t, std::unordered_set<const ggml_tensor *> & observable) {
    for (size_t depth = 0; t && depth < 32; ++depth) {
        observable.insert(t);
        if (!metadata_op(t->op)) { break; }
        t = t->view_src;
    }
}
std::unordered_set<const ggml_tensor *> observable_outputs(const ggml_cgraph * graph) {
    std::unordered_map<const ggml_tensor *, int32_t> local_uses;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * t = graph->nodes[i];
        for (const auto * source : t->src) { if (source) { ++local_uses[source]; } }
    }
    std::unordered_set<const ggml_tensor *> observable;
    // Split graphs retain the complete graph's use counts; allocation dependencies are separate.
    for (size_t slot = 0; slot < graph->visited_hash_set.size; ++slot) {
        if (!ggml_bitset_get(graph->visited_hash_set.used, slot)) { continue; }
        const auto * t = graph->visited_hash_set.keys[slot];
        if (!t || !(t->flags & GGML_TENSOR_FLAG_COMPUTE)) { continue; }
        const int32_t uses = graph->use_counts ? graph->use_counts[slot] : 0;
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || uses == 0 || uses > local_uses[t]) { mark_observable(t, observable); }
    }
    return observable;
}

result<std::vector<uint8_t>> materialize(const ggml_tensor * t, enum ggml_type type,
                                         const std::vector<pending_output> & pending, bool validate = true) {
    using bytes_result = result<std::vector<uint8_t>>;
    if (!readable_tensor(t) || dense_bytes(t) > staging_limit) {
        return bytes_result::failure(conversion_error("Unsupported source buffer or staging size"));
    }
    const size_t source_element = t->type == GGML_TYPE_F32 ? 4 : 2;
    const size_t target_element = type == GGML_TYPE_F32 ? 4 : 2;
    const size_t count = size_t(ggml_nelements(t));
    if (count > staging_limit / target_element) { return bytes_result::failure(conversion_error("Conversion exceeds staging size")); }
    // A retained producer provides FP16 directly, including strided metadata views.
    if (const auto * output = retained_storage(t, pending)) {
        const size_t offset = (reinterpret_cast<uintptr_t>(t->data) - reinterpret_cast<uintptr_t>(output->tensor->data)) / source_element;
        std::vector<uint8_t> packed(count * target_element);
        for (size_t row = 0; row < count / size_t(t->ne[0]); ++row) {
            for (size_t col = 0; col < size_t(t->ne[0]); ++col) {
                const size_t outer = row / size_t(t->ne[1]);
                const size_t index = offset + (col * t->nb[0] + (row % size_t(t->ne[1])) * t->nb[1] +
                    (outer % size_t(t->ne[2])) * t->nb[2] + (outer / size_t(t->ne[2])) * t->nb[3]) / source_element;
                if (index >= output->values->size()) { return bytes_result::failure(conversion_error("Retained view exceeds producer")); }
                const auto half = (*output->values)[index].bits;
                const size_t out = (row * size_t(t->ne[0]) + col) * target_element;
                if (type == GGML_TYPE_F16) { std::memcpy(packed.data() + out, &half, 2); }
                else { const float value = ggml_fp16_to_fp32(half); std::memcpy(packed.data() + out, &value, 4); }
            }
        }
        return bytes_result::success(std::move(packed));
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    const auto address = reinterpret_cast<uintptr_t>(t->data);
    // Partial aliases need an overlay when no single producer covers the full view.
    for (const auto & output : pending) {
        if (storage_root(t) != storage_root(output.tensor) || t->type != output.tensor->type) { continue; }
        const auto base = reinterpret_cast<uintptr_t>(output.tensor->data);
        if (base >= address ? base - address >= raw.size() : address - base >= dense_bytes(output.tensor)) { continue; }
        auto bytes = output_bytes(output);
        if (base >= address) {
            const size_t offset = base - address;
            std::memcpy(raw.data() + offset, bytes.data(), std::min(bytes.size(), raw.size() - offset));
        } else {
            const size_t offset = address - base;
            std::memcpy(raw.data(), bytes.data() + offset, std::min(raw.size(), bytes.size() - offset));
        }
    }
    std::vector<uint8_t> packed(count * target_element);
    if (t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_1) {
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
            if (validate && (!std::isfinite(value) || (type == GGML_TYPE_F16 && std::fabs(value) > 65504.0f))) {
                return bytes_result::failure(conversion_error("Nonfinite or overflowing FP16 boundary value", t));
            }
            if (type == GGML_TYPE_F32) { std::memcpy(packed.data() + out, &value, 4); }
            else { const auto half = ggml_fp32_to_fp16(value); std::memcpy(packed.data() + out, &half, 2); }
        }
    }
    return bytes_result::success(std::move(packed));
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
    for (size_t i = 0; i < sizeof(t->op_params) / sizeof(uint32_t); ++i) {
        uint32_t word; std::memcpy(&word, reinterpret_cast<const uint8_t *>(t->op_params) + i * sizeof(word), sizeof(word)); key.push_back(word);
    }
    key.push_back(4); // FP16 connected recipe version.

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

bool pending_alias(const ggml_tensor * t, const std::vector<pending_output> & pending) {
    for (const auto & output : pending) { if (storage_root(t) == storage_root(output.tensor)) { return true; } }
    return false;
}
const fp16_bits * retained_fp16(const ggml_tensor * t, const std::vector<pending_output> & pending) {
    if (!ggml_is_contiguous(t)) { return nullptr; }
    const auto * output = retained_storage(t, pending);
    if (!output) { return nullptr; }
    const size_t element = ggml_type_size(t->type);
    const size_t offset = reinterpret_cast<uintptr_t>(t->data) - reinterpret_cast<uintptr_t>(output->tensor->data);
    return offset % element ? nullptr : output->values->data() + offset / element;
}

result<std::vector<fp16_bits>> backend_context::pack(const ggml_tensor * t, const std::vector<pending_output> & pending) const {
    using packed_result = result<std::vector<fp16_bits>>;
    auto bytes = numerical_bytes(t, pending);
    if (!bytes.ok()) { return packed_result::failure(bytes.error()); }
    std::vector<fp16_bits> values(bytes.value().size() / 2);
    std::memcpy(values.data(), bytes.value().data(), bytes.value().size());
    if (runtime_->config().validate_numerics) { for (const auto half : values) {
        if (!std::isfinite(ggml_fp16_to_fp32(half.bits))) { return packed_result::failure(conversion_error("Nonfinite FP16 input")); }
    } }
    return packed_result::success(std::move(values));
}

result<std::vector<uint8_t>> backend_context::numerical_bytes(const ggml_tensor * t, const std::vector<pending_output> & pending) const {
    return materialize(t, GGML_TYPE_F16, pending, runtime_->config().validate_numerics);
}
result<std::vector<uint8_t>> backend_context::mask_bytes(const ggml_tensor * t, const std::vector<pending_output> & pending) const {
    return materialize(t,t->type,pending);
}
result<std::vector<float>> backend_context::float_parameters(const ggml_tensor * t, const std::vector<pending_output> & pending) const {
    auto bytes = materialize(t, GGML_TYPE_F32, pending);
    if (!bytes.ok()) { return result<std::vector<float>>::failure(bytes.error()); }
    std::vector<float> values(bytes.value().size()/sizeof(float));
    std::memcpy(values.data(),bytes.value().data(),bytes.value().size());
    return result<std::vector<float>>::success(std::move(values));
}
backend_context::~backend_context() {
    if (cpu_backend_) { ggml_backend_free(cpu_backend_); }
}

bool backend_context::reserve_cache(size_t additional, const prepared_run * keep) {
    if (additional > static_cache_limit) { return false; }
    uint64_t retained = 0;
    for (const auto & entry : cache_) { retained += entry->static_bytes; }
    while ((!keep && cache_.size() >= cache_limit) || retained > static_cache_limit - additional) {
        // QNN graph resources live until contextFree; individual eviction cannot reclaim them.
        if (runtime_->config().shared_context) { return false; }
        auto victim = std::find_if(cache_.begin(), cache_.end(), [keep](const auto & entry) { return entry.get() != keep; });
        if (victim == cache_.end()) { return false; }
        retained -= (*victim)->static_bytes; cache_.erase(victim);
    }
    return true;
}

enum ggml_status backend_context::compute(ggml_cgraph * graph) {
    auto fail = [&](const qairt_error & e) {
        GGML_LOG_ERROR("qairt: %s: %s (QNN error %llu)\n", e.operation.c_str(), e.message.c_str(), (unsigned long long)e.qnn_error.value_or(0));
        invalidate();
        return GGML_STATUS_FAILED;
    };
    if (!graph) { return fail(conversion_error("Null compute graph")); }
    const int count = ggml_graph_n_nodes(graph);
    auto ** nodes = ggml_graph_nodes(graph);
    auto workspace = [](const ggml_tensor * t) {
        size_t bytes = dense_bytes(t) * 3;
        for (const auto * source : {t->src[0], t->src[1], (t->op == GGML_OP_ROPE || t->op == GGML_OP_FLASH_ATTN_EXT || t->op == GGML_OP_SET_ROWS) ? t->src[2] : nullptr, t->op == GGML_OP_FLASH_ATTN_EXT ? t->src[3] : nullptr}) {
            if (source) { bytes += dense_bytes(source) * 4; }
        }
        return bytes;
    };
    for (int i = 0; i < count; ++i) {
        const auto * t = nodes[i];
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE) || metadata_op(t->op)) { continue; }
        if (!primitive_supported(t) || !readable_tensor(t)) {
            GGML_LOG_ERROR("qairt: admission failed for %s, op=%s, supported=%d, readable=%d\n", ggml_get_name(t), ggml_op_name(t->op), int(primitive_supported(t)), int(readable_tensor(t)));
            return fail(conversion_error("Unsupported operation, layout or output buffer", t));
        }
        for (const auto * source : {t->src[0], t->src[1], (t->op == GGML_OP_ROPE || t->op == GGML_OP_FLASH_ATTN_EXT || t->op == GGML_OP_SET_ROWS) ? t->src[2] : nullptr, t->op == GGML_OP_FLASH_ATTN_EXT ? t->src[3] : nullptr}) {
            if (!source) { continue; }
            if (!readable_tensor(source)) { return fail(conversion_error("Unsupported source allocation", source)); }
            if (dense_bytes(source) > staging_limit / 4) { return fail(conversion_error("Source exceeds staging budget", source)); }
        }
        const size_t bytes = workspace(t);
        if (bytes > staging_limit || dense_bytes(t) > staging_limit - bytes) { return fail(conversion_error("Operation exceeds staging budget", t)); }
    }
    const auto observable = observable_outputs(graph);
    std::vector<pending_output> pending;
    pending.reserve(128);
    for (int i = 0; i < count;) {
        if (!(nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) || metadata_op(nodes[i]->op)) { ++i; continue; }
        const int partition_begin = i;
        size_t operations = 0;
        std::vector<ggml_tensor *> partition;
        size_t budget = 0, peak_workspace = 0, tensor_budget = 0;
        while (i < count && partition.size() < 128) {
            auto * candidate = nodes[i];
            if (!(candidate->flags & GGML_TENSOR_FLAG_COMPUTE)) { ++i; continue; }
            bool host_dependency = false;
            if (candidate->op == GGML_OP_ROPE && candidate->src[2]) {
                const auto * factor = candidate->src[2];
                for (size_t depth = 0; factor && depth < 32; ++depth) {
                    for (const auto * previous : partition) { host_dependency |= previous == factor; }
                    factor = factor->view_src;
                }
            }
            if (host_dependency) { break; }
            if (!metadata_op(candidate->op)) {
                const size_t tensors = partition_tensor_cost(candidate);
                if (tensors > partition_tensor_limit - tensor_budget) { break; }
                const size_t next_workspace = std::max(peak_workspace, workspace(candidate));
                const size_t bytes = dense_bytes(candidate);
                if (next_workspace > staging_limit || bytes > staging_limit - next_workspace || budget > staging_limit - next_workspace - bytes) { break; }
                peak_workspace = next_workspace;
                budget += bytes;
                tensor_budget += tensors;
                ++operations;
            }
            partition.push_back(candidate);
            ++i;
        }
        if (partition.empty()) { return fail(conversion_error("Cannot form a bounded partition")); }
        // A singleton cannot amortize QNN staging and graph dispatch.
        if (operations == 1) {
            commit_outputs(pending, stats_);
            pending.clear();
            if (!cpu_backend_) { cpu_backend_ = ggml_backend_cpu_init(); }
            if (!cpu_backend_) { return fail(conversion_error("CPU backend unavailable for singleton partition")); }
            auto cpu_graph = *graph;
            cpu_graph.nodes = graph->nodes + partition_begin;
            cpu_graph.n_nodes = i - partition_begin;
            const auto status = ggml_backend_graph_compute(cpu_backend_, &cpu_graph);
            if (status != GGML_STATUS_SUCCESS) { return fail(conversion_error("CPU singleton partition failed")); }
            for (auto * node : partition) {
                if (!metadata_op(node->op) && qairt_buffer(node->buffer)) {
                    static_cast<buffer_context *>(node->buffer->context)->touch();
                }
            }
            continue;
        }
        std::unordered_set<const ggml_tensor *> needed = observable;
        for (int next = i; next < count; ++next) {
            if (!(nodes[next]->flags & GGML_TENSOR_FLAG_COMPUTE)) { continue; }
            for (const auto * source : nodes[next]->src) { mark_observable(source, needed); }
        }
        // APP_READ checkpoints preserve FP16 rounding across QNN's arithmetic rewrites.
        for (const auto * node : partition) {
            if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_RMS_NORM &&
                node->op != GGML_OP_NORM && node->op != GGML_OP_ROPE) { continue; }
            needed.insert(node);
            for (const auto * source : {node->src[0], node->op == GGML_OP_MUL_MAT ? node->src[1] : nullptr}) {
                for (size_t depth = 0; source && depth < 32; ++depth) {
                    needed.insert(source);
                    if (metadata_op(source->op)) { source = source->view_src; }
                    else if (source->op == GGML_OP_CONT || source->op == GGML_OP_DUP) { source = source->src[0]; }
                    else { break; }
                }
            }
        }
        std::vector<bool> exports;
        exports.reserve(partition.size());
        for (const auto * node : partition) {
            exports.push_back(runtime_->config().validate_numerics || needed.count(node) ||
                node->op == GGML_OP_SET_ROWS || node->op == GGML_OP_CPY);
        }
        const auto executed = run_partition(partition, exports, pending);
        if (!executed.ok()) { return fail(executed.error()); }
    }
    // Materialize public/CPU exits and side effects; partition-only exits stay FP16.
    std::vector<pending_output> external;
    for (auto & output : pending) {
        if (runtime_->config().validate_numerics || observable.count(output.tensor) ||
            output.tensor->op == GGML_OP_SET_ROWS || output.tensor->op == GGML_OP_CPY) { external.push_back(std::move(output)); }
    }
    commit_outputs(external, stats_);
    if (runtime_->config().enable_profile) {
        auto events = runtime_->profile_event_count();
        if (events.ok()) { GGML_LOG_INFO("qairt: basic profile events: %u\n", events.value()); }
    }
    return GGML_STATUS_SUCCESS;
}

}
