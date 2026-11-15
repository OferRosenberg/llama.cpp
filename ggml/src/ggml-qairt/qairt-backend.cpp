#include "qairt-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace ggml::qairt {
namespace {
constexpr size_t staging_limit = 256 * 1024 * 1024;
constexpr size_t cache_limit = 16;

qairt_error conversion_error(const char * text) { return host_error(error_stage::numerical_check, "float preparation", text); }

result<std::vector<uint8_t>> materialize(const ggml_tensor * t, enum ggml_type type,
                                         const std::vector<pending_output> & pending) {
    using bytes_result = result<std::vector<uint8_t>>;
    if (!readable_tensor(t) || dense_bytes(t) > staging_limit) {
        return bytes_result::failure(conversion_error("Unsupported source buffer or staging size"));
    }
    const size_t source_element = t->type == GGML_TYPE_F32 ? 4 : 2;
    const size_t target_element = type == GGML_TYPE_F32 ? 4 : 2;
    const size_t count = size_t(t->ne[0]) * size_t(t->ne[1]);
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
    for (size_t row = 0; row < size_t(t->ne[1]); ++row) {
        for (size_t col = 0; col < size_t(t->ne[0]); ++col) {
            const size_t offset = row * t->nb[1] + col * t->nb[0];
            const size_t out = (row * size_t(t->ne[0]) + col) * target_element;
            if (offset > raw.size() || source_element > raw.size() - offset) { return bytes_result::failure(conversion_error("Source stride exceeds allocation")); }
            if (t->type == type) { std::memcpy(packed.data() + out, raw.data() + offset, source_element); continue; }
            float value;
            if (t->type == GGML_TYPE_F32) { std::memcpy(&value, raw.data() + offset, 4); }
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
        key.push_back(value->type);
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) { key.push_back(value->ne[i]); key.push_back(value->nb[i]); }
    }
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

result<std::vector<fp16_bits>> backend_context::pack(const ggml_tensor * t, const std::vector<pending_output> & pending) const {
    using packed_result = result<std::vector<fp16_bits>>;
    auto bytes = materialize(t, GGML_TYPE_F16, pending);
    if (!bytes.ok()) { return packed_result::failure(bytes.error()); }
    std::vector<fp16_bits> values(bytes.value().size() / 2);
    std::memcpy(values.data(), bytes.value().data(), bytes.value().size());
    for (const auto half : values) {
        if (!std::isfinite(ggml_fp16_to_fp32(half.bits))) { return packed_result::failure(conversion_error("Nonfinite FP16 input")); }
    }
    return packed_result::success(std::move(values));
}

result<prepared_primitive *> backend_context::prepare(const ggml_tensor * t, const std::vector<pending_output> & pending) {
    using prepared_result = result<prepared_primitive *>;
    const bool constant = t->op == GGML_OP_MUL_MAT && static_source(t->src[0], pending);
    auto key = structural_key(t, constant);
    for (size_t i = 0; i < cache_.size(); ++i) {
        if (cache_[i]->key == key) {
            auto entry = std::move(cache_[i]);
            cache_.erase(cache_.begin() + i);
            cache_.push_back(std::move(entry));
            ++stats_.cache_hits;
            return prepared_result::success(cache_.back().get());
        }
    }
    if (cache_.size() == cache_limit) { cache_.erase(cache_.begin()); }
    auto context = context_owner::create(runtime_);
    if (!context.ok()) { return prepared_result::failure(context.error()); }
    auto graph = graph_owner::create(context.value(), "primitive");
    if (!graph.ok()) { return prepared_result::failure(graph.error()); }
    auto entry = std::make_unique<prepared_primitive>();
    entry->key = std::move(key); entry->graph = graph.value();
    const ggml_tensor * sources[2] = { t->src[0], t->src[1] };
    if (t->op == GGML_OP_MUL_MAT) { std::swap(sources[0], sources[1]); }
    tensor_id ids[2];
    for (size_t i = 0; i < 2; ++i) {
        const bool is_static = constant && i == 1;
        tensor_spec spec { i == 0 ? "A" : "B", is_static ? QNN_TENSOR_TYPE_STATIC : QNN_TENSOR_TYPE_APP_WRITE,
                           QNN_DATATYPE_FLOAT_16, {uint32_t(sources[i]->ne[1]), uint32_t(sources[i]->ne[0])} };
        result<tensor_id> registered = result<tensor_id>::failure(conversion_error("Tensor not registered"));
        if (is_static) {
            auto bytes = pack(sources[i], pending);
            if (!bytes.ok()) { return prepared_result::failure(bytes.error()); }
            registered = graph.value()->register_tensor(std::move(spec), {bytes.value().data(), bytes.value().size() * 2});
        } else { registered = graph.value()->register_tensor(std::move(spec)); }
        if (!registered.ok()) { return prepared_result::failure(registered.error()); }
        ids[i] = registered.value();
        if (!is_static) { entry->input_ids.push_back(ids[i]); }
    }
    auto y = graph.value()->register_tensor({"Y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16,
                                           {uint32_t(t->ne[1]), uint32_t(t->ne[0])}});
    if (!y.ok()) { return prepared_result::failure(y.error()); }
    entry->output_id = y.value();
    op_spec spec;
    if (t->op == GGML_OP_MUL_MAT) { spec = op_spec::matmul(ids[0], ids[1], y.value()); }
    else { spec = {"elementwise", "qti.aisw", elementwise_type(t->op), {ids[0], ids[1]}, {y.value()}, {}}; }
    auto added = graph.value()->add_node(spec);
    if (!added.ok()) { return prepared_result::failure(added.error()); }
    auto finalized = graph.value()->finalize();
    if (!finalized.ok()) { return prepared_result::failure(finalized.error()); }
    cache_.push_back(std::move(entry));
    ++stats_.preparations;
    return prepared_result::success(cache_.back().get());
}

enum ggml_status backend_context::compute(ggml_cgraph * graph) {
    auto fail = [&](const qairt_error & e) {
        GGML_LOG_ERROR("qairt: %s: %s\n", e.operation.c_str(), e.message.c_str());
        cache_.clear();
        return GGML_STATUS_FAILED;
    };
    if (!graph || ggml_graph_n_nodes(graph) > 128) { return fail(conversion_error("Partition exceeds M1 node limit")); }
    const int count = ggml_graph_n_nodes(graph);
    auto ** nodes = ggml_graph_nodes(graph);
    size_t budget = 0;
    for (int i = 0; i < count; ++i) {
        const auto * t = nodes[i];
        if (!primitive_supported(t) || !readable_tensor(t)) {
            if (runtime_->config().verbose_diagnostics) {
                GGML_LOG_WARN("qairt: rejected operation %d, type %d during partition admission\n", int(t->op), int(t->type));
            }
            return fail(conversion_error("Unsupported operation, layout or output buffer"));
        }
        if (metadata_op(t->op)) { continue; }
        size_t bytes = dense_bytes(t) * 3;
        for (const auto * source : {t->src[0], t->src[1]}) {
            if (!source) { continue; }
            if (!readable_tensor(source)) { return fail(conversion_error("Unsupported source allocation")); }
            if (dense_bytes(source) > staging_limit / 4) { return fail(conversion_error("Source exceeds M1 staging budget")); }
            bytes += dense_bytes(source) * 4;
        }
        if (bytes > staging_limit || budget > staging_limit - bytes) { return fail(conversion_error("Partition exceeds M1 staging budget")); }
        budget += bytes;
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
        auto prepared = prepare(t, pending);
        if (!prepared.ok()) { return fail(prepared.error()); }
        auto * entry = prepared.value();
        const ggml_tensor * sources[2] = {t->src[0], t->src[1]};
        if (t->op == GGML_OP_MUL_MAT) { std::swap(sources[0], sources[1]); }
        std::vector<std::vector<fp16_bits>> inputs;
        inputs.reserve(entry->input_ids.size());
        for (size_t j = 0; j < entry->input_ids.size(); ++j) {
            auto values = pack(sources[j], pending);
            if (!values.ok()) { return fail(values.error()); }
            inputs.push_back(std::move(values).value());
        }
        std::vector<fp16_bits> output(size_t(t->ne[0]) * size_t(t->ne[1]));
        auto binding = execution_binding::create(entry->graph);
        if (!binding.ok()) { return fail(binding.error()); }
        for (size_t j = 0; j < inputs.size(); ++j) {
            auto bound = binding.value()->bind_input(entry->input_ids[j], array_view<const fp16_bits>(inputs[j].data(), inputs[j].size()));
            if (!bound.ok()) { return fail(bound.error()); }
        }
        auto bound = binding.value()->bind_output(entry->output_id, array_view<fp16_bits>(output.data(), output.size()));
        if (!bound.ok()) { return fail(bound.error()); }
        auto executed = entry->graph->execute(*binding.value());
        if (!executed.ok()) { return fail(executed.error()); }
        ++stats_.executions;
        std::vector<uint8_t> bytes(dense_bytes(t));
        for (size_t j = 0; j < output.size(); ++j) {
            const float value = ggml_fp16_to_fp32(output[j].bits);
            if (!std::isfinite(value)) { return fail(conversion_error("Nonfinite QNN output")); }
            if (t->type == GGML_TYPE_F32) { std::memcpy(bytes.data() + j * 4, &value, 4); }
            else { std::memcpy(bytes.data() + j * 2, &output[j].bits, 2); }
        }
        pending.push_back({t, std::move(bytes)});
    }
    for (const auto & output : pending) { ggml_backend_tensor_set(output.tensor, output.bytes.data(), 0, output.bytes.size()); }
    if (runtime_->config().enable_profile) {
        auto events = runtime_->profile_event_count();
        if (events.ok()) { GGML_LOG_INFO("qairt: basic profile events: %u\n", events.value()); }
    }
    return GGML_STATUS_SUCCESS;
}

}
