#include "qairt-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

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
    if (t->type == GGML_TYPE_Q4_0) {
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

enum ggml_status backend_context::compute(ggml_cgraph * graph) {
    auto fail = [&](const qairt_error & e) {
        GGML_LOG_ERROR("qairt: %s: %s (QNN error %llu)\n", e.operation.c_str(), e.message.c_str(), (unsigned long long)e.qnn_error.value_or(0));
        cache_.clear();
        return GGML_STATUS_FAILED;
    };
    if (!graph || ggml_graph_n_nodes(graph) > 128) { return fail(conversion_error("Partition exceeds M2 node limit")); }
    const int count = ggml_graph_n_nodes(graph);
    auto ** nodes = ggml_graph_nodes(graph);
    size_t budget = 0, peak_workspace = 0;
    for (int i = 0; i < count; ++i) {
        const auto * t = nodes[i];
        if (metadata_op(t->op)) { continue; }
        if (!primitive_supported(t) || !readable_tensor(t)) {
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
        std::vector<ggml_tensor *> partition {t};
        while (i + 1 < count) {
            const auto * candidate=nodes[i+1];bool host_dependency=false;
            if(candidate->op==GGML_OP_ROPE && candidate->src[2]) {
                const auto * factor=candidate->src[2];
                for(size_t depth=0;factor && depth<32;++depth){
                    for(const auto * previous:partition){host_dependency|=previous==factor;}
                    factor=factor->view_src;
                }
            }
            if(host_dependency){break;}
            partition.push_back(nodes[++i]);
        }
        const auto executed = run_partition(partition, pending);
        if (!executed.ok()) { return fail(executed.error()); }
    }
    for (const auto & output : pending) { ggml_backend_tensor_set(output.tensor, output.bytes.data(), 0, output.bytes.size()); }
    if (runtime_->config().enable_profile) {
        auto events = runtime_->profile_event_count();
        if (events.ok()) { GGML_LOG_INFO("qairt: basic profile events: %u\n", events.value()); }
    }
    return GGML_STATUS_SUCCESS;
}

}
