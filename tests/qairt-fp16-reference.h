#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Test-only FP16 storage semantics with FP32 CPU arithmetic.
class fp16_reference {
public:
    explicit fp16_reference(size_t cache_limit = 256 * 1024 * 1024) : cache_limit_(cache_limit), cpu_(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), ggml_backend_free) {
        require(bool(cpu_), "Cannot initialize reference CPU");
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu_.get()));
        const auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (set_threads) { set_threads(cpu_.get(), 8); }
    }
    static float round(float value) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)); }
    static size_t offset(const ggml_tensor * t, size_t index) {
        size_t result = 0;
        for (int d = 0; d < 4; ++d) { result += (index % size_t(t->ne[d])) * t->nb[d]; index /= size_t(t->ne[d]); }
        return result;
    }
    static std::vector<float> read(const ggml_tensor * t, bool rounded = true) {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::vector<float> values(size_t(ggml_nelements(t)));
        if (ggml_is_quantized(t->type)) {
            require(t->nb[0] == ggml_type_size(t->type), "Unsupported quantized reference stride");
            const auto decode = ggml_get_type_traits(t->type)->to_float;
            require(decode != nullptr, "Missing reference weight decoder");
            const size_t width = size_t(t->ne[0]);
            for (size_t row = 0; row < values.size() / width; ++row) {
                decode(raw.data() + offset(t, row * width), values.data() + row * width, t->ne[0]);
            }
        } else if (ggml_is_contiguous(t) && t->type == GGML_TYPE_F32) {
            std::memcpy(values.data(), raw.data(), values.size() * sizeof(float));
        } else if (ggml_is_contiguous(t) && t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(raw.data()), values.data(), int64_t(values.size()));
        } else {
            require(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16, "Unsupported reference float type");
            for (size_t i = 0; i < values.size(); ++i) {
                const auto * p = raw.data() + offset(t, i);
                if (t->type == GGML_TYPE_F32) { std::memcpy(&values[i], p, sizeof(float)); }
                else { ggml_fp16_t bits; std::memcpy(&bits, p, sizeof(bits)); values[i] = ggml_fp16_to_fp32(bits); }
            }
        }
        if (rounded) { for (auto & value : values) { value = round(value); } }
        return values;
    }
    static void write(ggml_tensor * t, const std::vector<float> & values) {
        require(values.size() == size_t(ggml_nelements(t)), "Reference output shape mismatch");
        require(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16, "Unsupported reference output type");
        if (ggml_is_contiguous(t)) {
            if (t->type == GGML_TYPE_F32) {
                std::vector<float> rounded(values.size());
                for (size_t i = 0; i < values.size(); ++i) { rounded[i] = round(values[i]); }
                ggml_backend_tensor_set(t, rounded.data(), 0, rounded.size() * sizeof(float));
            } else {
                std::vector<ggml_fp16_t> half(values.size());
                ggml_fp32_to_fp16_row(values.data(), half.data(), int64_t(values.size()));
                ggml_backend_tensor_set(t, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
            }
            return;
        }
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        for (size_t i = 0; i < values.size(); ++i) {
            auto * p = raw.data() + offset(t, i);
            if (t->type == GGML_TYPE_F32) { const float value = round(values[i]); std::memcpy(p, &value, sizeof(value)); }
            else { const auto bits = ggml_fp32_to_fp16(values[i]); std::memcpy(p, &bits, sizeof(bits)); }
        }
        ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
    }
    std::vector<float> compute(const ggml_tensor * t) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        require(bool(ctx), "Cannot allocate reference context");
        auto * output = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, t->ne);
        output->op = t->op; output->flags = GGML_TENSOR_FLAG_COMPUTE;
        std::memcpy(output->op_params, t->op_params, sizeof(output->op_params));
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> integers;
        std::vector<std::pair<ggml_tensor *, std::vector<float>>> floats;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const auto * source = t->src[s];
            if (!source) { continue; }
            const bool integer = source->type == GGML_TYPE_I32 || source->type == GGML_TYPE_I64;
            auto * copy = ggml_new_tensor(ctx.get(), integer ? source->type : GGML_TYPE_F32, 4, source->ne);
            output->src[s] = copy;
            if (integer) {
                std::vector<uint8_t> raw(ggml_nbytes(source)), dense(ggml_nbytes(copy));
                ggml_backend_tensor_get(source, raw.data(), 0, raw.size());
                const size_t size = ggml_type_size(source->type);
                for (size_t i = 0; i < size_t(ggml_nelements(source)); ++i) { std::memcpy(dense.data() + i * size, raw.data() + offset(source, i), size); }
                integers.push_back({copy, std::move(dense)});
            } else {
                const std::string name = ggml_get_name(source);
                const bool weight = source->op == GGML_OP_NONE && name.find(".weight") != std::string::npos;
                // RoPE frequency factors are host parameters, not FP16 activations.
                if (t->op == GGML_OP_ROPE && s == 2) { floats.push_back({copy, read(source, false)}); }
                else if (weight) {
                    auto found = weights_.find(source);
                    if (found == weights_.end()) {
                        const auto values = read(source, false);
                        std::vector<ggml_fp16_t> half(values.size());
                        ggml_fp32_to_fp16_row(values.data(), half.data(), int64_t(values.size()));
                        const size_t bytes = half.size() * sizeof(ggml_fp16_t);
                        while (!weights_.empty() && cached_bytes_ + bytes > cache_limit_) {
                            cached_bytes_ -= weights_.begin()->second.size() * sizeof(ggml_fp16_t); weights_.erase(weights_.begin());
                        }
                        found = weights_.emplace(source, std::move(half)).first; cached_bytes_ += bytes;
                    }
                    std::vector<float> values(found->second.size());
                    ggml_fp16_to_fp32_row(found->second.data(), values.data(), int64_t(values.size()));
                    floats.push_back({copy, std::move(values)});
                } else { floats.push_back({copy, read(source)}); }
            }
        }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu_.get()), ggml_backend_buffer_free);
        require(bool(buffer), "Cannot allocate reference tensors");
        for (const auto & value : integers) { ggml_backend_tensor_set(value.first, value.second.data(), 0, value.second.size()); }
        for (const auto & value : floats) { ggml_backend_tensor_set(value.first, value.second.data(), 0, value.second.size() * sizeof(float)); }
        if (t->op == GGML_OP_SET_ROWS) {
            require(output->src[2] != nullptr, "Missing reference KV state");
            std::memcpy(output->data, output->src[2]->data, ggml_nbytes(output));
        }
        if (t->op == GGML_OP_GLU) {
            require(ggml_get_glu_op(t) == GGML_GLU_OP_SWIGLU && output->src[1], "Unsupported reference GLU");
            const auto * x = static_cast<const float *>(output->src[0]->data);
            const auto * up = static_cast<const float *>(output->src[1]->data);
            std::vector<float> result(size_t(ggml_nelements(t)));
            for (size_t i = 0; i < result.size(); ++i) {
                const float sigmoid = round(1.0f / (1.0f + std::exp(-x[i])));
                result[i] = round(round(x[i] * sigmoid) * up[i]);
            }
            return result;
        }
        if (t->op == GGML_OP_SOFT_MAX) {
            float scale, max_bias; std::memcpy(&scale, t->op_params, 4); std::memcpy(&max_bias, t->op_params + 1, 4);
            require(max_bias == 0.0f && !t->src[2], "Unsupported reference Softmax bias");
            auto * logits = static_cast<float *>(output->src[0]->data);
            const auto * mask = output->src[1];
            const auto * mask_values = mask ? static_cast<const float *>(mask->data) : nullptr;
            for (size_t i = 0; i < size_t(ggml_nelements(t)); ++i) {
                float value = scale == 1.0f ? logits[i] : round(logits[i] * round(scale));
                if (mask) {
                    size_t remaining = i, mask_index = 0, stride = 1;
                    for (int d = 0; d < 4; ++d) {
                        const size_t coordinate = remaining % size_t(t->ne[d]); remaining /= size_t(t->ne[d]);
                        mask_index += (coordinate % size_t(mask->ne[d])) * stride; stride *= size_t(mask->ne[d]);
                    }
                    value = round(value + mask_values[mask_index]);
                }
                logits[i] = value;
            }
            const float one = 1.0f; std::memcpy(output->op_params, &one, 4); output->src[1] = nullptr;
        }
        auto * graph = ggml_new_graph(ctx.get()); ggml_graph_add_node(graph, output);
        require(ggml_backend_supports_op(cpu_.get(), output), "CPU cannot execute emulated reference operator");
        require(ggml_backend_graph_compute(cpu_.get(), graph) == GGML_STATUS_SUCCESS, "Reference operator execution failed");
        return read(output);
    }
    size_t cached_weights() const { return weights_.size(); }
private:
    static void require(bool condition, const char * message) { if (!condition) { throw std::runtime_error(message); } }
    size_t cache_limit_;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu_;
    std::map<const ggml_tensor *, std::vector<ggml_fp16_t>> weights_;
    size_t cached_bytes_ = 0;
};

// Capture only reference-context CPU backends; the hybrid backend stays unchanged.
class fp16_reference_execution {
public:
    explicit fp16_reference_execution(int layer, int layers = 1) : reference_(layers > 1 ? size_t(8) * 1024 * 1024 * 1024 : 256 * 1024 * 1024), layer_(layer), first_(layer - layers + 1), device_(ggml_backend_get_device(cpu_device_backend())), init_(device_->iface.init_backend) {
        active_ = this; device_->iface.init_backend = initialize;
    }
    ~fp16_reference_execution() {
        stop_capture();
        for (auto & entry : original_) { entry.first->iface = entry.second; }
        active_ = nullptr;
    }
    void stop_capture() { device_->iface.init_backend = init_; }
    fp16_reference & reference() { return reference_; }
    size_t nodes = 0;
private:
    ggml_backend_t cpu_device_backend() {
        probe_.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        if (!probe_) { throw std::runtime_error("Missing CPU reference device"); }
        return probe_.get();
    }
    bool selected(const ggml_tensor * t, int depth = 0) const {
        if (!t || depth > 6) { return false; }
        const std::string name = ggml_get_name(t), weight = "blk." + std::to_string(layer_) + ".";
        for (int layer = first_; layer <= layer_; ++layer) {
            if (name.find("blk." + std::to_string(layer) + ".") == 0) { return true; }
        }
        if (t->op == GGML_OP_GET_ROWS && t->src[0] && std::string(ggml_get_name(t->src[0])) == "l_out-" + std::to_string(layer_ - 1)) { return true; }
        const auto dash = name.rfind('-');
        if (dash != std::string::npos && dash + 1 < name.size() && name[dash + 1] >= '0' && name[dash + 1] <= '9') {
            char * end = nullptr; const long layer = std::strtol(name.c_str() + dash + 1, &end, 10);
            if (*end == '\0' || *end == ' ') { return layer >= first_ && layer <= layer_; }
        }
        const auto cache = name.find("_l");
        if (cache != std::string::npos) {
            char * end = nullptr; const long layer = std::strtol(name.c_str() + cache + 2, &end, 10);
            if (end != name.c_str() + cache + 2) { return layer >= first_ && layer <= layer_; }
        }
        for (const auto * source : t->src) { if (selected(source, depth + 1)) { return true; } }
        return false;
    }
    bool emulated(const ggml_tensor * t) const {
        if (!selected(t) || std::string(ggml_get_name(t)) == "result_output") { return false; }
        switch (t->op) {
            case GGML_OP_MUL_MAT:
                return !ggml_is_quantized(t->src[0]->type) || t->src[0]->type == GGML_TYPE_Q4_0;
            case GGML_OP_RMS_NORM: case GGML_OP_MUL: case GGML_OP_ADD: case GGML_OP_ROPE:
            case GGML_OP_SOFT_MAX: case GGML_OP_GLU: case GGML_OP_SET_ROWS: case GGML_OP_GET_ROWS:
            case GGML_OP_CONT: return true;
            default: return false;
        }
    }
    static ggml_backend_t initialize(ggml_backend_dev_t device, const char * params) {
        auto * backend = active_->init_(device, params);
        if (backend) { active_->original_[backend] = backend->iface; backend->iface.graph_compute = compute; backend->iface.free = release; }
        return backend;
    }
    static ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph) {
        try {
            for (int begin = 0; begin < ggml_graph_n_nodes(graph);) {
                auto * t = ggml_graph_node(graph, begin);
                if (active_->emulated(t)) {
                    fp16_reference::write(t, active_->reference_.compute(t)); ++active_->nodes; ++begin;
                } else {
                    int end = begin + 1;
                    while (end < ggml_graph_n_nodes(graph) && !active_->emulated(ggml_graph_node(graph, end))) { ++end; }
                    auto view = *graph; view.n_nodes = end - begin; view.nodes = graph->nodes + begin;
                    const auto status = active_->original_.at(backend).graph_compute(backend, &view);
                    if (status != GGML_STATUS_SUCCESS) { return status; }
                    begin = end;
                }
            }
            return GGML_STATUS_SUCCESS;
        } catch (const std::exception & error) {
            std::fprintf(stderr, "FP16 reference failure: %s\n", error.what()); return GGML_STATUS_FAILED;
        }
    }
    static void release(ggml_backend_t backend) {
        const auto original = active_->original_.at(backend); active_->original_.erase(backend); original.free(backend);
    }
    inline static fp16_reference_execution * active_ = nullptr;
    fp16_reference reference_;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> probe_ {nullptr, ggml_backend_free};
    int layer_;
    int first_;
    ggml_backend_dev_t device_;
    decltype(ggml_backend_device_i::init_backend) init_;
    std::map<ggml_backend_t, ggml_backend_i> original_;
};
