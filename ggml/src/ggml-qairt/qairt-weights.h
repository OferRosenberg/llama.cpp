#pragma once

#include "qairt-tensor.h"
#include "ggml.h"
#include <vector>

namespace ggml::qairt {

bool q4_weight_layout(const ggml_tensor * tensor) noexcept;
result<std::vector<fp16_bits>> prepare_q4_weights(array_view<const uint8_t> source, size_t width, size_t rows, ggml_type type = GGML_TYPE_Q4_0);

struct w4_weights {
    size_t width = 0, rows = 0;
    std::vector<int8_t> codes;
    std::shared_ptr<const block_quantization> quantization;
    size_t bytes() const noexcept { return codes.size() + quantization->scales.size() * sizeof(Qnn_FloatScaleOffset_t); }
};
result<w4_weights> prepare_w4_weights(array_view<const uint8_t> source, size_t width, size_t rows, ggml_type type = GGML_TYPE_Q4_0);

class w4_weight_upload final {
public:
    status update(array_view<const uint8_t> source, size_t width, size_t rows, size_t offset, size_t bytes, ggml_type type = GGML_TYPE_Q4_0);
    std::shared_ptr<const w4_weights> get(size_t width, size_t rows, ggml_type type = GGML_TYPE_Q4_0) const noexcept;
    uint64_t preparations() const noexcept { return preparations_; }
private:
    size_t width_ = 0, rows_ = 0;
    ggml_type type_ = GGML_TYPE_Q4_0;
    bool loaded_ = false;
    uint64_t preparations_ = 0;
    std::vector<std::pair<size_t,size_t>> ranges_;
    std::shared_ptr<const w4_weights> payload_;
};

}
