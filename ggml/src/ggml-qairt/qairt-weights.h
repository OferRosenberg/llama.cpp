#pragma once

#include "qairt-tensor.h"
#include "ggml.h"
#include <vector>

namespace ggml::qairt {

bool q4_weight_layout(const ggml_tensor * tensor) noexcept;
result<std::vector<fp16_bits>> prepare_q4_weights(array_view<const uint8_t> source, size_t width, size_t rows);

}
