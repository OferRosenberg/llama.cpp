#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>

namespace ggml::qairt {

bool float_layout(const ggml_tensor * tensor, bool allow_transpose = true) noexcept;
bool row_layout(const ggml_tensor * tensor) noexcept;
bool silu_op(const ggml_tensor * tensor) noexcept;
bool index_layout(const ggml_tensor * tensor) noexcept;
bool position_layout(const ggml_tensor * tensor) noexcept;
struct rope_parameters {
    int32_t dimensions, mode, context;
    float base, scale, extension, attention, beta_fast, beta_slow;
    bool truncate;
};
rope_parameters rope_params(const ggml_tensor * tensor) noexcept;
uint32_t row_height(uint32_t rows) noexcept;
const char * rowwise_type(enum ggml_op op) noexcept;
bool metadata_layout(const ggml_tensor * tensor) noexcept;
bool weight_layout(const ggml_tensor * tensor) noexcept;
bool attention_supported(const ggml_tensor * tensor) noexcept;
bool primitive_supported(const ggml_tensor * tensor) noexcept;
bool metadata_op(enum ggml_op op) noexcept;
const char * elementwise_type(enum ggml_op op) noexcept;
size_t dense_bytes(const ggml_tensor * tensor) noexcept;

}
