#pragma once

#include "ggml.h"
#include <cstddef>

namespace ggml::qairt {

bool float_layout(const ggml_tensor * tensor, bool allow_transpose = true) noexcept;
bool primitive_supported(const ggml_tensor * tensor) noexcept;
bool metadata_op(enum ggml_op op) noexcept;
const char * elementwise_type(enum ggml_op op) noexcept;
size_t dense_bytes(const ggml_tensor * tensor) noexcept;

}
