#include "qairt-weights.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstring>
#include <limits>
namespace ggml::qairt {
namespace {
constexpr size_t preparation_limit = 256 * 1024 * 1024;
qairt_error invalid(const char * message) {
    return host_error(error_stage::numerical_check, "FP16 weight preparation", message);
}
bool valid_shape(size_t width, size_t rows) {
    return width && rows && width % 32 == 0 && width <= preparation_limit / 4 &&
           rows <= preparation_limit / 4 / width;
}
bool half_values(const std::vector<float> & row, fp16_bits * output) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (!std::isfinite(row[i]) || std::fabs(row[i]) > 65504.0f) { return false; }
        output[i].bits = ggml_fp32_to_fp16(row[i]);
    }
    return true;
}
}

bool q4_weight_layout(const ggml_tensor * t) noexcept {
    return t && t->type == GGML_TYPE_Q4_0 && t->ne[0] > 0 && t->ne[1] > 0 &&
           t->ne[2] == 1 && t->ne[3] == 1 && valid_shape(size_t(t->ne[0]), size_t(t->ne[1])) &&
           t->nb[0] == ggml_type_size(GGML_TYPE_Q4_0) &&
           t->nb[1] == ggml_row_size(GGML_TYPE_Q4_0, t->ne[0]);
}

result<std::vector<fp16_bits>> prepare_q4_weights(array_view<const uint8_t> source, size_t width, size_t rows) {
    using output_t = result<std::vector<fp16_bits>>;
    if (!valid_shape(width, rows)) { return output_t::failure(invalid("Invalid Q4_0 shape")); }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, int64_t(width));
    if (source.size() != row_bytes * rows || !source.data()) { return output_t::failure(invalid("Invalid Q4_0 payload")); }
    std::vector<fp16_bits> output(width * rows);
    std::vector<float> scratch(width);
    const auto decode = ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float;
    for (size_t row = 0; row < rows; ++row) {
        decode(source.data() + row * row_bytes, scratch.data(), int64_t(width));
        if (!half_values(scratch, output.data() + row * width)) {
            return output_t::failure(invalid("Nonfinite or overflowing Q4_0 weight"));
        }
    }
    return output_t::success(std::move(output));
}

}
