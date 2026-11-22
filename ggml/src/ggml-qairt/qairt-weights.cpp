#include "qairt-weights.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
namespace ggml::qairt {
namespace {
constexpr size_t preparation_limit = 256 * 1024 * 1024;
qairt_error invalid(const char * message) {
    return host_error(error_stage::numerical_check, "FP16 weight preparation", message);
}
bool supported_type(ggml_type type) { return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1; }
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
    return t && supported_type(t->type) && t->ne[0] > 0 && t->ne[1] > 0 &&
           t->ne[2] == 1 && t->ne[3] == 1 && valid_shape(size_t(t->ne[0]), size_t(t->ne[1])) &&
           t->nb[0] == ggml_type_size(t->type) &&
           t->nb[1] == ggml_row_size(t->type, t->ne[0]);
}

result<std::vector<fp16_bits>> prepare_q4_weights(array_view<const uint8_t> source, size_t width, size_t rows, ggml_type type) {
    using output_t = result<std::vector<fp16_bits>>;
    if (!supported_type(type) || !valid_shape(width, rows)) { return output_t::failure(invalid("Invalid Q4 shape")); }
    const size_t row_bytes = ggml_row_size(type, int64_t(width));
    if (source.size() != row_bytes * rows || !source.data()) { return output_t::failure(invalid("Invalid Q4 payload")); }
    std::vector<fp16_bits> output(width * rows);
    std::vector<float> scratch(width);
    const auto decode = ggml_get_type_traits(type)->to_float;
    for (size_t row = 0; row < rows; ++row) {
        decode(source.data() + row * row_bytes, scratch.data(), int64_t(width));
        if (!half_values(scratch, output.data() + row * width)) {
            return output_t::failure(invalid("Nonfinite or overflowing Q4 weight"));
        }
    }
    return output_t::success(std::move(output));
}

result<w4_weights> prepare_w4_weights(array_view<const uint8_t> source, size_t width, size_t rows, ggml_type type) {
    using output_t = result<w4_weights>;
    if (!supported_type(type) || !valid_shape(width, rows) || !source.data() || source.size() != width / 32 * ggml_type_size(type) * rows) {
        return output_t::failure(invalid("Invalid W4FP16 shape or payload"));
    }
    w4_weights output; output.width = width; output.rows = rows; output.codes.resize(width * rows);
    auto quantization = std::make_shared<block_quantization>();
    quantization->block_size = {1,1,1,32}; quantization->scales.resize(width * rows / 32);
    for (size_t block = 0; block < quantization->scales.size(); ++block) {
        const size_t block_bytes = ggml_type_size(type);
        const uint8_t * data = source.data() + block * block_bytes;
        ggml_fp16_t half; std::memcpy(&half, data, 2);
        const float delta = ggml_fp16_to_fp32(half);
        float minimum = 0;
        if (type == GGML_TYPE_Q4_1) { std::memcpy(&half, data + 2, 2); minimum = ggml_fp16_to_fp32(half); }
        if (!std::isfinite(delta) || !std::isfinite(minimum)) {
            return output_t::failure(invalid("Nonfinite W4FP16 block metadata"));
        }
        const float scale = delta == 0 ? 1.0f : std::fabs(delta);
        // HTP consumes floating offsets in quantized units, including fractional Q4_1 minima.
        const float offset = type == GGML_TYPE_Q4_1 ? (delta == 0 ? minimum :
            minimum / scale + (delta > 0 ? 8.0f : -7.0f)) : delta < 0 ? 1.0f : 0.0f;
        if (!std::isfinite(offset)) { return output_t::failure(invalid("Overflowing W4FP16 block offset")); }
        quantization->scales[block] = {scale, offset};
        for (size_t j = 0; j < 32; ++j) {
            const uint8_t byte = data[(type == GGML_TYPE_Q4_1 ? 4 : 2) + j % 16];
            const int nibble = j < 16 ? byte & 15 : byte >> 4;
            const int code = delta == 0 ? 0 : delta > 0 ? nibble - 8 : 7 - nibble;
            const float value = type == GGML_TYPE_Q4_1 ? delta * nibble + minimum : delta * (nibble - 8);
            if (std::fabs(value) > 65504.0f) { return output_t::failure(invalid("Overflowing W4FP16 block")); }
            output.codes[block * 32 + j] = int8_t(code);
        }
    }
    output.quantization = std::move(quantization);
    return output_t::success(std::move(output));
}

status w4_weight_upload::update(array_view<const uint8_t> source, size_t width, size_t rows, size_t offset, size_t bytes, ggml_type type) {
    if (!supported_type(type) || !valid_shape(width,rows) || source.size() != width / 32 * ggml_type_size(type) * rows || !source.data() ||
        offset > source.size() || bytes > source.size() - offset) {
        return status::failure(invalid("Invalid W4FP16 upload"));
    }
    if (!bytes) { return status::success(); }
    if (width != width_ || rows != rows_ || type != type_) { loaded_ = false; ranges_.clear(); width_ = width; rows_ = rows; type_ = type; }
    payload_.reset();
    if (!loaded_) {
        ranges_.push_back({offset,offset+bytes}); std::sort(ranges_.begin(),ranges_.end());
        std::vector<std::pair<size_t,size_t>> merged;
        for (const auto & range : ranges_) {
            if (!merged.empty() && range.first <= merged.back().second) { merged.back().second = std::max(merged.back().second,range.second); }
            else { merged.push_back(range); }
        }
        ranges_ = std::move(merged);
        if (ranges_.size() > 128) { return status::failure(invalid("Fragmented W4FP16 upload")); }
        loaded_ = ranges_.size() == 1 && ranges_[0].first == 0 && ranges_[0].second == source.size();
    }
    if (loaded_) {
        auto prepared = prepare_w4_weights(source,width,rows,type);
        if (!prepared.ok()) { return status::failure(prepared.error()); }
        payload_ = std::make_shared<const w4_weights>(std::move(prepared).value()); ++preparations_;
        ranges_.clear();
    }
    return status::success();
}
std::shared_ptr<const w4_weights> w4_weight_upload::get(size_t width, size_t rows, ggml_type type) const noexcept {
    return width == width_ && rows == rows_ && type == type_ ? payload_ : nullptr;
}

}
