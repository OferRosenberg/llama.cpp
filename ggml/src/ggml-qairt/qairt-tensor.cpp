#include "qairt-tensor.h"

#include <cmath>
#include <limits>

namespace ggml::qairt {

result<size_t> tensor_record::checked_bytes(const tensor_spec & spec) {
    const bool integer_tensor = spec.dtype == QNN_DATATYPE_INT_32 && !spec.dimensions.empty() && spec.dimensions.size() <= 2;
    const bool axis_tensor = spec.dtype == QNN_DATATYPE_UINT_32 && spec.role == QNN_TENSOR_TYPE_STATIC && !spec.dimensions.empty() && spec.dimensions.size() <= 2;
    const bool block_weight = spec.dtype == QNN_DATATYPE_SFIXED_POINT_8 && spec.role == QNN_TENSOR_TYPE_STATIC &&
        spec.quantization && spec.dimensions.size() == 4 && spec.dimensions[0] == 1 && spec.dimensions[1] == 1 &&
        spec.quantization->block_size == std::vector<uint32_t>({1,1,1,32}) && spec.dimensions[3] % 32 == 0;
    if (spec.quantization && !block_weight) {
        return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor", "Invalid block-weight encoding"));
    }
    if (spec.name.empty() || spec.name.size() > 128 || spec.name.find('\0') != std::string::npos ||
        (!block_weight && !axis_tensor && !integer_tensor && (spec.dtype != QNN_DATATYPE_FLOAT_16 || spec.dimensions.empty() || spec.dimensions.size() > 4)) ||
        (spec.role != QNN_TENSOR_TYPE_STATIC && spec.role != QNN_TENSOR_TYPE_APP_WRITE &&
         spec.role != QNN_TENSOR_TYPE_APP_READ && spec.role != QNN_TENSOR_TYPE_NATIVE)) {
        return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor", "Unsupported tensor name, dtype, rank or role"));
    }
    size_t bytes = block_weight ? 1 : spec.dtype == QNN_DATATYPE_FLOAT_16 ? sizeof(fp16_bits) : sizeof(uint32_t);
    for (uint32_t dimension : spec.dimensions) {
        if (!dimension || bytes > std::numeric_limits<size_t>::max() / dimension) {
            return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor.size", "Zero dimension or size overflow"));
        }
        bytes *= dimension;
    }
    if (bytes > UINT32_MAX) {
        return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor.size", "RAW buffer exceeds uint32 limit"));
    }
    if (block_weight && spec.quantization->scales.size() != bytes / 32) {
        return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor", "Invalid block-weight scale count"));
    }
    if (block_weight) {
        for (const auto & scale : spec.quantization->scales) {
            if (!std::isfinite(scale.scale) || scale.scale <= 0 || (scale.offset != 0 && scale.offset != 1)) {
                return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor", "Invalid W4FP16 scale or offset"));
            }
        }
    }
    return result<size_t>::success(bytes);
}

Qnn_Tensor_t tensor_record::descriptor(buffer_view_const buffer) const {
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    tensor.v1.id = id_.value;
    tensor.v1.name = spec_.name.c_str();
    tensor.v1.type = spec_.role;
    tensor.v1.dataType = spec_.dtype;
    tensor.v1.rank = static_cast<uint32_t>(abi_dimensions_.size());
    tensor.v1.dimensions = abi_dimensions_.data();
    tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    // QNN's input-buffer ABI uses void *, but input bytes remain read-only to this backend.
    tensor.v1.clientBuf.data = const_cast<void *>(buffer.data);
    tensor.v1.clientBuf.dataSize = static_cast<uint32_t>(buffer.bytes);
    if (spec_.quantization) {
        tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK;
        tensor.v1.quantizeParams.bwFloatBlockEncoding = {4, abi_blocks_.data(), abi_scales_.data()};
    }
    return tensor;
}

status validate_buffer(const tensor_record & tensor, buffer_view_const buffer) {
    auto bytes = tensor.required_bytes();
    if (!bytes.ok()) { return status::failure(bytes.error()); }
    const auto address = reinterpret_cast<uintptr_t>(buffer.data);
    const size_t alignment = tensor.spec().dtype == QNN_DATATYPE_SFIXED_POINT_8 ? 1 : tensor.spec().dtype == QNN_DATATYPE_FLOAT_16 ? alignof(fp16_bits) : alignof(uint32_t);
    if (!buffer.data || buffer.bytes != bytes.value() || address % alignment ||
        buffer.bytes > std::numeric_limits<uintptr_t>::max() - address) {
        return status::failure(host_error(error_stage::io_binding, "buffer", "Invalid pointer, alignment or byte length"));
    }
    return status::success();
}

}
