#include "qairt-tensor.h"

#include <limits>

namespace ggml::qairt {

result<size_t> tensor_record::checked_bytes(const tensor_spec & spec) {
    if (spec.name.empty() || spec.name.size() > 128 || spec.name.find('\0') != std::string::npos ||
        spec.dtype != QNN_DATATYPE_FLOAT_16 || spec.dimensions.size() != 2 ||
        (spec.role != QNN_TENSOR_TYPE_STATIC && spec.role != QNN_TENSOR_TYPE_APP_WRITE &&
         spec.role != QNN_TENSOR_TYPE_APP_READ && spec.role != QNN_TENSOR_TYPE_NATIVE)) {
        return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor", "Unsupported M0 tensor name, dtype, rank or role"));
    }
    size_t bytes = sizeof(fp16_bits);
    for (uint32_t dimension : spec.dimensions) {
        if (!dimension || bytes > std::numeric_limits<size_t>::max() / dimension) {
            return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor.size", "Zero dimension or size overflow"));
        }
        bytes *= dimension;
    }
    if (bytes > UINT32_MAX) {
        return result<size_t>::failure(host_error(error_stage::tensor_registration, "tensor.size", "RAW buffer exceeds uint32 limit"));
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
    return tensor;
}

status validate_buffer(const tensor_record & tensor, buffer_view_const buffer) {
    auto bytes = tensor.required_bytes();
    if (!bytes.ok()) { return status::failure(bytes.error()); }
    const auto address = reinterpret_cast<uintptr_t>(buffer.data);
    if (!buffer.data || buffer.bytes != bytes.value() || address % alignof(fp16_bits) ||
        buffer.bytes > std::numeric_limits<uintptr_t>::max() - address) {
        return status::failure(host_error(error_stage::io_binding, "buffer", "Invalid pointer, alignment or byte length"));
    }
    return status::success();
}

}
