#pragma once

#include "qairt-status.h"

#include <vector>

namespace ggml::qairt {

template<class T>
class array_view {
public:
    array_view(T * data, size_t count) noexcept : data_(data), count_(count) {}
    T * data() const noexcept { return data_; }
    size_t size() const noexcept { return count_; }
    T & operator[](size_t index) const { GGML_ASSERT(data_ && index < count_); return data_[index]; }
private:
    T * data_;
    size_t count_;
};

struct buffer_view_const { const void * data = nullptr; size_t bytes = 0; };
struct buffer_view { void * data = nullptr; size_t bytes = 0; };
struct fp16_bits { uint16_t bits = 0; };
static_assert(sizeof(fp16_bits) == 2);
template<class T> struct qnn_type_traits;
template<> struct qnn_type_traits<fp16_bits> {
    static constexpr Qnn_DataType_t dtype = QNN_DATATYPE_FLOAT_16;
};

struct tensor_id { uint32_t value = 0; };
inline constexpr tensor_id null_tensor_id {UINT32_MAX};
struct tensor_spec {
    std::string name;
    Qnn_TensorType_t role;
    Qnn_DataType_t dtype;
    std::vector<uint32_t> dimensions;
};

class graph_owner;
class tensor_record final {
public:
    tensor_record(const tensor_record &) = delete;
    tensor_record & operator=(const tensor_record &) = delete;
    tensor_record(tensor_record &&) = delete;
    tensor_record & operator=(tensor_record &&) = delete;
    static result<size_t> checked_bytes(const tensor_spec & spec);
    const tensor_spec & spec() const noexcept { return spec_; }
    tensor_id id() const noexcept { return id_; }
    result<size_t> required_bytes() const { return checked_bytes(spec_); }
    Qnn_Tensor_t descriptor(buffer_view_const buffer = {}) const;
    bool dimensions_unchanged() const noexcept { return abi_dimensions_ == spec_.dimensions; }
private:
    friend class graph_owner;
    explicit tensor_record(tensor_spec spec) : spec_(std::move(spec)), abi_dimensions_(spec_.dimensions) {}
    tensor_spec spec_;
    mutable std::vector<uint32_t> abi_dimensions_;
    tensor_id id_;
};

status validate_buffer(const tensor_record & tensor, buffer_view_const buffer);

}
