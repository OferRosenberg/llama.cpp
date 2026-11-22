#pragma once

#include "qairt-runtime.h"
#include "qairt-tensor.h"

namespace ggml::qairt {

class graph_owner;
class context_owner final {
public:
    context_owner(const context_owner &) = delete;
    context_owner & operator=(const context_owner &) = delete;
    context_owner(context_owner &&) = delete;
    context_owner & operator=(context_owner &&) = delete;
    static result<std::shared_ptr<context_owner>> create(std::shared_ptr<runtime_owner> runtime);
    Qnn_ContextHandle_t handle() const noexcept { return context_.get(); }
    std::shared_ptr<runtime_owner> runtime() const noexcept { return runtime_; }
    status close();
private:
    friend class graph_owner;
    context_owner() = default;
    Qnn_ErrorHandle_t close_raw() noexcept;
    static void dispose(context_owner * owner) noexcept;
    std::shared_ptr<runtime_owner> runtime_;
    unique_qnn_handle<context_tag> context_;
    uint32_t active_graphs_ = 0;
    bool admitted_ = false;
};

enum class graph_state { building, finalized, failed, closed };
struct scalar_parameter { std::string name; Qnn_Scalar_t value; };
struct tensor_parameter { std::string name; std::vector<uint32_t> dimensions; std::vector<uint32_t> values; Qnn_DataType_t dtype = QNN_DATATYPE_UINT_32; };
struct op_spec {
    std::string name;
    std::string package_name;
    std::string type_name;
    std::vector<tensor_id> inputs;
    std::vector<tensor_id> outputs;
    std::vector<scalar_parameter> parameters;
    std::vector<tensor_parameter> tensor_parameters;
    static op_spec normalization(tensor_id a, tensor_id gamma, tensor_id beta, tensor_id y, bool rms, float epsilon, uint32_t axis);
    static op_spec softmax(tensor_id a, tensor_id y, float beta, uint32_t axis_value);
    static op_spec matmul(tensor_id a, tensor_id w, tensor_id y);
    static op_spec transpose(tensor_id input, tensor_id output, std::vector<uint32_t> permutation);
    static op_spec gather_rows(tensor_id data, tensor_id indices, tensor_id output);
    static op_spec scatter_rows(tensor_id cache, tensor_id indices, tensor_id updates, tensor_id output);
    static op_spec sigmoid(tensor_id x, tensor_id y);
    static op_spec group_query_attention(std::vector<tensor_id> inputs, std::vector<tensor_id> outputs, uint32_t heads, uint32_t kv_heads, float scale);
    static op_spec rotary_embedding(tensor_id x, tensor_id cosine, tensor_id sine, tensor_id y, uint32_t dimensions, bool interleaved);
};
class execution_binding;

class graph_owner final {
public:
    static result<std::shared_ptr<graph_owner>> create(std::shared_ptr<context_owner> context, std::string name);
    ~graph_owner() noexcept;
    result<tensor_id> register_tensor(tensor_spec spec);
    result<tensor_id> register_tensor(tensor_spec spec, buffer_view_const payload);
    status add_node(const op_spec & spec);
    status finalize();
    status execute(execution_binding & binding) const;
    status close();
    const tensor_record * find_tensor(tensor_id id) const noexcept;
    graph_state state() const noexcept { return state_; }
    uint64_t diagnostic_id() const noexcept { return diagnostic_id_; }
    std::shared_ptr<context_owner> context() const noexcept { return context_; }
private:
    friend class execution_binding;
    graph_owner() = default;
    std::shared_ptr<context_owner> context_;
    Qnn_GraphHandle_t graph_ = nullptr;
    std::string name_;
    graph_state state_ = graph_state::building;
    std::vector<std::unique_ptr<tensor_record>> tensors_;
    std::vector<tensor_id> input_ids_;
    std::vector<tensor_id> output_ids_;
    mutable uint32_t active_bindings_ = 0;
    bool node_added_ = false;
    uint64_t diagnostic_id_ = 0;
    std::vector<tensor_id> produced_ids_;
};

class op_packet final {
public:
    static result<std::unique_ptr<op_packet>> create(const graph_owner & graph, const op_spec & spec);
    op_packet(const op_packet &) = delete;
    op_packet & operator=(const op_packet &) = delete;
    Qnn_OpConfig_t descriptor() noexcept;
private:
    friend class graph_owner;
    status register_parameters(graph_owner & graph);
    static result<std::unique_ptr<op_packet>> create_gather(const graph_owner & graph, const op_spec & spec);
    static result<std::unique_ptr<op_packet>> create_scatter(const graph_owner & graph, const op_spec & spec);
    static result<std::unique_ptr<op_packet>> create_attention(const graph_owner & graph, const op_spec & spec);
    op_packet() = default;
    op_spec spec_;
    std::vector<Qnn_Param_t> parameters_;
    std::vector<Qnn_Tensor_t> inputs_;
    std::vector<Qnn_Tensor_t> outputs_;
};

class execution_binding final {
public:
    static result<std::unique_ptr<execution_binding>> create(std::shared_ptr<const graph_owner> graph);
    ~execution_binding() noexcept;
    execution_binding(const execution_binding &) = delete;
    execution_binding & operator=(const execution_binding &) = delete;
    status bind_input(tensor_id id, buffer_view_const buffer);
    status bind_output(tensor_id id, buffer_view buffer);
    template<class T>
    status bind_input(tensor_id id, array_view<const T> values) {
        return bind_typed(id, values.data(), values.size(), sizeof(T), qnn_type_traits<T>::dtype, false);
    }
    template<class T>
    status bind_output(tensor_id id, array_view<T> values) {
        return bind_typed(id, values.data(), values.size(), sizeof(T), qnn_type_traits<T>::dtype, true);
    }
    status validate_complete() const;
private:
    friend class graph_owner;
    execution_binding() = default;
    status bind(tensor_id id, buffer_view_const buffer, bool output);
    status bind_typed(tensor_id id, const void * data, size_t count, size_t element_size, Qnn_DataType_t dtype, bool output);
    std::shared_ptr<const graph_owner> graph_;
    std::vector<Qnn_Tensor_t> inputs_;
    std::vector<Qnn_Tensor_t> outputs_;
    std::vector<buffer_view_const> input_buffers_;
    std::vector<buffer_view> output_buffers_;
    bool admitted_ = false;
};

}
