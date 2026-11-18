#pragma once

#include "qairt-graph.h"
#include "qairt-lowering.h"
#include "ggml-qairt.h"
#include <memory>
#include <vector>

namespace ggml::qairt {

struct buffer_context {
    void * storage = nullptr;
    size_t size = 0;
    uint64_t identity = 0;
    uint64_t generation = 0;
    void touch() noexcept;
};

bool qairt_buffer(ggml_backend_buffer_t buffer) noexcept;
bool readable_tensor(const ggml_tensor * tensor) noexcept;

struct pending_output {
    ggml_tensor * tensor;
    std::vector<uint8_t> bytes;
};
enum class input_preparation { none, q8_k, bf16 };
struct boundary_input {
    size_t node;
    size_t source;
    tensor_id id;
    input_preparation preparation = input_preparation::none;
};
struct rope_tables {
    std::vector<fp16_bits> cosine, sine;
    size_t rows = 0;
    tensor_id cosine_input, sine_input;
};
struct prepared_run {
    std::vector<uint64_t> key;
    std::shared_ptr<graph_owner> graph;
    std::vector<boundary_input> inputs;
    std::vector<tensor_id> outputs;
    size_t static_bytes = 0;
    std::unique_ptr<rope_tables> rope;
};

class backend_context final {
public:
    explicit backend_context(std::shared_ptr<runtime_owner> runtime) : runtime_(std::move(runtime)) {}
    backend_context(const backend_context &) = delete;
    backend_context & operator=(const backend_context &) = delete;
    ggml_status compute(ggml_cgraph * graph);
    void invalidate() noexcept { cache_.clear(); }
    ggml_backend_qairt_stats stats() const noexcept { return stats_; }
private:
    result<std::vector<fp16_bits>> pack(const ggml_tensor * tensor, const std::vector<pending_output> & pending, input_preparation preparation = input_preparation::none) const;
    result<prepared_run *> prepare_run(const std::vector<ggml_tensor *> & nodes, const std::vector<pending_output> & pending);
    result<prepared_run *> prepare_rope(const ggml_tensor * tensor, const std::vector<float> & factors, size_t rows);
    status run_rope(ggml_tensor * tensor, std::vector<pending_output> & pending);
    status run_attention(ggml_tensor * tensor, std::vector<pending_output> & pending);
    status run_attention_decomposed(ggml_tensor * tensor, std::vector<pending_output> & pending);
    bool native_gqa_unavailable_ = false;
    status run_set_rows(ggml_tensor * tensor, std::vector<pending_output> & pending);
    bool reserve_cache(size_t additional, const prepared_run * keep = nullptr);
    std::shared_ptr<runtime_owner> runtime_;
    std::vector<std::unique_ptr<prepared_run>> cache_;
    ggml_backend_qairt_stats stats_ {};
};

}
