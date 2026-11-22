#pragma once

#include "qairt-graph.h"
#include "qairt-lowering.h"
#include "qairt-weights.h"
#include "ggml-qairt.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml::qairt {

struct buffer_weight {
    size_t bytes = 0;
    w4_weight_upload upload;
};
struct buffer_context {
    void * storage = nullptr;
    size_t size = 0;
    uint64_t identity = 0;
    uint64_t generation = 0;
    bool w4fp16 = false;
    bool diagnostics = false;
    mutable std::mutex weights_mutex;
    std::unordered_map<size_t,buffer_weight> weights;
    void touch() noexcept;
};

bool qairt_buffer(ggml_backend_buffer_t buffer) noexcept;
bool readable_tensor(const ggml_tensor * tensor) noexcept;

struct pending_output {
    ggml_tensor * tensor;
    std::vector<uint8_t> bytes;
};
struct boundary_input {
    size_t node;
    size_t source;
    tensor_id id;
};
struct rope_tables {
    std::vector<fp16_bits> cosine, sine;
    size_t rows = 0;
    tensor_id cosine_input, sine_input;
};
enum class partition_input_kind { numerical, indices, scatter_indices, mask, cosine, sine };
struct partition_input {
    size_t slot;
    tensor_id id;
    partition_input_kind kind;
    size_t node = 0;
};
struct partition_rope {
    size_t node;
    rope_tables table;
    std::vector<float> factors;
};
struct partition_span { size_t node, first_op, count; };
struct partition_data {
    std::vector<partition_input> inputs;
    std::vector<size_t> output_nodes;
    std::vector<partition_rope> ropes;
    std::vector<partition_span> spans;
    size_t qnn_nodes = 0;
    size_t w4_matmuls = 0;
};
struct prepared_run {
    std::vector<uint64_t> key;
    std::shared_ptr<graph_owner> graph;
    std::vector<boundary_input> inputs;
    std::vector<tensor_id> outputs;
    size_t static_bytes = 0;
    std::unique_ptr<partition_data> partition;
};

class backend_context final {
public:
    explicit backend_context(std::shared_ptr<runtime_owner> runtime) : runtime_(std::move(runtime)) {}
    backend_context(const backend_context &) = delete;
    backend_context & operator=(const backend_context &) = delete;
    ggml_status compute(ggml_cgraph * graph);
    void invalidate() noexcept { cache_.clear(); shared_context_.reset(); }
    ggml_backend_qairt_stats stats() const noexcept { return stats_; }
private:
    result<std::vector<fp16_bits>> pack(const ggml_tensor * tensor, const std::vector<pending_output> & pending) const;
    result<std::vector<uint8_t>> mask_bytes(const ggml_tensor * tensor, const std::vector<pending_output> & pending) const;
    result<std::vector<float>> float_parameters(const ggml_tensor * tensor, const std::vector<pending_output> & pending) const;
    status run_partition(const std::vector<ggml_tensor *> & nodes, std::vector<pending_output> & pending);
    bool reserve_cache(size_t additional, const prepared_run * keep = nullptr);
    std::shared_ptr<runtime_owner> runtime_;
    std::shared_ptr<context_owner> shared_context_;
    uint64_t next_graph_name_ = 0;
    std::vector<std::unique_ptr<prepared_run>> cache_;
    ggml_backend_qairt_stats stats_ {};
};

}
