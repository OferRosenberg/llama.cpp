#include "llama.h"
#include "ggml-backend.h"
#include "ggml-qairt.h"
#include "qairt-model-profile.h"
#include "qairt-model-diagnostics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using device_stats_fn = bool (*)(ggml_backend_dev_t, ggml_backend_qairt_stats *);
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)
struct observation {
    backend_ptr reference {nullptr, ggml_backend_free};
    backend_ptr diagnostic_cpu {nullptr, ggml_backend_free};
    std::unique_ptr<qairt_local_diagnostics> diagnostic;
    std::unique_ptr<qairt_local_diagnostics> simulation;
    bool capture_quantization = false;
    std::map<std::string, qairt_quantization_snapshot> quantization;
    size_t local_checks = 0, local_skipped = 0;
    double local_maximum = 0.0;
    std::string local_worst;
    std::map<std::string, std::vector<float>> tensors;
    std::map<std::string, std::string> buffers;
};

static void logger(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || std::strstr(text, "- type") || std::strstr(text, "buffer size") || std::strstr(text, "offloaded") || std::strstr(text, "offloading")) { std::fputs(text, stderr); }
}
static void compare_local_projection(ggml_tensor * tensor, observation & seen) {
    const auto output_address = reinterpret_cast<uintptr_t>(tensor->data);
    for (const auto * source : tensor->src) {
        if (!source) { continue; }
        const auto input_address = reinterpret_cast<uintptr_t>(source->data);
        if (input_address >= output_address ? input_address - output_address < ggml_nbytes(tensor) :
            output_address - input_address < ggml_nbytes(source)) { ++seen.local_skipped; return; }
    }
    using ggml_context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    ggml_context_ptr context(ggml_init({ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false) + 1024, nullptr, true}), ggml_free);
    CHECK(context); if (!context) { return; }
    auto * reference = ggml_dup_tensor(context.get(), tensor);
    reference->op = tensor->op;
    reference->flags = GGML_TENSOR_FLAG_COMPUTE;
    std::memcpy(reference->op_params, tensor->op_params, sizeof(reference->op_params));
    std::copy(std::begin(tensor->src), std::end(tensor->src), std::begin(reference->src));
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), seen.reference.get()), ggml_backend_buffer_free);
    CHECK(buffer); if (!buffer) { return; }
    auto * graph = ggml_new_graph_custom(context.get(), 1, false);
    // Add only the output node: the original producers must not be recomputed.
    ggml_graph_add_node(graph, reference);
    const auto status = ggml_backend_graph_compute(seen.reference.get(), graph);
    CHECK(status == GGML_STATUS_SUCCESS); if (status != GGML_STATUS_SUCCESS) { return; }
    const size_t count = size_t(ggml_nelements(tensor));
    std::vector<float> expected(count), actual(count);
    ggml_backend_tensor_get(reference, expected.data(), 0, count * sizeof(float));
    ggml_backend_tensor_get(tensor, actual.data(), 0, count * sizeof(float));
    double squared = 0.0, energy = 0.0;
    for (size_t i = 0; i < count; ++i) {
        CHECK(std::isfinite(expected[i]) && std::isfinite(actual[i]));
        const double difference = double(actual[i]) - expected[i];
        squared += difference * difference; energy += double(expected[i]) * expected[i];
    }
    const double error = squared / std::max(energy, 1.0e-12);
    ++seen.local_checks;
    if (error > seen.local_maximum) { seen.local_maximum = error; seen.local_worst = ggml_get_name(tensor); }
    std::printf("local projection %s: NMSE=%.9g\n", ggml_get_name(tensor), error);
    CHECK(error <= 1e-5);
}

static bool observe(ggml_tensor * tensor, bool ask, void * user) {
    const char * name = ggml_get_name(tensor);
    auto & seen = *static_cast<observation *>(user);
    const bool diagnostic = seen.diagnostic && qairt_profile_filter::diagnostic_candidate(tensor);
    if (ask && diagnostic) { return true; }
    if (!ask && seen.simulation && tensor->op == GGML_OP_RMS_NORM &&
        qairt_stage_of(tensor) == qairt_stage::norm) { seen.simulation->simulate_fp16_norm(tensor); }
    if (!ask && seen.capture_quantization && tensor->op == GGML_OP_MUL_MAT &&
        ggml_is_quantized(tensor->src[0]->type) && std::strstr(name, "-15")) {
        seen.quantization[name] = qairt_snapshot_quantization(tensor->src[1]);
    }
    const bool local = seen.reference && tensor->op == GGML_OP_MUL_MAT && tensor->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(tensor) && tensor->buffer && std::strcmp(ggml_backend_buffer_name(tensor->buffer), "QAIRT") == 0;
    if (local && !ask) { compare_local_projection(tensor, seen); }
    if (tensor->op == GGML_OP_SET_ROWS && tensor->type == GGML_TYPE_F16 && ggml_is_contiguous(tensor)) {
        if (ask) { return true; }
        const std::string key = std::string("kv_write_") + name;
        const size_t count = std::min<size_t>(size_t(ggml_nelements(tensor)), 4096);
        std::vector<ggml_fp16_t> half(count); ggml_backend_tensor_get(tensor, half.data(), 0, count * sizeof(ggml_fp16_t));
        auto & values = static_cast<observation *>(user)->tensors[key]; values.resize(count);
        for (size_t i = 0; i < count; ++i) { values[i] = ggml_fp16_to_fp32(half[i]); }
        static_cast<observation *>(user)->buffers[key] = tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : "none";
        return true;
    }
    const char * suffix = std::strrchr(name, '-');
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || !suffix || (std::strcmp(suffix, "-0") != 0 && std::strcmp(suffix, "-15") != 0)) { return local || diagnostic; }
    if (ask) { return true; }
    const size_t count = std::min<size_t>(size_t(ggml_nelements(tensor)), 4096);
    auto & values = static_cast<observation *>(user)->tensors[name]; values.resize(count);
    ggml_backend_tensor_get(tensor, values.data(), 0, count * sizeof(float));
    static_cast<observation *>(user)->buffers[name] = tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : "none";
    return true;
}
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int count = llama_tokenize(vocab, text.c_str(), int32_t(text.size()), nullptr, 0, true, true);
    if (count >= 0 || count == INT32_MIN) { return {}; }
    std::vector<llama_token> tokens(size_t(-count));
    count = llama_tokenize(vocab, text.c_str(), int32_t(text.size()), tokens.data(), int32_t(tokens.size()), true, true);
    if (count < 0) { return {}; }
    tokens.resize(size_t(count)); return tokens;
}
static int decode(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t position) {
    auto batch = llama_batch_init(int32_t(tokens.size()), 0, 1);
    batch.n_tokens = int32_t(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[size_t(i)]; batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = i == batch.n_tokens - 1;
    }
    const int result = llama_decode(ctx, batch); llama_batch_free(batch); return result;
}
static int decode_htp(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t position) {
    qairt_profile_filter::force_selected(true);
    const int result = decode(ctx, tokens, position);
    qairt_profile_filter::force_selected(false);
    return result;
}
static bool compare(const float * cpu, const float * htp, size_t count, const char * label) {
    if (!cpu || !htp || !count) { CHECK(false); return false; }
    double squared = 0, reference = 0; float maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(htp[i])) { CHECK(false); return false; }
        const double difference = double(cpu[i]) - htp[i];
        squared += difference * difference; reference += double(cpu[i]) * cpu[i];
        maximum = std::max(maximum, float(std::fabs(difference)));
    }
    const double rmse = std::sqrt(squared / count);
    const double relative = std::sqrt(squared / std::max(reference, 1.0e-12));
    std::printf("%s: max_abs=%.6f rmse=%.6f relative_rmse=%.6f\n", label, maximum, rmse, relative);
    std::fflush(stdout);
    double divergence = 0;
    const bool logits = std::strstr(label, "logits") != nullptr;
    if (logits) {
        const double cpu_max = *std::max_element(cpu, cpu + count), htp_max = *std::max_element(htp, htp + count);
        double cpu_sum = 0, htp_sum = 0;
        for (size_t i = 0; i < count; ++i) { cpu_sum += std::exp(double(cpu[i]) - cpu_max); htp_sum += std::exp(double(htp[i]) - htp_max); }
        const double cpu_log_z = cpu_max + std::log(cpu_sum), htp_log_z = htp_max + std::log(htp_sum);
        for (size_t i = 0; i < count; ++i) { divergence += std::exp(double(cpu[i]) - cpu_log_z) * (double(cpu[i]) - cpu_log_z - htp[i] + htp_log_z); }
        std::printf("%s: KL(cpu||htp)=%.8f raw_relative_target_met=%d\n", label, divergence, int(relative <= 0.02));
        std::fflush(stdout);
    }
    const bool passed = maximum <= 0.75f && (logits ? divergence <= 0.001 : relative <= 0.02);
    CHECK(passed); return passed;
}
static void compare_quantization(const observation & cpu, const observation & hybrid) {
    for (const auto & entry : cpu.quantization) {
        const auto found = hybrid.quantization.find(entry.first);
        if (found != hybrid.quantization.end()) { qairt_compare_quantization(entry.first.c_str(), entry.second, found->second); }
    }
}
static void compare_observations(const observation & cpu, const observation & htp) {
    int compared = 0;
    for (const auto & entry : cpu.tensors) {
        auto found = htp.tensors.find(entry.first);
        if (found == htp.tensors.end() || found->second.size() != entry.second.size()) { continue; }
        compare(entry.second.data(), found->second.data(), entry.second.size(), entry.first.c_str()); ++compared;
    }
    std::printf("layer-zero/offloaded intermediate comparisons=%d\n", compared); CHECK(compared > 0);
}
static context_ptr make_context(llama_model * model, observation & observations, bool prefill, bool htp = false) {
    auto params = llama_context_default_params();
    params.n_ctx = 1024; params.n_batch = 128; params.n_ubatch = prefill ? 128 : 1;
    params.n_seq_max = 1; params.n_threads = 8; params.n_threads_batch = 8;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = GGML_TYPE_F16; params.type_v = GGML_TYPE_F16;
    const char * softmax_gate = std::getenv("GGML_QAIRT_EXPERIMENTAL_SOFTMAX");
    const char * attention_gate = std::getenv("GGML_QAIRT_EXPERIMENTAL_ATTENTION");
    params.offload_kqv = (softmax_gate && std::strcmp(softmax_gate, "1") == 0) || (attention_gate && std::strcmp(attention_gate, "1") == 0);
    params.op_offload = false;
    params.cb_eval = observe; params.cb_eval_user_data = &observations;
    qairt_profile_filter::force_selected(htp);
    context_ptr result(llama_init_from_model(model, params), llama_free);
    qairt_profile_filter::force_selected(false);
    return result;
}
static int test_block_profile() {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    CHECK(ctx); if (!ctx) { return 1; }
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
    ggml_set_name(input, "l_out-14");
    auto * attn_norm = ggml_rms_norm(ctx, input, 1e-5f);
    ggml_set_name(attn_norm, "norm-15");
    auto * attention = ggml_silu(ctx, attn_norm);
    ggml_set_name(attention, "attn_out-15");
    auto * boundary = ggml_add(ctx, attention, input);
    ggml_set_name(boundary, "ffn_inp-15");
    auto * ffn_norm = ggml_rms_norm(ctx, boundary, 1e-5f);
    ggml_set_name(ffn_norm, "norm-15");
    auto * ffn = ggml_silu(ctx, ffn_norm);
    auto * output = ggml_add(ctx, ffn, boundary);
    ggml_set_name(output, "l_out-15");
    auto * final_norm = ggml_rms_norm(ctx, output, 1e-5f);
    ggml_set_name(final_norm, "norm");
    CHECK(qairt_block_half_of(input) == qairt_block_half::outside);
    CHECK(qairt_block_half_of(attn_norm) == qairt_block_half::attention);
    CHECK(qairt_block_half_of(attention) == qairt_block_half::attention);
    CHECK(qairt_block_half_of(boundary) == qairt_block_half::attention);
    CHECK(qairt_block_half_of(ffn_norm) == qairt_block_half::ffn);
    CHECK(qairt_block_half_of(ffn) == qairt_block_half::ffn);
    CHECK(qairt_block_half_of(output) == qairt_block_half::ffn);
    CHECK(qairt_block_half_of(final_norm) == qairt_block_half::outside);
    auto * copied_boundary = ggml_dup(ctx, boundary);
    ggml_set_name(copied_boundary, "ffn_inp-15#CPU");
    CHECK(qairt_block_half_of(ggml_silu(ctx, copied_boundary)) == qairt_block_half::ffn);
    auto * side_branch = ggml_reshape_1d(ctx, attention, 32);
    CHECK(qairt_block_half_of(side_branch) == qairt_block_half::attention);
    auto * matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    ggml_set_name(matrix, "blk.15.attn_q.weight");
    auto * query = ggml_mul_mat(ctx, matrix, attn_norm); ggml_set_name(query, "Qcur-15");
    auto * key = ggml_mul_mat(ctx, matrix, attn_norm); ggml_set_name(key, "Kcur-15");
    auto * value = ggml_mul_mat(ctx, matrix, attn_norm); ggml_set_name(value, "Vcur-15");
    auto * probability = ggml_soft_max(ctx, query); ggml_set_name(probability, "kq_soft_max-15");
    auto * projected = ggml_mul_mat(ctx, matrix, probability); ggml_set_name(projected, "attn_out-15");
    CHECK(qairt_stage_of(attn_norm) == qairt_stage::norm);
    CHECK(qairt_stage_of(query) == qairt_stage::q);
    CHECK(qairt_stage_of(key) == qairt_stage::k);
    CHECK(qairt_stage_of(value) == qairt_stage::v);
    CHECK(qairt_stage_of(probability) == qairt_stage::softmax);
    CHECK(qairt_stage_of(projected) == qairt_stage::output);
    CHECK(qairt_stage_of(boundary) == qairt_stage::residual);
    CHECK(qairt_stage_of(ffn_norm) == qairt_stage::none);

    qairt_block_profile mode;
    CHECK(!qairt_parse_profile("invalid", mode));
    CHECK(!qairt_parse_profile("", mode));
    ggml_backend_device fake {};
    fake.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
    fake.iface.init_backend = [](ggml_backend_dev_t, const char *) -> ggml_backend_t { return nullptr; };
    for (const char * setting : {"cpu", "attention", "ffn", "block", "full"}) {
        CHECK(qairt_parse_profile(setting, mode));
        qairt_profile_filter scope(&fake, mode);
        CHECK(fake.iface.supports_op(&fake, attn_norm) == (mode == qairt_block_profile::attention || mode == qairt_block_profile::block || mode == qairt_block_profile::full));
        CHECK(fake.iface.supports_op(&fake, ffn_norm) == (mode == qairt_block_profile::ffn || mode == qairt_block_profile::block || mode == qairt_block_profile::full));
        CHECK(fake.iface.supports_op(&fake, final_norm) == (mode == qairt_block_profile::full));
    }
    for (const auto & pair : std::vector<std::pair<const char *, uint32_t>> {{"attention-front", 31}, {"attention-back", 992}, {"norm", 1}, {"qkv", 14}, {"rope", 16}, {"state", 96}, {"output-tail", 896}, {"qk", 6}, {"q", 2}, {"k", 4}, {"v", 8}, {"kv", 32}, {"softmax", 64}, {"output", 256}, {"residual", 512}}) {
        CHECK(qairt_parse_profile(pair.first, mode)); CHECK(qairt_profile_mask(mode) == pair.second);
        qairt_profile_filter scope(&fake, mode);
        CHECK(fake.iface.supports_op(&fake, attn_norm) == bool(pair.second & 1));
        CHECK(fake.iface.supports_op(&fake, query) == bool(pair.second & 2));
        CHECK(fake.iface.supports_op(&fake, key) == bool(pair.second & 4));
        CHECK(fake.iface.supports_op(&fake, value) == bool(pair.second & 8));
        CHECK(fake.iface.supports_op(&fake, probability) == bool(pair.second & 64));
        CHECK(fake.iface.supports_op(&fake, projected) == bool(pair.second & 256));
        CHECK(fake.iface.supports_op(&fake, boundary) == bool(pair.second & 512));
        CHECK(!fake.iface.supports_op(&fake, ffn_norm));
    }

    fake.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return false; };
    {
        qairt_profile_filter scope(&fake, qairt_block_profile::full);
        CHECK(!fake.iface.supports_op(&fake, attn_norm));
    }
    CHECK(!fake.iface.supports_op(&fake, ffn_norm));
    ggml_backend simulated_backend {};
    simulated_backend.iface.graph_compute = [](ggml_backend_t, ggml_cgraph *) { return GGML_STATUS_SUCCESS; };
    fake.context = &simulated_backend;
    fake.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
    fake.iface.init_backend = [](ggml_backend_dev_t dev, const char *) { return static_cast<ggml_backend_t>(dev->context); };
    {
        qairt_profile_filter scope(&fake, qairt_block_profile::ffn);
        CHECK(fake.iface.supports_op(&fake, ffn_norm));
        auto * backend = fake.iface.init_backend(&fake, nullptr);
        auto * graph = ggml_new_graph(ctx);
        ggml_graph_add_node(graph, ffn_norm);
        auto * original_input = ffn_norm->src[0];
        ffn_norm->src[0] = input;
        CHECK(backend->iface.graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        CHECK(scope.verified());
        ffn_norm->src[0] = original_input;
    }


    CHECK(qairt_block_half_of(attn_norm, 14) == qairt_block_half::attention);
    CHECK(qairt_block_half_of(input, 14) == qairt_block_half::attention);
    auto * previous_input = ggml_dup_tensor(ctx, input); ggml_set_name(previous_input, "l_out-13");
    auto * previous_norm = ggml_rms_norm(ctx, previous_input, 1e-5f); ggml_set_name(previous_norm, "norm-14");
    CHECK(qairt_stage_of(previous_norm, 14) == qairt_stage::norm);
    {
        qairt_profile_filter scope(&fake, qairt_block_profile::norm, 14);
        CHECK(fake.iface.supports_op(&fake, previous_norm));
        observation seen;
        seen.diagnostic = std::make_unique<qairt_local_diagnostics>(nullptr);
        CHECK(observe(previous_norm, true, &seen));
        // False here aborts the remainder of the scheduler split, rather than
        // merely declining to store a checkpoint.
        CHECK(observe(previous_norm, false, &seen));
    }
    CHECK(qairt_block_half_of(previous_input, 14) == qairt_block_half::outside);
    ggml_free(ctx);
    auto * host_cpu = ggml_backend_cpu_init();
    auto * memory = ggml_init({64*1024, nullptr, true});
    auto * dense = ggml_new_tensor_2d(memory, GGML_TYPE_F32, 2, 3);
    auto * transposed = ggml_transpose(memory, dense);
    auto * destination = ggml_new_tensor_2d(memory, GGML_TYPE_F16, 2, 3);
    auto * source = ggml_new_tensor_2d(memory, GGML_TYPE_F32, 2, 1);
    auto * indices = ggml_new_tensor_1d(memory, GGML_TYPE_I64, 1);
    auto * write = ggml_set_rows(memory, destination, source, indices);
    ggml_set_name(write, "host_cache");
    auto * host_buffer = ggml_backend_alloc_ctx_tensors(memory, host_cpu);
    const float data[] = {1, 2, 3, 4, 5, 6}, expected_transpose[] = {1, 3, 5, 2, 4, 6};
    ggml_backend_tensor_set(dense, data, 0, sizeof(data));
    const auto transpose_bytes = qairt_diagnostic_dense(transposed);
    CHECK(transpose_bytes.size() == sizeof(expected_transpose));
    CHECK(std::memcmp(transpose_bytes.data(), expected_transpose, sizeof(expected_transpose)) == 0);
    ggml_backend_buffer_clear(host_buffer, 0);
    ggml_backend_tensor_set(source, data, 0, 2*sizeof(float));
    const int64_t index = 1; ggml_backend_tensor_set(indices, &index, 0, sizeof(index));
    {
        qairt_local_diagnostics replay(host_cpu);
        replay.prepare(write);
        auto * host_graph = ggml_new_graph_custom(memory, 1, false);
        write->flags = GGML_TENSOR_FLAG_COMPUTE; ggml_graph_add_node(host_graph, write);
        CHECK(ggml_backend_graph_compute(host_cpu, host_graph) == GGML_STATUS_SUCCESS);
        // Changing producers after execution must not alter the saved reference.
        const float changed[] = {99, 99}; ggml_backend_tensor_set(source, changed, 0, sizeof(changed));
        replay.finish(write); CHECK(replay.report());
    }
    ggml_backend_buffer_free(host_buffer); ggml_free(memory); ggml_backend_free(host_cpu);
    std::printf("QAIRT block-profile host: %d failures\n", failures);
    return failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--profile-host") == 0) { return test_block_profile(); }
    if (argc < 2 || argc > 4) { std::fprintf(stderr, "usage: test-qairt-model MODEL [--prefill|--capacity|--prompt TEXT]\n"); return 1; }
    const char * profile_setting = std::getenv("QAIRT_TEST_BLOCK_PROFILE");
    qairt_block_profile profile_mode;
    if (!qairt_parse_profile(profile_setting, profile_mode)) {
        std::fprintf(stderr, "QAIRT_TEST_BLOCK_PROFILE requires a valid block/stage profile (see qairt-model-profile.h)\n"); return 1;
    }
    const bool custom_prompt = argc == 4 && std::strcmp(argv[2], "--prompt") == 0;
    if (argc == 4 && !custom_prompt) { return 1; }
    const bool prefill = argc == 3;
    const bool capacity = argc == 3 && std::strcmp(argv[2], "--capacity") == 0;
    if (argc == 3 && !capacity && std::strcmp(argv[2], "--prefill") != 0) { return 1; }
    llama_log_set(logger, nullptr); ggml_backend_load_all(); llama_backend_init();
    auto device = ggml_backend_dev_by_name("QAIRT0"); CHECK(device);
    if (!device) { return 1; }
    auto reg = ggml_backend_dev_backend_reg(device);
    auto stats = reinterpret_cast<device_stats_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_device_stats"));
    CHECK(stats); if (!stats) { return 1; }
    ggml_backend_dev_t cpu_devices[] = {nullptr}, qairt_devices[] = {device, nullptr};
    auto cpu_params = llama_model_default_params(); cpu_params.devices = cpu_devices; cpu_params.n_gpu_layers = 0;
    auto htp_params = llama_model_default_params(); htp_params.devices = qairt_devices; htp_params.n_gpu_layers = 2;
    const char * layer_setting = std::getenv("QAIRT_TEST_N_GPU_LAYERS");
    if (layer_setting) {
        char * end = nullptr; const long value = std::strtol(layer_setting, &end, 10);
        if (!*layer_setting || *end || value < 2 || value > 17) { std::fprintf(stderr, "QAIRT_TEST_N_GPU_LAYERS requires 2..17 for this fixture\n"); return 1; }
        htp_params.n_gpu_layers = int(value);
    }
    if (profile_setting && htp_params.n_gpu_layers > 3) {
        std::fprintf(stderr, "Block profiles require QAIRT_TEST_N_GPU_LAYERS=2 or 3\n"); return 1;
    }
    std::printf("requested GPU layers=%d\n", htp_params.n_gpu_layers);
    htp_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_ptr cpu_model(llama_model_load_from_file(argv[1], cpu_params), llama_model_free);
    model_ptr htp_model(llama_model_load_from_file(argv[1], htp_params), llama_model_free);
    CHECK(cpu_model && htp_model); if (!cpu_model || !htp_model) { return 1; }
    const auto * vocab = llama_model_get_vocab(cpu_model.get());
    const size_t vocabulary = size_t(llama_vocab_n_tokens(vocab));
    std::printf("model: layers=%d embedding=%d vocabulary=%zu\n", llama_model_n_layer(cpu_model.get()), llama_model_n_embd(cpu_model.get()), vocabulary);
    observation cpu_observations, htp_observations;
    const char * local_setting = std::getenv("QAIRT_TEST_LOCAL_REFERENCE");
    if (local_setting && std::strcmp(local_setting, "0") != 0 && std::strcmp(local_setting, "1") != 0) {
        std::fprintf(stderr, "QAIRT_TEST_LOCAL_REFERENCE must be 0 or 1\n"); return 1;
    }
    if (local_setting && std::strcmp(local_setting, "1") == 0) {
        htp_observations.reference.reset(ggml_backend_init_by_name("CPU", nullptr));
        CHECK(htp_observations.reference); if (!htp_observations.reference) { return 1; }
    }
    std::unique_ptr<qairt_profile_filter> profile_filter;
    if (profile_setting) {
        CHECK(llama_model_n_layer(htp_model.get()) == 16);
        if (llama_model_n_layer(htp_model.get()) != 16) { return 1; }
        profile_filter = std::make_unique<qairt_profile_filter>(device, profile_mode, 17 - htp_params.n_gpu_layers);
        std::printf("block profile=%s split=ffn_inp-15\n", profile_setting);
    }
    const char * diagnostics = std::getenv("QAIRT_TEST_LOCAL_OPS");
    if (diagnostics && std::strcmp(diagnostics, "1") == 0) {
        cpu_observations.capture_quantization = htp_observations.capture_quantization = true;
        CHECK(profile_filter); if (!profile_filter) { return 1; }
        htp_observations.diagnostic_cpu.reset(ggml_backend_cpu_init());
        CHECK(htp_observations.diagnostic_cpu); if (!htp_observations.diagnostic_cpu) { return 1; }
        htp_observations.diagnostic = std::make_unique<qairt_local_diagnostics>(htp_observations.diagnostic_cpu.get());
        profile_filter->set_observer(qairt_local_diagnostics::before, qairt_local_diagnostics::after, htp_observations.diagnostic.get());
    }
    const char * simulation = std::getenv("QAIRT_TEST_SIMULATE_FP16_NORM");
    if (simulation && std::strcmp(simulation, "1") == 0) {
        CHECK(profile_filter && profile_filter->is_cpu() && !htp_observations.diagnostic);
        if (!profile_filter || !profile_filter->is_cpu() || htp_observations.diagnostic) { return 1; }
        cpu_observations.capture_quantization = htp_observations.capture_quantization = true;
        htp_observations.diagnostic_cpu.reset(ggml_backend_cpu_init());
        htp_observations.simulation = std::make_unique<qairt_local_diagnostics>(htp_observations.diagnostic_cpu.get());
    }
    auto cpu = make_context(cpu_model.get(), cpu_observations, prefill);
    auto htp = make_context(htp_model.get(), htp_observations, prefill, true);
    CHECK(cpu && htp); if (!cpu || !htp) { return 1; }
    auto seed = tokenize(vocab, custom_prompt ? argv[3] : "The capital of France is");
    CHECK(!seed.empty()); if (seed.empty()) { return 1; }
    std::vector<llama_token> prompt = seed;
    if (prefill) {
        prompt.resize(capacity ? 1024 : 141);
        for (size_t i = seed.size(); i < prompt.size(); ++i) { prompt[i] = seed[std::min<size_t>(i % seed.size(), seed.size() - 1)]; }
    }
    ggml_backend_qairt_stats initial {}, warm {}, next {};
    CHECK(stats(device, &initial));
    int32_t position = 0;
    for (size_t start = 0; start < prompt.size(); start += prefill ? 128 : 1) {
        const size_t length = std::min<size_t>(prefill ? 128 : 1, prompt.size() - start);
        std::vector<llama_token> chunk(prompt.begin() + start, prompt.begin() + start + length);
        const auto start_time = llama_time_us();
        const int cpu_status = decode(cpu.get(), chunk, position), htp_status = decode_htp(htp.get(), chunk, position);
        CHECK(cpu_status == 0 && htp_status == 0); if (cpu_status || htp_status) { return 1; } position += int32_t(length);
        CHECK(stats(device, &next)); std::printf("position=%d seconds=%.3f preparations=%llu executions=%llu\n", position, (llama_time_us() - start_time) / 1.0e6, (unsigned long long) next.preparations, (unsigned long long) next.executions); std::fflush(stdout);
    }
    compare(llama_get_logits_ith(cpu.get(), -1), llama_get_logits_ith(htp.get(), -1), vocabulary, "prompt logits");
    compare_observations(cpu_observations, htp_observations);
    compare_quantization(cpu_observations, htp_observations);
    for (const char * name : {"Qcur-15", "Kcur-15", "ffn_inp-15", "l_out-15", "kq_soft_max-15"}) {
        const auto found = htp_observations.buffers.find(name);
        if (found != htp_observations.buffers.end()) { std::printf("executed tensor %s: buffer=%s\n", name, found->second.c_str()); }
    }
    const char * softmax_gate = std::getenv("GGML_QAIRT_EXPERIMENTAL_SOFTMAX");
    if (softmax_gate && std::strcmp(softmax_gate, "1") == 0 && !profile_filter) {
        const auto found = htp_observations.buffers.find("kq_soft_max-15");
        CHECK(found != htp_observations.buffers.end() && found->second == "QAIRT");
    }
    const char * attention_gate = std::getenv("GGML_QAIRT_EXPERIMENTAL_ATTENTION");
    if (attention_gate && std::strcmp(attention_gate, "1") == 0 && !profile_filter) {
        size_t writes = 0;
        for (const auto & tensor : htp_observations.buffers) {
            if (tensor.first.rfind("kv_write_", 0) == 0 && tensor.second == "QAIRT") { ++writes; std::printf("executed KV write %s: buffer=%s\n", tensor.first.c_str(), tensor.second.c_str()); }
        }
        CHECK(writes >= 2);
    }
    CHECK(stats(device, &warm));
    CHECK(profile_filter && profile_filter->is_cpu() ? warm.executions == initial.executions : warm.executions > initial.executions);
    int disagreements = 0; std::string generated;
    if (capacity) {
        CHECK(decode(cpu.get(), {seed.back()}, position) != 0);
        CHECK(decode_htp(htp.get(), {seed.back()}, position) != 0);
    } else {
        std::printf("teacher-forced greedy tokens:");
        for (int step = 0; step < 8; ++step) {
            const auto * reference = llama_get_logits_ith(cpu.get(), -1);
            const auto * actual = llama_get_logits_ith(htp.get(), -1);
            const auto token_cpu = llama_token(std::max_element(reference, reference + vocabulary) - reference);
            const auto token_htp = llama_token(std::max_element(actual, actual + vocabulary) - actual);
            disagreements += token_cpu != token_htp;
            char piece[256]; const int length = llama_token_to_piece(vocab, token_htp, piece, sizeof(piece), 0, false);
            if (length > 0 && length <= int(sizeof(piece))) { generated.append(piece, size_t(length)); }
            std::printf(" %d/%d", token_cpu, token_htp);
            CHECK(decode(cpu.get(), {token_cpu}, position) == 0); CHECK(decode_htp(htp.get(), {token_cpu}, position) == 0); ++position;
            compare(llama_get_logits_ith(cpu.get(), -1), llama_get_logits_ith(htp.get(), -1), vocabulary, "decode logits");
            compare_quantization(cpu_observations, htp_observations);
            CHECK(stats(device, &next));
            if (step > 0) { CHECK(next.preparations == warm.preparations); }
            warm = next;
        }
        std::printf("\ngreedy disagreements=%d/8\nteacher-forced output: %s\n", disagreements, generated.c_str());
        CHECK(disagreements <= 1);
    }
    llama_memory_clear(llama_get_memory(cpu.get()), true); llama_memory_clear(llama_get_memory(htp.get()), true);
    CHECK(decode(cpu.get(), {seed.front()}, 0) == 0); CHECK(decode_htp(htp.get(), {seed.front()}, 0) == 0);
    compare(llama_get_logits_ith(cpu.get(), -1), llama_get_logits_ith(htp.get(), -1), vocabulary, "reset logits");
    CHECK(stats(device, &next));
    std::printf("HTP totals: preparations=%llu hits=%llu executions=%llu\n", (unsigned long long) next.preparations, (unsigned long long) next.cache_hits, (unsigned long long) next.executions);
    if (htp_observations.reference) {
        CHECK(htp_observations.local_checks > 0);
        std::printf("local projections: count=%zu skipped=%zu max_NMSE=%.9g worst=%s\n",
            htp_observations.local_checks, htp_observations.local_skipped,
            htp_observations.local_maximum, htp_observations.local_worst.c_str());
    }
    if (htp_observations.simulation) { CHECK(htp_observations.simulation->report()); }
    if (htp_observations.diagnostic) { CHECK(htp_observations.diagnostic->report()); }
    if (profile_filter) { profile_filter->report(profile_setting); CHECK(profile_filter->verified()); }
    std::printf("QAIRT M2 model: %d failures\n", failures);
    htp.reset(); cpu.reset(); htp_model.reset(); cpu_model.reset(); profile_filter.reset(); llama_backend_free();
    return failures ? 1 : 0;
}
