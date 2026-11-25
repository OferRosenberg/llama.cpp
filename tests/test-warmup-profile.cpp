#include "arg.h"
#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-qairt.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

static bool parse_size(const char * size) {
    std::vector<std::string> args {"test-warmup-profile", "-m", "unused.gguf", "--warmup-batch-size", size};
    std::vector<char *> argv;
    for (auto & arg : args) { argv.push_back(arg.data()); }
    common_params params;
    return common_params_parse(int(argv.size()), argv.data(), params, LLAMA_EXAMPLE_CLI);
}

static ggml_backend_t captured_backend = nullptr;
static ggml_backend_t (*original_init)(ggml_backend_dev_t, const char *) = nullptr;
static ggml_backend_t capture_init(ggml_backend_dev_t device, const char * params) {
    captured_backend = original_init(device, params);
    return captured_backend;
}

static void model_test(const char * path, bool cpu) {
    std::vector<std::string> args {"test-warmup-profile", "-m", path, "--device", cpu ? "none" : "QAIRT0",
        "-ngl", cpu ? "0" : "29", "--fit", "off", "-c", "1024", "-b", "128", "-ub", "128",
        "-ctk", "f16", "-ctv", "f16", "-fa", "off", "--warmup-batch-size", "27"};
    std::vector<char *> argv;
    for (auto & arg : args) { argv.push_back(arg.data()); }
    common_params params;
    CHECK(common_params_parse(int(argv.size()), argv.data(), params, LLAMA_EXAMPLE_CLI));
    ggml_backend_load_all();
    auto device = cpu ? nullptr : ggml_backend_dev_by_name("QAIRT0");
    if (!cpu && !device) { CHECK(device != nullptr); return; }
    if (device) {
        original_init = device->iface.init_backend;
        device->iface.init_backend = capture_init;
    }
    common_init_result_ptr result;
    try { result = common_init_from_params(params); }
    catch (...) {
        if (device) { device->iface.init_backend = original_init; }
        throw;
    }
    if (device) { device->iface.init_backend = original_init; }
    auto * ctx = result ? result->context() : nullptr;
    CHECK(ctx != nullptr);
    if (!ctx) { return; }
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1);
    CHECK(!common_warmup_profile(ctx, 0));
    CHECK(!common_warmup_profile(ctx, 129));
    ggml_backend_qairt_get_stats_t get_stats = nullptr;
    ggml_backend_qairt_stats initialized {}, computed {};
    if (device) {
        get_stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_qairt_get_stats"));
        CHECK(get_stats && captured_backend);
        if (!get_stats || !captured_backend) { return; }
        CHECK(get_stats(captured_backend, &initialized));
        CHECK(initialized.preparations > 0 && initialized.finalization_failures == 0);
    }
    const auto * vocab = llama_model_get_vocab(result->model());
    const std::string prompt = "<|im_start|>user\nWhat is 17 plus 25? Answer only the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    llama_tokens tokens = common_tokenize(vocab, prompt, true, true);
    CHECK(tokens.size() == 27);
    common_batch batch = common_batch_get_one(ctx, tokens);
    CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    auto * logits = llama_get_logits_ith(ctx, -1);
    CHECK(logits != nullptr);
    if (logits) { for (int32_t i = 0; i < llama_vocab_n_tokens(vocab); ++i) { CHECK(std::isfinite(logits[i])); } }
    const llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    CHECK(!common_warmup_profile(ctx, 27));
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == pos);
    const llama_token token = tokens.back();
    common_batch decode = common_batch_get_one(ctx, &token, 1);
    CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, decode.get()) == 0);
    llama_synchronize(ctx);
    if (cpu) {
        auto memory = llama_get_memory(ctx);
        llama_memory_seq_cp(memory, 0, 42, -1, -1);
        CHECK(llama_memory_seq_rm(memory, 0, -1, -1));
        const llama_pos other_pos = llama_memory_seq_pos_max(memory, 42);
        CHECK(other_pos >= 0);
        CHECK(!common_warmup_profile(ctx, 27));
        CHECK(llama_memory_seq_pos_max(memory, 42) == other_pos);
    }
    if (get_stats) {
        CHECK(get_stats(captured_backend, &computed));
        CHECK(computed.preparations == initialized.preparations);
        CHECK(computed.executions > initialized.executions);
        CHECK(computed.cache_hits > initialized.cache_hits);
        std::printf("INIT_GRAPHS=%llu QUERY_NEW_GRAPHS=%llu INIT_PREPARATION_MS=%.2f INIT_FINALIZATION_MS=%.2f\n",
            (unsigned long long) initialized.preparations,
            (unsigned long long) (computed.preparations - initialized.preparations),
            initialized.preparation_time_us / 1000.0, initialized.finalization_time_us / 1000.0);
    }
}
static void boundary_benchmark(const char * model_path, const char * reference_path, bool save) {
    const std::filesystem::path directory(reference_path);
    std::vector<std::string> args {"test-warmup-profile", "-m", model_path, "--device", "QAIRT0",
        "-ngl", "29", "--fit", "off", "-c", "1024", "-b", "128", "-ub", "128",
        "-t", "8", "-tb", "8", "-ctk", "f16", "-ctv", "f16", "-fa", "off", "--no-warmup"};
    std::vector<char *> argv;
    for (auto & arg : args) { argv.push_back(arg.data()); }
    common_params params;
    CHECK(common_params_parse(int(argv.size()), argv.data(), params, LLAMA_EXAMPLE_CLI));
    auto device = ggml_backend_dev_by_name("QAIRT0");
    CHECK(device != nullptr);
    if (!device) { return; }
    original_init = device->iface.init_backend;
    device->iface.init_backend = capture_init;
    common_init_result_ptr result;
    try { result = common_init_from_params(params); }
    catch (...) { device->iface.init_backend = original_init; throw; }
    device->iface.init_backend = original_init;
    auto * ctx = result ? result->context() : nullptr;
    CHECK(ctx != nullptr);
    if (!ctx) { return; }
    auto get_stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_qairt_get_stats"));
    CHECK(get_stats && captured_backend);
    if (!get_stats || !captured_backend) { return; }
    const auto * vocab = llama_model_get_vocab(result->model());
    const std::string prompt = "<|im_start|>user\nExplain how rain forms. Write three complete sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    auto tokens = common_tokenize(vocab, prompt, true, true);
    CHECK(common_warmup_profile(ctx, int32_t(tokens.size())));
    ggml_backend_qairt_stats initialized {}, computed {};
    CHECK(get_stats(captured_backend, &initialized));
    llama_tokens replay;
    if (!save) {
        const auto path = directory / "tokens.txt";
        if (std::filesystem::file_size(path) > 1024 * 1024) { throw std::runtime_error("reference exceeds read limit"); }
        std::ifstream file(path);
        llama_token token;
        while (file >> token) { replay.push_back(token); }
        CHECK(replay.size() == 20);
        if (replay.size() != 20) { return; }
    }
    const int32_t vocabulary = llama_vocab_n_tokens(vocab);
    std::vector<std::vector<float>> frames;
    auto capture = [&] {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) { throw std::runtime_error("missing benchmark logits"); }
        frames.emplace_back(logits, logits + vocabulary);
        for (float value : frames.back()) { if (!std::isfinite(value)) { throw std::runtime_error("nonfinite benchmark logits"); } }
    };
    const int64_t prefill_start = ggml_time_us();
    common_batch batch = common_batch_get_one(ctx, tokens);
    CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    llama_synchronize(ctx);
    capture();
    const int64_t prefill_end = ggml_time_us();
    llama_tokens generated;
    int agreements = 0;
    const int64_t decode_start = ggml_time_us();
    for (int step = 0; step < 20; ++step) {
        const auto & logits = frames.back();
        const llama_token greedy = llama_token(std::max_element(logits.begin(), logits.end()) - logits.begin());
        const llama_token token = save ? greedy : replay[size_t(step)];
        agreements += save || greedy == token;
        generated.push_back(token);
        common_batch decode = common_batch_get_one(ctx, &token, 1);
        CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, decode.get()) == 0);
        llama_synchronize(ctx);
        capture();
    }
    const int64_t decode_end = ggml_time_us();
    CHECK(get_stats(captured_backend, &computed));
    CHECK(computed.preparations == initialized.preparations);
    if (save) {
        std::filesystem::create_directories(directory);
        std::ofstream token_file(directory / "tokens.txt");
        for (auto token : generated) { token_file << token << '\n'; }
    }
    double worst_nmse = 0;
    constexpr size_t chunk = 32768;
    for (size_t frame = 0; frame < frames.size(); ++frame) {
        double error = 0, energy = 0;
        for (size_t begin = 0; begin < size_t(vocabulary); begin += chunk) {
            const auto path = directory / ("frame_" + std::to_string(frame) + "_part_" + std::to_string(begin / chunk) + ".txt");
            const size_t end = std::min(begin + chunk, size_t(vocabulary));
            if (save) {
                std::ofstream file(path);
                if (!file) { throw std::runtime_error("cannot write reference"); }
                file << std::setprecision(9);
                for (size_t i = begin; i < end; ++i) { file << frames[frame][i] << '\n'; }
            } else {
                if (std::filesystem::file_size(path) > 1024 * 1024) { throw std::runtime_error("reference exceeds read limit"); }
                std::ifstream file(path);
                for (size_t i = begin; i < end; ++i) {
                    float reference;
                    if (!(file >> reference)) { throw std::runtime_error("incomplete reference"); }
                    const double difference = double(frames[frame][i]) - reference;
                    error += difference * difference; energy += double(reference) * reference;
                }
            }
        }
        if (!save) {
            const double nmse = error / std::max(energy, 1e-30);
            worst_nmse = std::max(worst_nmse, nmse);
            CHECK(nmse <= 1e-5);
        }
    }
    std::string response;
    for (auto token : generated) { response += common_token_to_piece(vocab, token); }
    std::printf("BENCH_PROMPT_TOKENS=%zu BENCH_DECODE_STEPS=20 INIT_GRAPHS=%llu QUERY_NEW_GRAPHS=%llu\n",
        tokens.size(), (unsigned long long) initialized.preparations,
        (unsigned long long) (computed.preparations - initialized.preparations));
    std::printf("BENCH_INIT_PREPARATION_MS=%.2f BENCH_INIT_FINALIZATION_MS=%.2f BENCH_PREFILL_TPS=%.4f BENCH_DECODE_TPS=%.4f\n",
        initialized.preparation_time_us / 1000.0, initialized.finalization_time_us / 1000.0,
        tokens.size() * 1e6 / double(prefill_end - prefill_start), 20e6 / double(decode_end - decode_start));
    std::printf("BENCH_EXPORTED_TENSORS=%llu BENCH_FP16_HANDOFFS=%llu BENCH_MATERIALIZED_OUTPUTS=%llu\n",
        (unsigned long long)(computed.exported_tensors - initialized.exported_tensors),
        (unsigned long long)(computed.fp16_handoffs - initialized.fp16_handoffs),
        (unsigned long long)(computed.materialized_outputs - initialized.materialized_outputs));
    std::printf("BENCH_GREEDY_AGREEMENTS=%d/20 BENCH_WORST_LOGITS_NMSE=%.9g\nBENCH_RESPONSE=%s\n",
        agreements, worst_nmse, response.c_str());
}

int main(int argc, char ** argv) {
    common_init();
    try {
        if (argc >= 2) {
            llama_backend_init();
            if (argc == 4 && (std::string(argv[2]) == "--benchmark-save" || std::string(argv[2]) == "--benchmark-check")) {
                boundary_benchmark(argv[1], argv[3], std::string(argv[2]) == "--benchmark-save");
            } else {
                model_test(argv[1], argc == 3 && std::string(argv[2]) == "--cpu");
            }
            llama_backend_free();
        } else {
            CHECK(parse_size("27"));
            CHECK(!parse_size("0"));
            CHECK(!parse_size("-1"));
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Warmup profile error: %s\n", error.what());
        ++failures;
    }
    std::printf("Warmup profile: %d failures\n", failures);
    return failures ? 1 : 0;
}
