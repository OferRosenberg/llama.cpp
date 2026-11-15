#include "qairt-target.h"
#include "qairt-runtime.h"
#include "qairt-graph.h"

#include <cstdio>
#include <memory>
#include <string>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <cstdlib>
#include <cerrno>
#include <HTP/QnnHtpDevice.h>

using namespace ggml::qairt;

static int checks = 0;
static int failures = 0;

#define CHECK(condition) do { \
    ++checks; \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        ++failures; \
    } \
} while (0)

static void test_configuration() {
    auto htp = config_parser::parse_target("HTP");
    CHECK(htp.ok());
    if (htp.ok()) { CHECK(htp.value() == target_kind::htp); }
    auto gpu = config_parser::parse_target("GPU");
    CHECK(gpu.ok());
    if (gpu.ok()) { CHECK(gpu.value() == target_kind::gpu); }
    auto cpu = config_parser::parse_target("CPU");
    CHECK(cpu.ok());
    if (cpu.ok()) { CHECK(cpu.value() == target_kind::cpu); }
    for (const char * token : {"", "htp", "AUTO", " HTP", "HTP ", "GPU0"}) {
        CHECK(!config_parser::parse_target(token).ok());
    }
    runtime_config config;
    CHECK(config.target == target_kind::htp);
    CHECK(!config_parser::validate(config).ok());
    config.library_directory = "relative";
    CHECK(!config_parser::validate(config).ok());
    config.library_directory = "C:\\QNN";
    CHECK(config_parser::validate(config).ok());
    CHECK(target_catalog::describe(target_kind::htp).adapter_enabled);
    CHECK(!target_catalog::describe(target_kind::gpu).adapter_enabled);
    CHECK(!target_catalog::describe(target_kind::cpu).adapter_enabled);
    CHECK(!target_catalog::create_policy(target_kind::gpu).ok());
    CHECK(!target_catalog::create_policy(target_kind::cpu).ok());
    auto policy = target_catalog::create_policy(target_kind::htp);
    CHECK(policy.ok());
    if (policy.ok()) {
        matmul_spec spec;
        CHECK(policy.value()->admit_matmul(spec).ok());
        spec.k = 0;
        CHECK(!policy.value()->admit_matmul(spec).ok());
        spec.k = 4;
        spec.dtype = QNN_DATATYPE_FLOAT_32;
        CHECK(!policy.value()->admit_matmul(spec).ok());
    }
}

static void test_providers() {
    CHECK(!qnn_api::from_providers(nullptr, 0).ok());
    QnnInterface_t provider = QNN_INTERFACE_INIT;
    provider.providerName = "fake";
    const QnnInterface_t * providers[] = { nullptr, &provider };
    auto selected = qnn_api::from_providers(providers, 2);
    CHECK(selected.ok());
    if (selected.ok()) {
        CHECK(selected.value().identity().provider_name == "fake");
        CHECK(!selected.value().validate_required_functions(false).ok());
    }
    ++provider.apiVersion.coreApiVersion.major;
    CHECK(!qnn_api::from_providers(providers, 2).ok());
    provider.apiVersion.coreApiVersion.major = QNN_API_VERSION_MAJOR;
    provider.apiVersion.coreApiVersion.minor = QNN_API_VERSION_MINOR - 1;
    CHECK(!qnn_api::from_providers(providers, 2).ok());
    CHECK(!dll_module::load("relative.dll").ok());
    CHECK(!dll_module::load("C:\\does-not-exist\\QnnHtp.dll").ok());
}

namespace ggml::qairt {
struct runtime_test_access {
    static std::shared_ptr<api_domain> domain(QNN_INTERFACE_VER_TYPE table) {
        QnnInterface_t provider = QNN_INTERFACE_INIT;
        provider.providerName = "fake-htp";
        provider.QNN_INTERFACE_VER_NAME = table;
        const QnnInterface_t * providers[] = { &provider };
        auto selected = qnn_api::from_providers(providers, 1);
        return std::shared_ptr<api_domain>(new api_domain({}, std::move(selected).value()));
    }
    static result<std::shared_ptr<runtime_owner>> runtime(std::shared_ptr<api_domain> domain, bool profile = false) {
        runtime_config config;
        config.library_directory = "C:\\QNN";
        config.enable_profile = profile;
        return runtime_factory::create(std::make_shared<const runtime_config>(config), std::move(domain));
    }
};
}

struct fake_state {
    error_stage fail_stage = error_stage::configuration;
    bool inject_failure = false;
    int finalized = 0;
    int executions = 0;
    int tensor_index = 0;
    int free_calls = 0;
    std::vector<int> teardown;
    std::vector<fp16_bits> weight;
};
static fake_state fake;
static int fake_handles[5];
static Qnn_ErrorHandle_t fake_code(error_stage stage) {
    return fake.inject_failure && fake.fail_stage == stage ? QNN_COMMON_ERROR_GENERAL : QNN_SUCCESS;
}
static Qnn_ErrorHandle_t fake_log(QnnLog_Callback_t, QnnLog_Level_t, Qnn_LogHandle_t * out) {
    *out = &fake_handles[0]; return fake_code(error_stage::logging);
}
static Qnn_ErrorHandle_t fake_backend(Qnn_LogHandle_t, const QnnBackend_Config_t **, Qnn_BackendHandle_t * out) {
    *out = &fake_handles[1]; return fake_code(error_stage::backend_create);
}
static Qnn_ErrorHandle_t fake_device(Qnn_LogHandle_t, const QnnDevice_Config_t **, Qnn_DeviceHandle_t * out) {
    *out = &fake_handles[2]; return fake_code(error_stage::device_create);
}
static Qnn_ErrorHandle_t fake_profile(Qnn_BackendHandle_t, QnnProfile_Level_t, Qnn_ProfileHandle_t * out) {
    *out = &fake_handles[3]; return fake_code(error_stage::profile_create);
}
static Qnn_ErrorHandle_t fake_context(Qnn_BackendHandle_t, Qnn_DeviceHandle_t, const QnnContext_Config_t **, Qnn_ContextHandle_t * out) {
    *out = &fake_handles[4]; return fake_code(error_stage::context_create);
}
static Qnn_ErrorHandle_t fake_log_free(Qnn_LogHandle_t) { fake.teardown.push_back(0); return QNN_SUCCESS; }
static Qnn_ErrorHandle_t fake_backend_free(Qnn_BackendHandle_t) { fake.teardown.push_back(1); return QNN_SUCCESS; }
static Qnn_ErrorHandle_t fake_device_free(Qnn_DeviceHandle_t) { fake.teardown.push_back(2); return QNN_SUCCESS; }
static Qnn_ErrorHandle_t fake_profile_free(Qnn_ProfileHandle_t) { fake.teardown.push_back(3); return QNN_SUCCESS; }
static Qnn_ErrorHandle_t fake_context_free(Qnn_ContextHandle_t, Qnn_ProfileHandle_t profile) {
    ++fake.free_calls;
    if (fake.inject_failure && fake.fail_stage == error_stage::cleanup) { return QNN_COMMON_ERROR_GENERAL; }
    CHECK(profile == nullptr || profile == &fake_handles[3]);
    fake.teardown.push_back(4); return QNN_SUCCESS;
}
static Qnn_ErrorHandle_t fake_events(Qnn_ProfileHandle_t, const QnnProfile_EventId_t ** events, uint32_t * count) {
    *events = nullptr; *count = 0; return QNN_SUCCESS;
}
static Qnn_ErrorHandle_t fake_graph(Qnn_ContextHandle_t, const char *, const QnnGraph_Config_t ** configs, Qnn_GraphHandle_t * out) {
    CHECK(configs && configs[0] && !configs[1]);
    *out = &fake_handles[4]; return fake_code(error_stage::graph_create);
}
static Qnn_ErrorHandle_t fake_tensor(Qnn_GraphHandle_t, Qnn_Tensor_t * tensor) {
    if (fake_code(error_stage::tensor_registration) != QNN_SUCCESS) { return fake_code(error_stage::tensor_registration); }
    static const uint32_t ids[] = { 0, 41, 7 };
    tensor->v1.id = fake.tensor_index < 3 ? ids[fake.tensor_index] : static_cast<uint32_t>(100 + fake.tensor_index);
    ++fake.tensor_index;
    if (tensor->v1.type == QNN_TENSOR_TYPE_STATIC) {
        const auto * data = static_cast<const fp16_bits *>(tensor->v1.clientBuf.data);
        fake.weight.assign(data, data + tensor->v1.clientBuf.dataSize / sizeof(fp16_bits));
    }
    return QNN_SUCCESS;
}
static Qnn_ErrorHandle_t fake_validate(Qnn_BackendHandle_t, Qnn_OpConfig_t op) {
    CHECK(std::strcmp(op.v1.typeName, "MatMul") == 0);
    CHECK(op.v1.inputTensors[0].v1.id == 0 && op.v1.inputTensors[1].v1.id == 41 && op.v1.outputTensors[0].v1.id == 7);
    CHECK(op.v1.numOfParams == 2 && !op.v1.params[0].scalarParam.bool8Value && op.v1.params[1].scalarParam.bool8Value);
    return fake_code(error_stage::op_validation);
}
static Qnn_ErrorHandle_t fake_add(Qnn_GraphHandle_t, Qnn_OpConfig_t) { return fake_code(error_stage::node_addition); }
static Qnn_ErrorHandle_t fake_finalize(Qnn_GraphHandle_t, Qnn_ProfileHandle_t, Qnn_SignalHandle_t) {
    ++fake.finalized; return fake_code(error_stage::graph_finalize);
}
static Qnn_ErrorHandle_t fake_execute(Qnn_GraphHandle_t, const Qnn_Tensor_t * inputs, uint32_t ni,
        Qnn_Tensor_t * outputs, uint32_t no, Qnn_ProfileHandle_t, Qnn_SignalHandle_t signal) {
    ++fake.executions;
    CHECK(ni == 1 && no == 1 && !signal);
    const auto * a = static_cast<const fp16_bits *>(inputs[0].v1.clientBuf.data);
    auto * y = static_cast<fp16_bits *>(outputs[0].v1.clientBuf.data);
    for (size_t m = 0; m < 2; ++m) {
        for (size_t n = 0; n < 3; ++n) {
            float sum = 0;
            for (size_t k = 0; k < 4; ++k) {
                sum += ggml_fp16_to_fp32(a[m * 4 + k].bits) * ggml_fp16_to_fp32(fake.weight[n * 4 + k].bits);
            }
            y[m * 3 + n].bits = ggml_fp32_to_fp16(sum);
        }
    }
    return fake_code(error_stage::graph_execute);
}
static QNN_INTERFACE_VER_TYPE fake_table() {
    QNN_INTERFACE_VER_TYPE table = QNN_INTERFACE_VER_TYPE_INIT;
    table.logCreate = fake_log; table.logFree = fake_log_free;
    table.backendCreate = fake_backend; table.backendFree = fake_backend_free;
    table.deviceCreate = fake_device; table.deviceFree = fake_device_free;
    table.profileCreate = fake_profile; table.profileFree = fake_profile_free; table.profileGetEvents = fake_events;
    table.contextCreate = fake_context; table.contextFree = fake_context_free;
    table.graphCreate = fake_graph; table.graphAddNode = fake_add;
    table.graphFinalize = fake_finalize; table.graphExecute = fake_execute;
    table.tensorCreateGraphTensor = fake_tensor; table.backendValidateOpConfig = fake_validate;
    return table;
}
static void test_runtime() {
    fake = {};
    auto domain = runtime_test_access::domain(fake_table());
    auto runtime = runtime_test_access::runtime(domain, true);
    CHECK(runtime.ok());
    if (!runtime.ok()) { return; }
    auto context = context_owner::create(runtime.value());
    CHECK(context.ok());
    if (!context.ok()) { return; }
    CHECK(!runtime.value()->close().ok());
    CHECK(context.value()->close().ok());
    CHECK(context.value()->close().ok());
    CHECK(runtime.value()->close().ok());
    CHECK(runtime.value()->close().ok());
    CHECK(fake.teardown == std::vector<int>({4, 3, 2, 1, 0}));
    CHECK(domain->statistics().acquired == 5 && domain->statistics().released == 5);
    for (auto stage : { error_stage::logging, error_stage::backend_create, error_stage::device_create, error_stage::profile_create }) {
        fake = {}; fake.inject_failure = true; fake.fail_stage = stage;
        domain = runtime_test_access::domain(fake_table());
        CHECK(!runtime_test_access::runtime(domain, true).ok());
        CHECK(domain->statistics().acquired == domain->statistics().released);
    }
    fake = {};
    domain = runtime_test_access::domain(fake_table());
    auto cleanup_runtime = runtime_test_access::runtime(domain);
    auto cleanup_context = context_owner::create(cleanup_runtime.value());
    fake.inject_failure = true; fake.fail_stage = error_stage::cleanup;
    CHECK(!cleanup_context.value()->close().ok());
    CHECK(!cleanup_context.value()->close().ok());
    CHECK(fake.free_calls == 1);
    CHECK(!cleanup_runtime.value()->close().ok());
    cleanup_context.value().reset();
    cleanup_runtime.value().reset();
    CHECK(fake.teardown.empty());
    CHECK(domain->statistics().release_failures == 1);
}

static void test_tensor_sizes() {
    tensor_spec spec { "A", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, {2, 4} };
    auto bytes = tensor_record::checked_bytes(spec);
    CHECK(bytes.ok() && bytes.value() == 16);
    spec.dimensions = {UINT32_MAX, UINT32_MAX};
    CHECK(!tensor_record::checked_bytes(spec).ok());
    spec.dimensions = {0, 4};
    CHECK(!tensor_record::checked_bytes(spec).ok());
    spec.dimensions = {2, 4}; spec.dtype = QNN_DATATYPE_FLOAT_32;
    CHECK(!tensor_record::checked_bytes(spec).ok());
    spec.dtype = QNN_DATATYPE_FLOAT_16; spec.role = QNN_TENSOR_TYPE_APP_READWRITE;
    CHECK(!tensor_record::checked_bytes(spec).ok());
}

static void test_platform_and_handles() {
    htp_target_policy policy;
    QnnDevice_PlatformInfo_t info = QNN_DEVICE_PLATFORM_INFO_INIT;
    auto empty = policy.read_platform(info);
    CHECK(empty.ok() && !empty.value().soc_model);
    info.v1.numHwDevices = 1;
    CHECK(!policy.read_platform(info).ok());
    QnnDevice_HardwareDeviceInfo_t device = QNN_DEVICE_HARDWARE_DEVICE_INFO_INIT;
    QnnHtpDevice_DeviceInfoExtension_t extension {};
    extension.devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
    extension.onChipDevice.socModel = 0x7fffffff;
    extension.onChipDevice.arch = static_cast<QnnHtpDevice_Arch_t>(73);
    device.v1.deviceInfoExtension = &extension;
    info.v1.hwDevices = &device;
    auto unknown = policy.read_platform(info);
    CHECK(unknown.ok() && !unknown.value().soc_model && unknown.value().htp_architecture == 73);

    fake = {};
    auto domain = runtime_test_access::domain(fake_table());
    unique_qnn_handle<log_tag> first;
    first.adopt_into_empty(&fake_handles[0], domain);
    unique_qnn_handle<log_tag> second(std::move(first));
    CHECK(!first && second);
    CHECK(second.close().ok());
    CHECK(second.close().ok());
    CHECK(domain->statistics().acquired == 1 && domain->statistics().released == 1);
    static_assert(!std::is_copy_constructible<unique_qnn_handle<log_tag>>::value);
    static_assert(!std::is_move_assignable<unique_qnn_handle<log_tag>>::value);
    static_assert(!std::is_move_constructible<runtime_owner>::value);
    static_assert(!std::is_move_constructible<context_owner>::value);
}

class matmul_fixture final {
public:
    matmul_fixture() {
        const float weights[] = {1, 0, 0, 1, 0, 1, 1, 0, 1, -1, 1, -1};
        for (size_t i = 0; i < weight_.size(); ++i) { weight_[i].bits = ggml_fp32_to_fp16(weights[i]); }
        set_scale(1);
    }
    void set_scale(float scale) {
        scale_ = scale;
        const float input[] = {1, 2, 3, 4, -1, 0, 1, 2};
        for (size_t i = 0; i < input_.size(); ++i) { input_[i].bits = ggml_fp32_to_fp16(input[i] * scale); }
        for (auto & value : output_) { value.bits = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN()); }
    }
    array_view<const fp16_bits> input() const { return { input_.data(), input_.size() }; }
    array_view<const fp16_bits> weight() const { return { weight_.data(), weight_.size() }; }
    array_view<fp16_bits> output() { return { output_.data(), output_.size() }; }
    status verify() const {
        const float expected[] = {5, 5, -2, 1, 1, -2};
        for (size_t i = 0; i < output_.size(); ++i) {
            const float actual = ggml_fp16_to_fp32(output_[i].bits);
            if (!std::isfinite(actual) || std::fabs(actual - expected[i] * scale_) > 0.001f) {
                return status::failure(host_error(error_stage::numerical_check, "MatMul", "FP16 output differs from independent reference"));
            }
        }
        return status::success();
    }
private:
    float scale_ = 1;
    std::array<fp16_bits, 8> input_ {};
    std::array<fp16_bits, 12> weight_ {};
    std::array<fp16_bits, 6> output_ {};
};

static status exercise_graph(std::shared_ptr<runtime_owner> runtime, bool negative_cases = false) {
    matmul_fixture fixture;
    auto context = context_owner::create(runtime);
    if (!context.ok()) { return status::failure(context.error()); }
    auto graph = graph_owner::create(context.value(), "qairt_m0_matmul");
    if (!graph.ok()) { return status::failure(graph.error()); }
    auto a = graph.value()->register_tensor({ "A", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, {2, 4} });
    if (!a.ok()) { return status::failure(a.error()); }
    const auto * first_record = graph.value()->find_tensor(a.value());
    const auto * first_name = first_record->spec().name.c_str();
    auto weight = fixture.weight();
    auto w = graph.value()->register_tensor({ "W", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, {3, 4} },
        { weight.data(), weight.size() * sizeof(fp16_bits) });
    if (!w.ok()) { return status::failure(w.error()); }
    auto y = graph.value()->register_tensor({ "Y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, {2, 3} });
    if (!y.ok()) { return status::failure(y.error()); }
    if (negative_cases) {
        CHECK(first_record == graph.value()->find_tensor(a.value()));
        CHECK(first_name == first_record->spec().name.c_str());
        CHECK(!context.value()->close().ok());
        CHECK(graph.value()->find_tensor({123456}) == nullptr);
        CHECK(graph.value()->find_tensor(a.value()) != nullptr);
        CHECK(a.value().value == 0);
        CHECK(!graph.value()->register_tensor({ "A", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, {2, 4} }).ok());
        CHECK(!execution_binding::create(graph.value()).ok());
    }
    auto added = graph.value()->add_node(op_spec::matmul(a.value(), w.value(), y.value()));
    if (!added.ok()) { return added; }
    auto finalized = graph.value()->finalize();
    if (!finalized.ok()) { return finalized; }
    auto binding = execution_binding::create(graph.value());
    if (!binding.ok()) { return status::failure(binding.error()); }
    if (negative_cases) {
        CHECK(!binding.value()->validate_complete().ok());
        CHECK(!graph.value()->execute(*binding.value()).ok());
        CHECK(!graph.value()->close().ok());
        CHECK(!binding.value()->bind_input(w.value(), fixture.input()).ok());
        CHECK(!binding.value()->bind_input(a.value(), { static_cast<const void *>(fixture.input().data()), size_t(2) }).ok());
        CHECK(!binding.value()->bind_input(a.value(), array_view<const fp16_bits>(fixture.input().data(), SIZE_MAX)).ok());
        auto overlap_binding = execution_binding::create(graph.value());
        CHECK(overlap_binding.ok());
        CHECK(overlap_binding.value()->bind_input(a.value(), fixture.input()).ok());
        CHECK(overlap_binding.value()->bind_output(y.value(), buffer_view { const_cast<fp16_bits *>(fixture.input().data()), 12 }).ok());
        CHECK(!overlap_binding.value()->validate_complete().ok());
    }
    auto bound = binding.value()->bind_input(a.value(), fixture.input());
    if (!bound.ok()) { return bound; }
    bound = binding.value()->bind_output(y.value(), fixture.output());
    if (!bound.ok()) { return bound; }
    if (negative_cases) { CHECK(!binding.value()->bind_input(a.value(), fixture.input()).ok()); }
    for (float scale : {1.f, 2.f, 1.f}) {
        fixture.set_scale(scale);
        auto executed = graph.value()->execute(*binding.value());
        if (!executed.ok()) { return executed; }
        auto verified = fixture.verify();
        if (!verified.ok()) { return verified; }
    }
    binding.value().reset();
    auto closed = graph.value()->close();
    if (!closed.ok()) { return closed; }
    return context.value()->close();
}

static void test_graphs() {
    fake = {};
    auto domain = runtime_test_access::domain(fake_table());
    auto runtime = runtime_test_access::runtime(domain);
    CHECK(runtime.ok());
    if (!runtime.ok()) { return; }
    CHECK(exercise_graph(runtime.value(), true).ok());
    CHECK(fake.finalized == 1 && fake.executions == 3);
    CHECK(runtime.value()->close().ok());
    CHECK(domain->statistics().acquired == domain->statistics().released);
    for (auto stage : { error_stage::context_create, error_stage::graph_create, error_stage::tensor_registration,
                        error_stage::op_validation, error_stage::node_addition, error_stage::graph_finalize, error_stage::graph_execute }) {
        fake = {};
        domain = runtime_test_access::domain(fake_table());
        runtime = runtime_test_access::runtime(domain);
        fake.inject_failure = true; fake.fail_stage = stage;
        auto exercised = exercise_graph(runtime.value());
        CHECK(!exercised.ok());
        if (!exercised.ok()) { CHECK(exercised.error().stage == stage); }
        CHECK(runtime.value()->close().ok());
        CHECK(domain->statistics().acquired == domain->statistics().released);
        if (stage != error_stage::graph_execute) { CHECK(fake.executions == 0); }
    }
}

static void print_error(const qairt_error & error) {
    std::fprintf(stderr, "QAIRT stage=%d operation=%s: %s", static_cast<int>(error.stage), error.operation.c_str(), error.message.c_str());
    if (error.qnn_error) { std::fprintf(stderr, " QNN=0x%llx", static_cast<unsigned long long>(*error.qnn_error)); }
    if (error.windows_error) { std::fprintf(stderr, " Win32=%u", *error.windows_error); }
    std::fprintf(stderr, "\n");
}

static int hardware_test(uint32_t iterations, bool profile, bool verbose) {
    auto captured = process_config::capture_once();
    if (!captured.ok()) { print_error(captured.error()); return 1; }
    auto config = std::make_shared<runtime_config>(*captured.value());
    config->enable_profile = profile;
    config->verbose_diagnostics = verbose;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        auto runtime = runtime_factory::create(config);
        if (!runtime.ok()) { print_error(runtime.error()); return 1; }
        const auto & identity = runtime.value()->domain()->identity();
        const auto & platform = runtime.value()->platform();
        std::printf("QAIRT %s core API %u.%u.%u, target HTP", identity.provider_name.c_str(),
            identity.version.coreApiVersion.major, identity.version.coreApiVersion.minor, identity.version.coreApiVersion.patch);
        if (platform.soc_model) { std::printf(", SoC %u", *platform.soc_model); }
        if (platform.htp_architecture) { std::printf(", HTP architecture %u", *platform.htp_architecture); }
        std::printf("\n");
        auto exercised = exercise_graph(runtime.value());
        if (!exercised.ok()) { print_error(exercised.error()); return 1; }
        if (profile) {
            auto events = runtime.value()->profile_event_count();
            if (!events.ok()) { print_error(events.error()); return 1; }
            std::printf("QAIRT profile events: %u\n", events.value());
        }
        auto domain = runtime.value()->domain();
        auto closed = runtime.value()->close();
        if (!closed.ok()) { print_error(closed.error()); return 1; }
        const auto & counts = domain->statistics();
        if (counts.acquired != counts.released || counts.release_failures) {
            std::fprintf(stderr, "QAIRT resource accounting failed\n"); return 1;
        }
        std::printf("QAIRT lifecycle %u/%u passed: 3 FP16 executions, changed inputs, cleanup %llu/%llu\n",
            iteration + 1, iterations, static_cast<unsigned long long>(counts.released.load()),
            static_cast<unsigned long long>(counts.acquired.load()));
    }
    return 0;
}

static int capture_test(bool failure) {
    _putenv_s("GGML_QAIRT_DEVICE", failure ? "invalid" : "");
    _wputenv_s(L"GGML_QAIRT_LIBRARY_DIR", L"C:\\QNN");
    auto first = process_config::capture_once();
    CHECK(first.ok() == !failure);
    _putenv_s("GGML_QAIRT_DEVICE", "GPU");
    _wputenv_s(L"GGML_QAIRT_LIBRARY_DIR", L"C:\\other");
    auto second = process_config::capture_once();
    CHECK(second.ok() == first.ok());
    if (first.ok() && second.ok()) {
        CHECK(first.value().get() == second.value().get());
        CHECK(second.value()->target == target_kind::htp);
        CHECK(second.value()->library_directory == "C:\\QNN");
    } else if (!first.ok() && !second.ok()) {
        CHECK(first.error().message == second.error().message);
        CHECK(first.error().stage == second.error().stage);
    }
    std::printf("QAIRT configuration capture checks: %d, failures: %d\n", checks, failures);
    return failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--host-tests") {
            test_configuration();
            test_providers();
            test_runtime();
            test_tensor_sizes();
            test_platform_and_handles();
            test_graphs();
            std::printf("QAIRT host checks: %d, failures: %d\n", checks, failures);
            return failures ? 1 : 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--capture-success") { return capture_test(false); }
        if (argc == 2 && std::string(argv[1]) == "--capture-failure") { return capture_test(true); }
        if (argc >= 2 && std::string(argv[1]) == "--hardware-test") {
            uint32_t iterations = 3;
            bool profile = false;
            bool verbose = false;
            for (int i = 2; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--profile") { profile = true; }
                else if (arg == "--verbose") { verbose = true; }
                else if (arg == "--binary-round-trip") {
                    std::fprintf(stderr, "QAIRT: optional binary round trip is not implemented in M0\n"); return 2;
                } else if (arg == "--iterations" && i + 1 < argc) {
                    char * end = nullptr;
                    errno = 0;
                    const auto count = std::strtoul(argv[++i], &end, 10);
                    if (errno || !end || *end || count < 1 || count > 100) {
                        std::fprintf(stderr, "QAIRT iterations must be between 1 and 100\n"); return 2;
                    }
                    iterations = static_cast<uint32_t>(count);
                } else {
                    std::fprintf(stderr, "QAIRT unknown or incomplete option: %s\n", argv[i]); return 2;
                }
            }
            return hardware_test(iterations, profile, verbose);
        }
        std::fprintf(stderr, "Usage: test-qairt-smoke --host-tests | --hardware-test [--iterations N] [--profile] [--verbose]\n");
        return 2;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "QAIRT test exception: %s\n", error.what());
        return 1;
    }
}
