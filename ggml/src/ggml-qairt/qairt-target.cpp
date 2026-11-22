#include "qairt-target.h"

#include <cstdlib>
#include <mutex>
#include <array>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>

namespace ggml::qairt {

class htp_config_storage final : public target_config_storage {
public:
    htp_config_storage() {
        custom_.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        custom_.precision = QNN_PRECISION_FLOAT16;
        graph_.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_.customConfig = &custom_;
        pointers_ = { &graph_, nullptr };
    }
    htp_config_storage(const htp_config_storage &) = delete;
    htp_config_storage & operator=(const htp_config_storage &) = delete;
    const QnnGraph_Config_t ** graph() noexcept override { return pointers_.data(); }
private:
    QnnHtpGraph_CustomConfig_t custom_ = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    QnnGraph_Config_t graph_ = QNN_GRAPH_CONFIG_INIT;
    std::array<const QnnGraph_Config_t *, 2> pointers_ {};
};

result<target_kind> config_parser::parse_target(std::string_view value) {
    if (value == "HTP") { return result<target_kind>::success(target_kind::htp); }
    if (value == "GPU") { return result<target_kind>::success(target_kind::gpu); }
    if (value == "CPU") { return result<target_kind>::success(target_kind::cpu); }
    return result<target_kind>::failure(host_error(error_stage::configuration, "parse_target", "Expected HTP, GPU or CPU"));
}

status config_parser::validate(const runtime_config & config) {
    if (config.library_directory.empty() || !config.library_directory.is_absolute()) {
        return status::failure(host_error(error_stage::configuration, "library_directory", "An absolute GGML_QAIRT_LIBRARY_DIR is required", config.target));
    }
    if (config.target != target_kind::htp && config.target != target_kind::gpu && config.target != target_kind::cpu) {
        return status::failure(host_error(error_stage::configuration, "target", "Invalid target", config.target));
    }
    if (config.enable_w4fp16 && config.target != target_kind::htp) {
        return status::failure(host_error(error_stage::configuration, "GGML_QAIRT_W4FP16", "W4FP16 requires HTP", config.target));
    }
    if (config.app_write_weights && config.enable_w4fp16) {
        return status::failure(host_error(error_stage::configuration, "GGML_QAIRT_APP_WRITE_WEIGHTS", "HTP BW_FLOAT_BLOCK weights require STATIC tensors", config.target));
    }
    if (config.share_weight_graphs && !config.app_write_weights) {
        return status::failure(host_error(error_stage::configuration, "GGML_QAIRT_SHARE_WEIGHT_GRAPHS", "Graph sharing requires APP_WRITE weights", config.target));
    }
    return status::success();
}

result<runtime_config> config_parser::from_environment() {
    runtime_config config;
    const char * target = std::getenv("GGML_QAIRT_DEVICE");
    if (target) {
        auto parsed = parse_target(target);
        if (!parsed.ok()) { return result<runtime_config>::failure(parsed.error()); }
        config.target = parsed.value();
    }
    const wchar_t * directory = _wgetenv(L"GGML_QAIRT_LIBRARY_DIR");
    if (directory) { config.library_directory = directory; }
    for (const auto & option : { std::make_pair("GGML_QAIRT_PROFILE", &config.enable_profile),
                                std::make_pair("GGML_QAIRT_DIAGNOSTICS", &config.verbose_diagnostics),
                                std::make_pair("GGML_QAIRT_W4FP16", &config.enable_w4fp16),
                                std::make_pair("GGML_QAIRT_APP_WRITE_WEIGHTS", &config.app_write_weights),
                                std::make_pair("GGML_QAIRT_SHARE_WEIGHT_GRAPHS", &config.share_weight_graphs),
                                std::make_pair("GGML_QAIRT_SHARED_CONTEXT", &config.shared_context) }) {
        const char * value = std::getenv(option.first);
        if (!value) { continue; }
        if (std::string_view(value) != "0" && std::string_view(value) != "1") {
            return result<runtime_config>::failure(host_error(error_stage::configuration, option.first, "Expected 0 or 1"));
        }
        *option.second = std::string_view(value) == "1";
    }
    auto checked = validate(config);
    if (!checked.ok()) { return result<runtime_config>::failure(checked.error()); }
    return result<runtime_config>::success(std::move(config));
}

result<std::shared_ptr<const runtime_config>> process_config::capture_once() {
    static std::once_flag once;
    static std::optional<result<std::shared_ptr<const runtime_config>>> captured;
    std::call_once(once, [] {
        auto config = config_parser::from_environment();
        if (config.ok()) {
            captured.emplace(result<std::shared_ptr<const runtime_config>>::success(
                std::make_shared<const runtime_config>(std::move(config).value())));
        } else {
            captured.emplace(result<std::shared_ptr<const runtime_config>>::failure(config.error()));
        }
    });
    return *captured;
}

target_descriptor target_catalog::describe(target_kind target) {
    switch (target) {
        case target_kind::htp: return { target, "QNN HTP", "QnnHtp.dll", true };
        case target_kind::gpu: return { target, "QNN GPU", "QnnGpu.dll", false };
        case target_kind::cpu: return { target, "QNN CPU", "QnnCpu.dll", false };
    }
    return { target, "Unknown", "", false };
}

result<std::unique_ptr<target_policy>> target_catalog::create_policy(target_kind target) {
    if (target != target_kind::htp) {
        return result<std::unique_ptr<target_policy>>::failure(host_error(
            error_stage::configuration, "create_policy", "Target adapter not enabled", target));
    }
    return result<std::unique_ptr<target_policy>>::success(std::make_unique<htp_target_policy>());
}

const target_descriptor & htp_target_policy::descriptor() const noexcept { return descriptor_; }
status htp_target_policy::admit_matmul(const matmul_spec & spec) const {
    if (!spec.m || !spec.k || !spec.n || uint64_t(spec.m) * spec.k > UINT32_MAX / 2 ||
        uint64_t(spec.n) * spec.k > UINT32_MAX / 2 || uint64_t(spec.m) * spec.n > UINT32_MAX / 2 ||
        spec.transpose_input0 || !spec.transpose_input1 ||
        spec.dtype != QNN_DATATYPE_FLOAT_16) {
        return status::failure(host_error(error_stage::op_validation, "admit_matmul", "Unsupported FP16 MatMul dimensions or transpose policy"));
    }
    return status::success();
}

result<std::unique_ptr<target_config_storage>> htp_target_policy::build_configs() const {
    return result<std::unique_ptr<target_config_storage>>::success(std::make_unique<htp_config_storage>());
}

result<platform_snapshot> htp_target_policy::read_platform(const QnnDevice_PlatformInfo_t & info) const {
    platform_snapshot snapshot;
    if (info.version != QNN_DEVICE_PLATFORM_INFO_VERSION_1 || info.v1.numHwDevices > 64 ||
        (info.v1.numHwDevices && !info.v1.hwDevices)) {
        return result<platform_snapshot>::failure(host_error(error_stage::platform_query, "platformInfo", "Invalid platform metadata"));
    }
    if (info.v1.numHwDevices) {
        const auto & device = info.v1.hwDevices[0];
        if (device.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1 || device.v1.numCores > 64 ||
            (device.v1.numCores && !device.v1.cores)) {
            return result<platform_snapshot>::failure(host_error(error_stage::platform_query, "deviceInfo", "Invalid device metadata"));
        }
        snapshot.device_id = device.v1.deviceId;
        if (device.v1.numCores && device.v1.cores[0].version == QNN_DEVICE_CORE_INFO_VERSION_1) {
            snapshot.core_id = device.v1.cores[0].v1.coreId;
        }
        const auto * extension = device.v1.deviceInfoExtension;
        if (extension && extension->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
            const auto soc = extension->onChipDevice.socModel;
            const auto arch = static_cast<uint32_t>(extension->onChipDevice.arch);
            // The Windows user driver can report the generic undefined sentinel for SoC.
            if (soc && soc != 0x7fffffff) { snapshot.soc_model = soc; }
            if (arch && arch != 0x7fffffff) { snapshot.htp_architecture = arch; }
        }
    }
    return result<platform_snapshot>::success(snapshot);
}

}
