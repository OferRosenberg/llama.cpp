#pragma once

#include "qairt-status.h"

#include <filesystem>
#include <memory>
#include <string_view>

namespace ggml::qairt {

struct runtime_config {
    target_kind target = target_kind::htp;
    std::filesystem::path library_directory;
    bool enable_w4fp16 = false;
    bool app_write_weights = false;
    bool share_weight_graphs = false;
    bool shared_context = false;
    bool enable_profile = false;
    bool verbose_diagnostics = false;
};

struct target_descriptor {
    target_kind kind;
    std::string display_name;
    std::string provider_filename;
    bool adapter_enabled;
    uint32_t policy_version = 1;
};

struct matmul_spec {
    uint32_t m = 2;
    uint32_t k = 4;
    uint32_t n = 3;
    bool transpose_input0 = false;
    bool transpose_input1 = true;
    Qnn_DataType_t dtype = QNN_DATATYPE_FLOAT_16;
};

struct platform_snapshot {
    std::optional<uint32_t> device_id;
    std::optional<uint32_t> core_id;
    std::optional<uint32_t> soc_model;
    std::optional<uint32_t> htp_architecture;
};

class target_config_storage {
public:
    virtual ~target_config_storage() = default;
    virtual const QnnBackend_Config_t ** backend() noexcept { return nullptr; }
    virtual const QnnDevice_Config_t ** device() noexcept { return nullptr; }
    virtual const QnnContext_Config_t ** context() noexcept { return nullptr; }
    virtual const QnnGraph_Config_t ** graph() noexcept = 0;
};

class config_parser final {
public:
    static result<runtime_config> from_environment();
    static result<target_kind> parse_target(std::string_view value);
    static status validate(const runtime_config & config);
};

class process_config final {
public:
    static result<std::shared_ptr<const runtime_config>> capture_once();
};

class target_policy {
public:
    virtual ~target_policy() = default;
    virtual const target_descriptor & descriptor() const noexcept = 0;
    virtual status admit_matmul(const matmul_spec & spec) const = 0;
    virtual result<std::unique_ptr<target_config_storage>> build_configs() const = 0;
    virtual result<platform_snapshot> read_platform(const QnnDevice_PlatformInfo_t & info) const = 0;
};

class htp_target_policy final : public target_policy {
public:
    const target_descriptor & descriptor() const noexcept override;
    status admit_matmul(const matmul_spec & spec) const override;
    result<std::unique_ptr<target_config_storage>> build_configs() const override;
    result<platform_snapshot> read_platform(const QnnDevice_PlatformInfo_t & info) const override;
private:
    target_descriptor descriptor_ { target_kind::htp, "QNN HTP", "QnnHtp.dll", true };
};

class target_catalog final {
public:
    static target_descriptor describe(target_kind target);
    static result<std::unique_ptr<target_policy>> create_policy(target_kind target);
};

}
