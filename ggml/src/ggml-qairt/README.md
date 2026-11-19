# QAIRT backend

Current connected-graph checkpoint: shared qairt-partition.cpp lowering joins operators, branches, layout transforms and KV ScatterNd. Block 27 plus final norm/scale execute once per shape profile in --fused Qwen3 validation. Numerical QNN tensors are FP16. Functional suites pass; model local/intermediate accuracy remains unresolved. See docs/backend/QAIRT.md.


The registered ggml-qairt backend now supports expanded FP16 HTP recipes for projections, normalization/scales, residuals, SiLU/SwiGLU, RoPE, masked Softmax, grouped attention, Gather and KV ScatterNd. Numerical QNN tensors use FP16; index/control tensors use SDK-required integer types. GPU/CPU target policies remain architecturally present but disabled.

See [current build, support, ownership and accuracy status](../../../docs/backend/QAIRT.md). The M2 Full FP16 checkpoint builds on Windows ARM64 and passes all 14 QAIRT CTests. All 28 Qwen3 blocks are covered by the per-block hybrid fixture, with identical 20-token greedy output against the FP16-emulated CPU reference. Numerical accuracy exceptions, CPU fallbacks and fixture limits remain documented.

## Historical M0 runtime-core record

The following sections preserve the initial M0 implementation and its validation. Their future-tense M1 references describe that earlier checkpoint.

This directory implements the first QAIRT integration milestone: constructing, finalizing and executing a direct FP16 MatMul graph through the released QNN API. It is an internal static library, not a registered GGML backend. GGUF loading, GGML buffers, scheduler offload, model lowering and KV cache integration start in M1.

Target configuration supports `HTP`, `GPU` and `CPU`. HTP is the default and the only enabled execution policy in M0. GPU/CPU selections fail before QNN initialization. No target fallback is performed by this core. QNN itself may select an HTP transport or driver path appropriate to the machine.

## Build on Windows ARM64

Use the native ARM64 MSVC toolchain and an installed QAIRT SDK. No generated model library, Python converter or QNN import library is required. SDK headers remain private; the HTP provider is loaded at runtime.

```powershell
cmake -S C:\Github\llama.cpp -B C:\Github\qairt-m0-arm64 -G "Visual Studio 17 2022" -A ARM64 -DGGML_QAIRT=ON -DGGML_QAIRT_TESTS=ON -DLLAMA_BUILD_TESTS=ON -DQAIRT_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.52.0.261104 -DGGML_OPENMP=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build C:\Github\qairt-m0-arm64 --config Release --parallel 10
ctest --test-dir C:\Github\qairt-m0-arm64 -C Release -R qairt --output-on-failure
```

`GGML_QAIRT` defaults OFF. `GGML_QAIRT_TESTS` builds one smoke executable and registers host/fresh-process configuration tests. Host cases use a fake QNN interface table and need no SDK runtime DLLs. `GGML_QAIRT_HW_TESTS` defaults OFF; enable it explicitly to register the HTP CTest case. Test options require `LLAMA_BUILD_COMMON=ON` and `LLAMA_BUILD_TESTS=ON`.

The internal `ggml-qairt-core` target remains STATIC with `GGML_BACKEND_DL=ON`. M1 will add the actual registered backend through the ordinary GGML helper. The M0 Windows loader is the only implemented platform loader.

## Hardware smoke test

```powershell
$env:GGML_QAIRT_DEVICE = 'HTP'
$env:GGML_QAIRT_LIBRARY_DIR = 'C:\Qualcomm\AIStack\QAIRT\2.52.0.261104\lib\aarch64-windows-msvc'
& C:\Github\qairt-m0-arm64\bin\Release\test-qairt-smoke.exe --hardware-test --iterations 3
```

The target/directory are captured once per process, including a failed capture. `--profile` enables a basic profile handle and reports the returned event count. `--verbose` enables debug logging. `--iterations` accepts 1 through 100. The optional binary round-trip experiment is deferred; requesting it returns a distinct nonzero result. Full event extraction and timing analysis are also deferred.

Each lifecycle builds A `[2,4]`, STATIC W `[3,4]`, and Y `[2,3]`, then computes `Y = A * W^T`. One finalized graph executes at input scales 1, 2 and 1. The fixed reference is `[[5,5,-2],[1,1,-2]]` at scale 1; FP16 output must be finite and within 0.001 of the scaled reference. Cleanup checks successful handle acquisition/free counts; these counts do not prove reclamation of all driver memory.

## Implementation boundaries

| File | Responsibility |
| --- | --- |
| `qairt-status.h` | Stage-specific errors and bounded backend-private result types |
| `qairt-target.*` | Strict configuration, process capture, target policy, HTP precision settings and copied platform metadata |
| `qairt-runtime.*` | Windows provider loading, API negotiation, serialized calls and resource ownership |
| `qairt-tensor.*` | Stable FP16 descriptor storage, checked sizes and RAW buffer validation |
| `qairt-graph.*` | Context/graph ownership, MatMul packets, finalization and synchronous execution |
| `tests/test-qairt-smoke.cpp` | Numerical fixture, hardware runner, fake-table failures and configuration subprocess cases |

All owner creation, close, binding mutation and execution must be sequenced by the caller. The QNN call mutex does not make owner lifecycle or graph metadata thread-safe. Child counters guard explicit close; retained parent references protect ordinary destruction. Runtime/context owners cannot be copied or moved.

SDK cleanup failure has uncertain post-failure handle state. A handle is never freed twice after such a failure. The custom runtime/context deleter retains the already allocated owner and its remaining ancestors until process exit, avoiding allocation during failure cleanup. The smoke test reports failure rather than claiming successful teardown. This deliberate retention also applies in the host failure-injection case.

Private inheritance, overloads and templates follow the approved object-oriented design. They are limited to target policy, typed resource ownership and views/results; no C++ API is exported from GGML. CPU/GPU policy implementations, asynchronous calls, concurrent lifecycle support, registered memory and persistent caching are deferred.

## Local validation: November 14, 2026

Validated against local llama.cpp commit `d90cfd4a9`, QAIRT SDK `2.52.0.261104`, QNN core API `2.41.0`, and native ARM64 MSVC `19.41.34123.0`.

| Configuration or check | Result |
| --- | --- |
| `C:\Github\qairt-m0-arm64`, QAIRT enabled, `GGML_BACKEND_DL=OFF` | Complete configured Release build succeeded, including llama CLI/server, examples, tests and M0 core |
| Same configuration, QAIRT CTest group with hardware enabled | 4/4 tests passed: host, capture success, capture failure and HTP |
| Host cases | 160 checks passed; parser, provider negotiation, ownership, failure injection, IDs, descriptors, overflow, overlap and changed inputs |
| HTP required fixture | Three complete lifecycles, nine FP16 executions, four owned handles acquired/released per lifecycle |
| HTP optional basic profiling | Three FP16 executions; four profile events returned; five handles acquired/released |
| `C:\Github\qairt-m0-arm64-dl`, `GGML_BACKEND_DL=ON` | M0 core/smoke target built and all three host/configuration CTest entries passed |
| `C:\Github\qairt-m0-arm64-off`, QAIRT disabled and no SDK configured | GGML/CPU Release build succeeded |
| Existing `test-backend-tensor` | Passed |
| `llama-cli --version` | Ran successfully and reported ARM64/MSVC |
| Changed-source whitespace, ASCII and newline checks; `git diff --check` | Passed; the external EditorConfig checker was not installed |

The provider reported HTP architecture 73 and an unspecified SoC. SDK diagnostics reported failure of its traditional transport and some unsupported driver options/events, then selected the HTP user-driver path. Required MatMul and basic profile checks still passed. These results establish the tiny M0 fixture only; transformer graphs, GGUF quantization, KV state and complete profiling compatibility require later milestones.

The complete build emitted existing compiler/build warnings and built without embedded Web UI assets. UI downloads were disabled for subsequent builds. No QAIRT source compilation error remains. M0 is recorded in local commit `05e54784b`. M1 adds the registered backend; see [current setup and support](../../../docs/backend/QAIRT.md).
