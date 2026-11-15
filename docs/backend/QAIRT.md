# QAIRT backend (M1)

The optional ggml-qairt backend uses the released QNN C API on Windows ARM64. M1 validates a bounded set of float primitives on HTP. Whole-model GGUF execution, Q4_K/Q6_K conversion, attention/KV and persistent contexts are later milestones.

## Build

~~~powershell
cmake -S C:\Github\llama.cpp -B C:\Github\qairt-m1-arm64 -G "Visual Studio 17 2022" -A ARM64 -DGGML_QAIRT=ON -DGGML_QAIRT_TESTS=ON -DGGML_QAIRT_HW_TESTS=ON -DLLAMA_BUILD_TESTS=ON -DQAIRT_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.52.0.261104 -DGGML_OPENMP=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build C:\Github\qairt-m1-arm64 --config Release --parallel 10
~~~

GGML_QAIRT defaults OFF. An OFF build needs no QAIRT SDK. SDK headers stay private; provider DLLs are loaded from the configured runtime directory. The M0 core remains an internal static library; the registered backend uses ordinary GGML static/shared/module integration. GGML_BACKEND_DL=ON requires BUILD_SHARED_LIBS=ON. Windows is currently the implemented loader platform. Installed static-library packaging is not validated by this milestone; use the source-tree CMake targets for integration.

## Configuration

~~~powershell
$env:GGML_QAIRT_DEVICE = 'HTP'
$env:GGML_QAIRT_LIBRARY_DIR = 'C:\Qualcomm\AIStack\QAIRT\2.52.0.261104\lib\aarch64-windows-msvc'
& C:\Github\qairt-m1-arm64\bin\Release\llama-cli.exe --list-devices
ctest --test-dir C:\Github\qairt-m1-arm64 -C Release -R '^qairt-' --output-on-failure
~~~

GGML_QAIRT_DEVICE accepts strict uppercase HTP, GPU or CPU; HTP is default. Only HTP's adapter is enabled in M1. Selecting GPU/CPU produces an explicit disabled-adapter diagnostic and no usable QAIRT device. A present DLL alone does not enable a target. Configuration, including capture failure, is immutable after first capture. GGML_QAIRT_LIBRARY_DIR must be an absolute native directory, not a DLL path. No SDK build-machine path is embedded as a runtime default.

Optional GGML_QAIRT_PROFILE=0|1 enables basic QNN event-count profiling. GGML_QAIRT_DIAGNOSTICS=0|1 enables runtime detail and rejected-partition operation/type diagnostics. Other values are configuration errors.

One device, QAIRT0, is exposed after a runtime discovery probe succeeds and closes cleanly. Physical HTP and discovered architecture appear in its description. Its GPU device class follows OpenVINO's selected-target model-offload convention; it is not a claim about the physical processor. Memory availability is reported as unknown (zero), rather than inventing a dedicated device-memory capacity. QNN CPU will remain distinct from native ggml CPU.

## Implemented support

| Operation | Boundary and shape contract | Implementation |
| --- | --- | --- |
| MUL_MAT | F32/F16 weights and activations; output F32; rank two; no batches; default precision only | FP16 QNN MatMul; explicit float boundary conversion |
| ADD, MUL | F16 inputs and output; rank two; equal shapes or dimension-one broadcasts | FP16 QNN ElementWiseAdd/ElementWiseMultiply |
| DUP, CONT, CPY | F32/F16; equal element count; contiguous destination; dense or rank-two transposed source | Checked host materialization/cast |
| NONE, RESHAPE, VIEW, TRANSPOSE, PERMUTE | Bounded rank-two metadata alias over a readable admitted allocation | Metadata only |

SUB, DIV and FP32 elementwise are rejected: broader backend tests exposed rounding differences beyond their existing error limits on this HTP driver. Explicit full-precision MatMul requests are also rejected because FP16 QNN execution cannot satisfy them. No global test tolerance was weakened. Unsupported operations/types/layouts/buffers remain on native CPU through normal scheduler placement; an execution failure after placement is returned as failure.

GGML's MatMul constructor forbids a directly transposed left operand. Use CONT(TRANSPOSE(source)) to materialize the requested orientation. Noncontiguous numerical destinations, arbitrary gapped/overlapping strides, rank-three/four batches, tiled repeat broadcasts, quantized arithmetic and unimplemented operators are not admitted. Source buffer types must be CPU accessible with original GGML layout or QAIRT-owned; opaque buffers require ordinary GGML copies.

All buffers retain original GGML representations, including quantized bytes uploaded for byte-access tests. Retaining quantized bytes does not enable quantized operations or whole-model loading. QAIRT storage is aligned CPU-accessible memory and truthfully reports is_host=true. Pinned host allocations, host-pointer import, mmap support, async execution and events are not advertised.

## Preparation, reuse and failure handling

The M1 path compiles one QNN graph per numerical primitive and normalizes the complete scheduled partition before executing it. This is a bring-up path; it does not claim a throughput improvement over CPU. QNN dimensions reverse the admitted GGML rank-two dimensions: GGML weights [K,N] become QNN [N,K], activations [K,M] become [M,K], output [N,M] becomes [M,N]. MatMul uses transpose_in0=false and transpose_in1=true.

GGML WEIGHTS usage makes a QAIRT-owned MatMul source a QNN STATIC tensor. FP32 constants are converted once per preparation; FP16 constants are oriented/copied into owned staging. Other sources, including CPU-host weights whose writes cannot be tracked here, are rebound as APP_WRITE each execution. Static registration copies bytes according to the QNN contract.

Each stream caches at most 16 finalized primitive graphs. Exact keys include operation/type/shape/stride and static allocation identity/generation/offset. No reusable QNN ID or live caller tensor pointer is placed in a global weight cache. set/memset/copy/clear/reset advance generations. Fresh execution bindings always repack current dynamic inputs. Eviction releases graph/context ancestry through M0 RAII. Streams share a weakly retained runtime while alive; the last stream releases QNN resources before CRT shutdown. Discovery retains copied metadata/configuration rather than live QNN handles through process exit.

The adapter admits at most 128 nodes and uses a conservative 256 MiB partition staging budget. Oversized individual numerical/copy patterns are rejected by shared support checks. Complete-partition budget failure is explicit; the scheduler is not assumed to retry an already placed partition.

Outputs remain private until every node succeeds. Uncommitted predecessor results, including admitted aliases, feed subsequent nodes. Admission/preparation/conversion/binding/QNN failures clear preparation state and leave every destination unchanged. FP16 numerical inputs/outputs must be finite; FP32 values above 65504 cannot be converted to FP16. Copying between identical types preserves original bytes. Failed uncertain SDK frees retain owners through the existing M0 cleanup rule and are not retried.

A registry get_proc_address entry named ggml_backend_qairt_get_stats exposes per-stream preparation, hit and QNN execution counters. No full profiler or persistent-cache compatibility is claimed.

## Validation

Validation uses the native released SDK 2.52.0.261104, core API 2.41.0 and ARM64 MSVC 19.41.34123.0. Existing CPU-reference tests are used without changing their tolerances. Focused tests cover original-byte/batched access, float constants and boundaries, rectangular/transposed graphs, repeated values, static mutation/reset invalidation, failure atomicity and native CPU placement.

~~~powershell
& C:\Github\qairt-m1-arm64\bin\Release\test-backend-ops.exe -b QAIRT0 -o ADD,MUL,SUB,DIV -p 'type=f16.*ne=\[[0-9]+,[0-9]+,1,1\].*nf=1'
& C:\Github\qairt-m1-arm64\bin\Release\test-backend-ops.exe -b QAIRT0 -o MUL_MAT -p 'type_a=(f16|f32),type_b=f32,m=100,'
& C:\Github\qairt-m1-arm64\bin\Release\test-backend-ops.exe -b QAIRT0 -o CONT,CPY -p '(type_src|type)=f(16|32)'
~~~

The initial full elementwise probe exposed 13 failures in 32 cases. Tightened dtype/operator/size admission left 14 accepted cases, all passing. The rectangular MatMul sample passed 32/32; CONT/CPY passed 6/6. A filter matching zero cases was discarded as evidence. The support CSV in docs/ops/QAIRT.csv retains one actual probe per operation/support result and is a compact discovery snapshot, not exhaustive numerical validation.

The SDK's traditional transport fails on this machine and it selects the HTP user-driver path, reporting architecture 73. Unsupported-option and profiling-event diagnostics remain; successful small graphs do not prove full driver compatibility. Installed-package integration, full model inference, perplexity, sustained performance, persistent caching and GPU/CPU execution remain unvalidated.

Final November 15 results: complete ARM64 Release build passed; QAIRT plus tensor regression CTests passed 9/9. Module and fully static builds each passed 8/8 QAIRT CTests. QAIRT OFF built GGML/ARM CPU without an SDK. The focused HTP suite passed 338 checks, including FP16 activation boundaries; optional basic profiling also passed and reported six events. Four additional existing F16/F32-activation MatMul cases, including K=2051, passed. Source whitespace/ASCII/newline checks and git diff --check passed. M1 changes remain local and uncommitted.
