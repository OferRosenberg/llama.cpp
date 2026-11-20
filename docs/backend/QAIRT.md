# QAIRT backend: FP16 HTP implementation

## Accepted qualification for progression

On November 19, 2026, the user accepted the FP16-emulated short/long results as sufficient qualification to move to the next development steps. This acceptance applies to the current Qwen3 block-27 plus final-normalization HTP fixture.

All local projection NMSE, relative-error, final-logit and greedy-token checks pass. The 15 short and 14 long absolute intermediate-error failures remain documented exceptions: FFN down-projection output, attention residual output and final residual output. Original numerical thresholds, measured results and test exit behavior remain unchanged. This is approval to progress with known accuracy exceptions, not a claim of universal model or device qualification.

The optional ggml-qairt backend uses the released QNN C API on Windows ARM64. By default, numerical QNN weights, activations, intermediates and outputs use FP16. Opt-in `GGML_QAIRT_W4FP16=1` uses native block-quantized Q4_0 projection weights with FP16 activations and outputs; see [runtime W4FP16 weights](QAIRT-w4fp16.md). Current validation offloads all 27 numerical operations in Qwen3 block 27 plus final RMSNorm/scale. Builds and functional suites pass; original CPU model local/intermediate accuracy still fails unchanged acceptance gates.

## Build and runtime

```powershell
cmake -S C:\Github\llama.cpp -B C:\Github\qairt-qwen3-m2-arm64 -G "Visual Studio 17 2022" -A ARM64 -DGGML_QAIRT=ON -DGGML_QAIRT_TESTS=ON -DGGML_QAIRT_HW_TESTS=ON -DLLAMA_BUILD_TESTS=ON -DQAIRT_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.52.0.261104 -DGGML_OPENMP=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build C:\Github\qairt-qwen3-m2-arm64 --config Release --parallel 10
$env:GGML_QAIRT_DEVICE = 'HTP'
$env:GGML_QAIRT_LIBRARY_DIR = 'C:\Qualcomm\AIStack\QAIRT\2.52.0.261104\lib\aarch64-windows-msvc'
$env:ADSP_LIBRARY_PATH = 'C:\Qualcomm\AIStack\QAIRT\2.52.0.261104\lib\hexagon-v73\unsigned'
$env:PATH = $env:GGML_QAIRT_LIBRARY_DIR + ';' + $env:PATH
ctest --test-dir C:\Github\qairt-qwen3-m2-arm64 -C Release -R '^qairt-' --output-on-failure
```

GGML_QAIRT defaults OFF. Disabled builds do not require the SDK. SDK headers remain private; provider libraries load from the configured absolute directory. Backend registration follows ordinary GGML static/shared/module integration. GGML_BACKEND_DL=ON requires BUILD_SHARED_LIBS=ON. Installed static packaging is not separately qualified.

GGML_QAIRT_DEVICE accepts HTP, GPU or CPU. Only HTP execution is enabled; GPU/CPU adapters remain disabled. Selection, library directory and capture failures are immutable after first process capture. GGML_QAIRT_PROFILE=0|1 enables basic profiling; GGML_QAIRT_DIAGNOSTICS=0|1 enables diagnostic detail. GGML_QAIRT_W4FP16=0|1 enables upload-time weight repacking and native HTP W4FP16 MatMul; default 0, HTP V69+ required. Set before discovery/model loading. Invalid values fail configuration.

## Admitted operation families

Support is bounded by shape, representation, precision, buffer and staging checks. The table identifies recipes, not universal support for every GGML variant.

| GGML operation | QNN HTP recipe |
|---|---|
| MUL_MAT, F32/F16 or admitted Q4_0 source weights | FP16 MatMul by default; opt-in W4FP16 supplies initialized STATIC signed-byte block weights for eligible Q4_0 projections |
| Grouped/batched floating MUL_MAT | In-graph StridedSlice / rank-two MatMul per head / Concat with explicit KV-group mapping |
| ADD / MUL | FP16 ElementWiseAdd / ElementWiseMultiply; admitted F32/F16 boundaries and singleton broadcasting |
| RMS_NORM / NORM | RMSNorm / LayerNorm over final feature axis; flatten/factor independent rows into rank four |
| SILU | Sigmoid(x), Multiply(x,sigmoid) |
| Split SWIGLU | Sigmoid(gate), Multiply(gate,sigmoid), Multiply(result,up) |
| ROPE | Genie-style host table selection; QNN Transpose / RotaryEmbedding / Transpose |
| SOFT_MAX | Host finite-mask preparation; in-graph scale Multiply / mask Add / ordinary Softmax |
| FLASH_ATTN_EXT | Shared FP16 QK / scale / mask / Softmax / V decomposition |
| SET_ROWS | FP16 ScatterNd with checked INT32 indices; row or scalar transposed-cache layouts |
| GET_ROWS | FP16 Gather, axis zero, dynamic INT32 indices |
| VIEW / RESHAPE / TRANSPOSE / PERMUTE | GGML metadata aliases and stride handling |
| CONT / DUP / CPY | In-graph Reshape / StridedSlice / Transpose; preserve GGML destination representation |

Projection shapes: source weights ne=[K,N], activation ne=[K,T], destination ne=[N,T]; QNN weights [N,K], input [T,K], output [T,N]. MatMul uses transpose_in0=false and transpose_in1=true. Q4_0 K must be divisible by 32; source rows must satisfy the admitted layout.

Normalization maps feature D to axis 3 in [1,H,R/H,D]. RoPE receives [B,H,T,D], with rank-three cosine/sine [B,T,rotary_dim/2]; QNN Transposes adapt dense [B,T,H,D]. Normal/interleaved and NEOX modes are admitted with checked parameters. Unsupported RoPE variants/offsets remain rejected.

F32 numerical boundaries must be finite and representable in FP16. FP16 outputs widen to F32 when required. Gather/ScatterNd INT32 tensors are control metadata; INT64 scatter indices are checked and narrowed. There is no native QNN BF16 numerical tensor path. Floating MatMul accepts default/F32/BF16 accumulation hints with FP16 operands; explicit source precision overrides and nondefault accumulation for Q4_0 projections remain rejected. QNN tensor dtype does not guarantee a particular internal accumulator implementation.

Quantized fused MaskSoftmax is not assumed. Mask -infinity becomes -65504, with unmasked-finite-row validation before execution. All-masked rows, unsupported broadcasting/variants and other source-weight formats remain rejected. SUB/DIV are not advertised as GGML operations merely because private QNN packet validation recognizes them.

## Buffers, preparation and state

Original GGML buffers preserve their source representations, including Q4_0 bytes. QAIRT storage is aligned host-accessible memory. It reports is_host=true by default; W4FP16 mode reports is_host=false to require upload callbacks during model loading, with ordinary GGML transfers for CPU consumers. Validated host buffer types, including CPU buffers, are accepted directly by the scheduler. Opaque source buffers require ordinary GGML copies. Pinned allocation, host-pointer import, mmap, registered/shared memory, async execution and events are not advertised.

Default Q4_0 rows decode through GGML type traits with bounded F32 scratch and FP16 staging. W4FP16 mode repacks eligible weights into immutable buffer-owned signed-byte/block-metadata sidecars during upload; graph-local metadata copies prevent SDK registration from mutating shared state. Graph construction reuses sidecars and keeps FP16 fallback for other sources. Dynamic numerical inputs convert directly to FP16 for every execution. Tracked QAIRT WEIGHTS allocations can supply STATIC QNN data. Allocation identity/generation/offset enter preparation identity; mutations force reprepare. Untracked sources are rebound as APP_WRITE. STATIC registration uses the released API's copied-payload contract.

prepared_run owns finalized graph/context references, boundary input IDs, output IDs and optional RoPE tables. Structural keys include types, shapes, strides, parameters, topology, FP16 recipe and tracked source identity. Activations, indices and masks refresh on cache hits. A shared partition_builder lowers connected operators, branches, compatible metadata views and transactional KV writes into one graph. Grouped attention uses in-graph head slices, MatMuls and Concat. Every GGML numerical output remains readable; temporary decomposition tensors are NATIVE. graph_optimize retains readable node buffers through partition completion using add_alloc_dep.

The LRU retains at most 512 profiles and a 1 GiB logical static-weight/table budget. Partition staging is 256 MiB; RoPE table caching is limited to 64 MiB. These bounds exclude opaque SDK/driver overhead. Persistent binary caches remain deferred. The fused Qwen3 fixture executes block 27 and final normalization as one graph per shape profile.

llama.cpp owns KV allocation, capacity, sequence indexing and reset. HTP ScatterNd produces staged F16 K/V writes with range/duplicate checks and untouched-row validation. Outputs remain private until the whole QAIRT partition succeeds. Staged predecessors feed subsequent recipes. Any conversion/preparation/binding/execution failure invalidates preparation and preserves destinations. SDK frees with uncertain handle state retain owners under the M0 cleanup rule.

Production reuses owning runtime resources. Correctness fixtures release buffer, graph/context, device/backend/logger and provider ownership between fixtures, avoiding the earlier cross-case RoPE preparation failure.

## Real-model placement and accuracy

Target: C:\Models\Qwen3-1.7B-Q4_0.gguf. Blocks 0-26 remain CPU. The expanded fixture offloads seven block-27 projections, four RMSNorms/scales, Q/K RoPE, two attention MatMuls, masked Softmax, SwiGLU, both residuals, both KV writes and both Gathers. Final RMSNorm/scale also run on HTP: 29 numerical GGML nodes per frame.

The tied Q6_K embedding/output remains CPU. Host duties include source decoding, direct boundary conversion, external boundary packing, position-table selection and mask construction/encoding. Metadata aliases and CONT helpers are excluded from numerical HTP counts. The one-block fixture uses n_gpu_layers=2 because llama.cpp counts the output-device slot.

| Profile | Worst logit maxabs | Worst KL | Worst local NMSE | Failing assertions |
|---|---:|---:|---:|---:|
| Fused short | 0.680374622 | 2.80317716e-7 | 0.00015322445 | 167 |
| Fused long | 0.715574265 | 6.53793968e-8 | 0.000220370055 | 173 |
| Projection-only short control | 0.666971207 | 1.24152408e-7 | 0.000151564319 | 161 |

Short means 28-token prefill; long means 128-token prefill plus 13-token tail. Both include eight teacher-forced decode steps and deterministic reset. All expanded logit maxabs <= 0.75 / KL <= 0.001 checks pass with zero greedy disagreements. Local NMSE <= 1e-5 and applicable intermediate maxabs <= 0.75 / relative RMSE <= 0.02 still fail. These baseline measurements fail the original accuracy gates; limits are unchanged.

## Verification and commands

November 19, 2026: full configured ARM64 Release build passes; all 13 QAIRT CTests and all 22 isolated numerical comparisons pass. Worst isolated NMSE is 2.31278568e-6. CPU short/long controls match exactly with zero failures. Standard support probing covers 23,594 cases; [QAIRT.csv](../ops/QAIRT.csv) retains 135 representative operation/support outcomes and does not establish numerical/performance qualification.

```powershell
$bin = 'C:\Github\qairt-qwen3-m2-arm64\bin\Release'
$model = 'C:\Models\Qwen3-1.7B-Q4_0.gguf'
& "$bin\test-qairt-ops.exe"
& "$bin\test-qairt-model.exe" $model --cpu-control
& "$bin\test-qairt-model.exe" $model --cpu-control --prefill 128 --tail 13
& "$bin\test-qairt-model.exe" $model --ops all
& "$bin\test-qairt-model.exe" $model --ops all --prefill 128 --tail 13
& "$bin\test-qairt-model.exe" $model --ops projections
```

--ops all is the expanded default. --ops projections restricts numerical HTP offload to seven projections. Functional suites/CPU controls require exit zero. Model accuracy commands currently return one for retained failures. This implementation is preserved in the local M2 Full FP16 checkpoint.


Standard GGML CPU-reference smoke checks also pass for nine representative Add, Multiply, LayerNorm, RMSNorm, SiLU, split SwiGLU, Gather, Softmax and normal-mode RoPE cases. The generic QAIRT NMSE hook applies the previously agreed FP16 budget as max(existing operation tolerance, 1e-5). Before that hook was restored, normal-mode RoPE measured NMSE approximately 1.26e-7 against the generic 1e-7 default. No model accuracy gate changed.

## Connected graph validation

Use test-qairt-model MODEL --fused for the consolidated profile, or --fused --prefill 128 --tail 13 for long validation. The ordinary diagnostic callback can split graph_compute calls at checkpoints. The fused observer reads retained node buffers after execution; it asserts one QNN execution per frame. Cold prefill and first decode each prepare one graph; subsequent decode/reset frames reuse the matching profile. Multi-token graphs contain 193 QNN nodes and decode graphs 191.

All numerical QNN edges remain FP16 inside the graph. Application-readable GGML outputs widen only after graph execution and validation. Numerical intermediate exports remain for the GGML buffer contract; eliminating them requires explicit external-consumer information. Native GroupQueryAttention is not selected; admitted FLASH_ATTN_EXT uses the shared FP16 decomposition.

## FP16-emulated model reference

Add `--fp16-reference` to `test-qairt-model MODEL --fused` for a precision-emulated CPU reference in block 27 and final RMSNorm/scale. Earlier CPU blocks and the output head retain their existing behavior. The test decodes Q4_0 weights to FP16-rounded floating values, rounds numerical inputs/outputs at operator boundaries, and executes F32 CPU arithmetic internally. Masked Softmax and SwiGLU also round their decomposition intermediates. CPU RoPE retains internal F32 trigonometric evaluation; this is not a reproduction of native QNN internal arithmetic.

The short/long FP16 reference controls match exactly. All local projection NMSE checks pass (worst 1.03446516e-9 / 6.75396652e-10); logit maxabs is 0.160899878 / 0.292719841. Fifteen/fourteen intermediate comparisons still exceed the unchanged maxabs limit of 0.75, while relative RMSE checks pass. Progression is accepted with these documented accuracy exceptions. Without the flag, the original CPU reference and results remain available. `test-qairt-model --fp16-reference-self-test` exercises the emulation without loading a model.

## M2 Full FP16 all-block checkpoint

The November 19, 2026 all-block fixture covers all 28 Qwen3 blocks (0-27), using per-block scheduler cuts to respect the 128-GGML-node partition limit. This does not qualify unrestricted full-model CLI offload. Numerical QNN weights, activations and intermediates remain FP16; positions and indices remain integer metadata.

With `--fused --fp16-reference --offload-last all --decode-steps 20 --prompt 'Explain how rain forms. Write three complete sentences.'`, hybrid and FP16-emulated CPU execution produce identical 20-token greedy output across 22 frames. Reset is exactly reproducible. All 4,246 local projection comparisons pass; worst local NMSE is 8.82395592e-9 against 1e-5. Worst logit maxabs is 0.445525169 against 0.75.

Known numerical and fixture gaps remain: decode13/decode17 KL is 0.00659981862/0.00163475668 against 0.001; worst intermediate maxabs is 16 against 0.75, and worst relative RMSE is 0.108090024 against 0.02. The executable returns one with 1,948 assertions: 1,880 checkpoint accuracy assertions, 66 unmatched-reference-key assertions (three per frame), and two logit assertions. Matching generated text does not establish that these numerical gates pass.

Each frame executes 193 Q4_0 projections and 508 other numerical operations on HTP across 31 QNN executions, with 29 preparations and two cache hits. Q4_1 FFN-down projections in blocks 0-2 and Q6_K embedding/output remain on CPU. The 1 GiB static cache budget causes repeated graph preparation. ARM64 Release builds and all 14 QAIRT CTests passed after this model run.
