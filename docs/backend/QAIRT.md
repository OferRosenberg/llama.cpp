# QAIRT backend (M2 hybrid bring-up)

## Default FP16 bring-up path (November 17, 2026)

HTP graphs receive FP16 weights and FP16 activations. Q4_K and Q6_K weights are decoded with GGML helpers and converted to FP16 before QNN registration. F32 and BF16 weights and activations convert directly to FP16; F16 inputs retain their values. Original GGML storage remains unchanged. QNN static projection weights remain row-major [N,K], activations [M,K], outputs [M,N], with the existing MatMul transpose policy and prepared-graph cache.

`GGML_QAIRT_DIRECT_FP16` now defaults to `1`. Direct conversion bypasses host Q8_K quantize/reconstruct and F32-to-BF16 compatibility rounding. Explicit `GGML_QAIRT_DIRECT_FP16=0` retains the earlier CPU-compatible preparation for diagnostic comparisons. Earlier CPU-reference qualification results below refer to that compatibility mode; they do not establish full-model accuracy for the new default.

The ARM64 backend, test executables and llama-cli build successfully. The focused default-path fixture passes 326 checks at the existing 1e-5 NMSE budget, covering all five weight formats, static/dynamic weights, repeated execution, unchanged original weight bytes, FP16 underflow and nonfinite-input rejection with unchanged outputs. Full-model accuracy experiments remain stopped.

Current accuracy policy: QAIRT HTP FP16 NMSE <= 1e-5. Earlier 1e-7 comparison results are historical; see [Initial FP16 accuracy budget](#initial-fp16-accuracy-budget-november-15-2026).

The optional ggml-qairt backend uses the released QNN C API on Windows ARM64. It loads ordinary GGUF models through llama.cpp and executes admitted dense projections, normalization and SiLU graphs on HTP. M2 currently qualifies a limited CPU/HTP profile; broader transformer offload remains open.

## Build and configuration

```powershell
cmake -S C:\Github\llama.cpp -B C:\Github\qairt-m2-arm64 -G "Visual Studio 17 2022" -A ARM64 -DGGML_QAIRT=ON -DGGML_QAIRT_TESTS=ON -DGGML_QAIRT_HW_TESTS=ON -DLLAMA_BUILD_TESTS=ON -DQAIRT_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.52.0.261104 -DGGML_OPENMP=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build C:\Github\qairt-m2-arm64 --config Release --parallel 8
$env:GGML_QAIRT_DEVICE = 'HTP'
$env:GGML_QAIRT_LIBRARY_DIR = 'C:\Qualcomm\AIStack\QAIRT\2.52.0.261104\lib\aarch64-windows-msvc'
# Older CPU-reference suites require their original compatibility preparation.
$env:GGML_QAIRT_DIRECT_FP16 = '0'
ctest --test-dir C:\Github\qairt-m2-arm64 -C Release -R 'qairt-|test-backend-tensor' --output-on-failure
# default-fp16 and direct-fp16 cases explicitly select the direct path.
$env:GGML_QAIRT_DIRECT_FP16 = '1'
```

GGML_QAIRT defaults OFF; an OFF build needs no SDK. SDK headers stay private and provider DLLs are loaded at runtime. The internal core is static; registration uses the existing GGML static/shared/module helpers. GGML_BACKEND_DL=ON requires BUILD_SHARED_LIBS=ON. Installed static-library packaging remains unvalidated.

GGML_QAIRT_DEVICE accepts strict uppercase HTP, GPU or CPU. HTP is default and the only enabled adapter. GPU/CPU report a disabled adapter and expose no usable QAIRT device. Configuration, including failure, is captured once per process. GGML_QAIRT_LIBRARY_DIR must be an absolute native directory. Optional GGML_QAIRT_PROFILE=0|1 reports basic QNN profile event counts; GGML_QAIRT_DIAGNOSTICS=0|1 enables detailed diagnostics.

QAIRT0 appears after a successful runtime probe and cleanup. Its description reports the physical HTP architecture. Its GGML GPU device class follows OpenVINO's selected-target offload convention; memory capacity is reported as unknown. Future QNN CPU execution remains distinct from native GGML CPU.

## Native normalization and experimental Softmax

GGML RMS_NORM lowers to QNN RmsNorm: y = x / sqrt(mean(x*x) + epsilon). GGML NORM lowers to QNN LayerNorm: y = (x - mean(x)) / sqrt(mean((x - mean(x))^2) + epsilon). Both reduce only GGML ne[0]. QNN InstanceNorm and GroupNorm are not selected because their grouping and spatial reductions do not match this independent-row contract.

Inputs and outputs are contiguous F16/F32 GGML tensors with matching shape/type, up to four dimensions. QNN tensors and HTP arithmetic remain FP16; F32 is a converted application boundary. Epsilon is a positive finite FLOAT32 scalar parameter. Extra sources and strided row layouts are rejected.

Flatten the outer GGML dimensions into R independent rows and retain C = ne[0]. QNN descriptors use [1,H,R/H,C], axis 3. For R <= 4096, H = 1. Larger batches use the largest divisor of R no greater than 256. Admission requires R/H <= 4096 and the existing staging limit. This preserves contiguous bytes without a transpose or extra QNN Reshape node; output conversion restores the original GGML shape. For example, [4,1,65536,1] becomes [1,256,256,4]. Large prime row counts outside this profile are rejected for CPU scheduling.

Normalization registers explicit STATIC FP16 gamma=1 and beta=0 vectors of length C. Although master definitions permit omission, this HTP runtime failed finalization without them. An owned STATIC UINT32 axes tensor [3] is registered through tensorCreateGraphTensor before graphAddNode; an unregistered parameter tensor with ID zero is insufficient. Affine vectors and axes count toward the prepared-cache static payload budget. Epsilon and all original dimensions/strides participate in the cache key.

GGML SOFT_MAX uses regular FP16 QNN Softmax with UINT32 axis=3 and FLOAT32 beta=1. With an additive mask, the QNN graph first runs ElementWiseAdd into a native FP16 tensor, then Softmax. The masked recipe preserves [batch, heads, queries, keys] and supports equal or dimension-one head/batch broadcasting. Mask queries can be padded; unused rows are omitted from the QNN binding. Host preparation applies GGML scale before the mask and centres logits using the valid combined row maximum. Exact negative-infinity mask entries become -65504 with a zero logit at the excluded position. Finite mask values and centred logits must fit FP16. NaN, positive infinity, entirely excluded rows, periodic mask repetition, ALiBi bias, attention sinks, and nonpositive/nonfinite scale are rejected. Unmasked centering remains unchanged. The recipe uses regular operations and does not depend on quantized MaskSoftmax fusion or the residual Add experimental gate.

Softmax remains disabled by default pending broader model/profile qualification; QAIRT validation uses the approved 1e-5 FP16 NMSE budget. Set GGML_QAIRT_EXPERIMENTAL_SOFTMAX=1 before backend registration to test the recipe. This strict 0|1 option is captured once with other runtime configuration. Both device capability queries and direct graph computation enforce the gate. It does not enable FP32 HTP arithmetic or relax global test tolerances.

SDK references: local docs/QAIRT-Docs/QNN/OpDef/MasterOpDef.html and HtpOpDefSupplement.html, include/QNN/QnnOpDef.h, and the QAISW QNN converter translations. HTP normalization supports rank <= 4 and final-channel reduction; Softmax supports rank <= 5 and the final axis. The integration uses rank four for both.
## Supported operations and representations

| Operation | Contract | Execution |
| --- | --- | --- |
| MUL_MAT | Contiguous rank-two F16/F32/Q4_K/Q6_K weights; rank-two dense/transposed F16/F32 activations; default accumulator/source precision | FP16 QNN MatMul; quantized weights require F32 activations |
| MUL | Rank-two F16 operands/output; equal shapes or dimension-one broadcasts | FP16 QNN ElementWiseMultiply |
| RMS_NORM, NORM | Contiguous F16/F32, rank <= 4; positive epsilon; bounded row tiling | FP16 QNN RmsNorm / LayerNorm |
| UNARY(SILU) | Contiguous F16/F32, rank <= 4; matching shapes/types; bounded staging | FP16 QNN Sigmoid then ElementWiseMultiply |
| ADD (experimental) | Contiguous F16/F32, rank <= 4; equal or dimension-one skip broadcasts | FP16 QNN ElementWiseAdd; disabled by default |
| ROPE (experimental) | Contiguous F16/F32, NORMAL/NEOX, I32 positions and optional F32 factors; bounded cache/staging | Selected host sine/cosine rows and FP16 QNN RotaryEmbedding; disabled by default |
| SOFT_MAX (experimental) | Matching contiguous F16/F32; optional contiguous F16/F32 additive mask, padded queries and ordinary head/batch broadcasts; positive scale, no ALiBi or sinks | FP16 QNN ElementWiseAdd then Softmax for masks; disabled by default |
| DUP, CONT, CPY | F16/F32; equal element count; contiguous rank-two destination; dense or rank-two transposed source | Checked host materialization/cast |
| NONE | Admitted float metadata, bounded I32 position vectors or rank-two weights, including Q4_K/Q6_K | No numerical work |
| RESHAPE, VIEW, TRANSPOSE, PERMUTE | Bounded F16/F32 metadata, up to four dimensions; checked span/source window | Alias only; batched MatMul remains unsupported |

Explicit accumulator or source precision hints are rejected, including F32/BF16 requirements. This initial implementation supports default precision only. Directly transposed left MatMul operands are prohibited by GGML; materialize CONT(TRANSPOSE(source)) first. Quantized transpose views, arbitrary gapped numerical strides, rank-three/four MatMul/elementwise batches, repeat tiling and unlisted types/operators remain unsupported.

SUB, DIV and unlisted elementwise recipes remain rejected. ADD has a native recipe behind an experimental gate; default admission remains disabled. Repeated strict ADD tests exposed HTP FP16 rounding errors above the existing 1e-7 error bound, including errors of 1.08e-7 and 1.75e-7. M2 withdraws M1's ADD admission. Its regression chains now use MUL, preserving output atomicity and cache checks. Global backend test tolerances are unchanged.

All allocations retain original GGML bytes and strides. QAIRT buffers are aligned CPU-accessible storage with is_host=true. supports_buft accepts QAIRT's own storage only, so CPU-resident weights remain on CPU when op_offload=false. Operation admission can read borrowed host tensors; this does not grant ownership of their buffer type. Opaque device storage needs ordinary GGML copies. mmap/import, pinned memory, async execution and events are not advertised.

Q4_K/Q6_K use GGML's existing reference decoders, then checked FP16 conversion. They are not QNN low-bit tensor encodings. With explicit GGML_QAIRT_DIRECT_FP16=0, their F32 activation boundary reproduces GGML CPU's Q8_K activation preparation before reconstructing FP16. The default converts activations directly to FP16. Zero blocks are accepted; overflowing Q8_K scales and nonfinite reconstructed values fail without publishing outputs.

## Graph preparation, reuse and failure handling

GGML weights [K,N], activations [K,M] and results [N,M] map to row-major QNN [N,K], [M,K] and [M,N]. MatMul sets transpose_in0=false and transpose_in1=true. Compatible consecutive numerical operations form QNN graphs of up to 16 nodes. All numerical results are APP_READ tensors, including results consumed by later nodes. Metadata/materialization and alias-sensitive dependencies split runs. Quantized MatMul also splits at an internal F32 activation predecessor to preserve the Q8_K boundary recipe.

QAIRT-owned WEIGHTS sources become QNN STATIC tensors. Borrowed CPU-host weights are dynamic inputs because their writes are not tracked here. Each backend stream retains at most 512 finalized runs and 1 GiB of logical static FP16 bytes with LRU eviction. This count does not measure SDK/driver memory, compiled graph code or conversion workspace. Different shapes can duplicate static weights and evict profiles.

Exact cache keys include numerical order, dtype/dimensions/strides, topology, boundary mapping and Q8_K preparation recipe. Static keys also include allocation identity, write generation and offset. Only default precision is admitted, so no varying precision parameter is omitted. Dynamic addresses and values are rebound each invocation; cached entries retain indices and QNN IDs rather than caller tensor pointers. Buffer writes/reset advance generations.

The scheduled partition is capped at 128 nodes with a conservative 256 MiB staging bound. Admission accounts for pending outputs and peak conversion; combined-run planning accounts for simultaneous I/O buffers. Oversized individual operators remain CPU candidates. Aggregate execution-budget failure is returned explicitly; no scheduler retry is assumed.

Each execution packs fresh inputs, executes QNN, validates all numerical outputs and commits every scheduled destination only after complete success. Pending results feed later runs and admitted aliases. Failure clears prepared state and leaves all destinations unchanged. Numerical FP16 values must be finite; F32 values outside FP16 range are rejected. Same-type host copies preserve bytes. Uncertain SDK cleanup failures retain owners under the existing M0 rule and are never freed twice.

Registry get_proc_address provides ggml_backend_qairt_get_stats for per-stream counters and ggml_backend_qairt_get_device_stats for aggregate preparation/hit/execution counters. Aggregate access is mutex-protected; individual backend streams require normal GGML sequencing. These counters establish real QNN execution without exposing llama internals.

## Model validation profile and limits

```powershell
& C:\Github\qairt-m2-arm64\bin\Release\test-qairt-model.exe C:\Models\Llama-3.2-1B-Instruct-Q4_K_M.gguf
& C:\Github\qairt-m2-arm64\bin\Release\test-qairt-model.exe C:\Models\Llama-3.2-1B-Instruct-Q4_K_M.gguf --prefill
& C:\Github\qairt-m2-arm64\bin\Release\test-qairt-model.exe C:\Models\Llama-3.2-1B-Instruct-Q4_K_M.gguf --capacity
& C:\Github\qairt-m2-arm64\bin\Release\test-qairt-model.exe C:\Models\Llama-3.2-1B-Instruct-Q4_K_M.gguf --prompt 'Two plus two equals'
```

These local model tests are opt-in and never download a fixture. The loader reports 34 F32, 96 Q4_K and 17 Q6_K tensors; the model has 16 layers, embedding 2048 and vocabulary 128256. The passing profile uses n_gpu_layers=2: one repeating layer is placed on QAIRT, while the intended output projection remains CPU because it exceeds capabilities. The original projection-only profile executed seven projection graphs per ordinary decode. Native normalization admission now allows additional QNN row graphs; the updated validation results below supersede projection-only counts. Default residual Add, embeddings, oversized vocabulary projection, RoPE, BF16 attention accumulation, default Softmax and external KV writes retain CPU boundaries.

Context is 1024, one sequence, F16 K/V, flash attention disabled, offload_kqv=false and op_offload=false. Decode uses microbatch1. Prefill uses microbatch128 plus a 13-token tail; it specializes actual shapes. CPU owns KV append, full-capacity rejection and reset. Context shifting, multiple sequences and arbitrary model architectures are unqualified.

| Local test, November 15, 2026 | Observed result |
| --- | --- |
| Default teacher-forced decode | Zero failures; max logit KL 0.00043886; greedy agreement 7/8; 7 preparations, 98 hits, 105 executions |
| 128-token prefill, 13-token tail, decode transition | Zero failures; max logit KL 0.00005230; greedy agreement 8/8; 15 preparations, 62 hits, 77 executions |
| Fill context 1024, reject append, reset | Zero failures; final prompt KL 0.00007261; 11 preparations, 52 hits, 63 executions |
| Story prompt | Zero failures; greedy agreement 8/8; 7 preparations, 91 hits, 98 executions |
| Arithmetic prompt | Zero failures; greedy agreement 8/8; max decode KL 0.00065021; 7 preparations, 91 hits, 98 executions |
| Existing Q4_K/Q6_K operator comparisons | 4/4 accepted cases passed unchanged CPU-reference thresholds |
| Existing F16 elementwise comparisons after ADD withdrawal | 8/8 accepted MUL cases passed |
| Existing CONT/CPY comparisons | 7/7 accepted cases passed |

The new model fixture uses max absolute logit error 0.75 and KL(cpu||HTP)<=0.001, intermediate relative RMSE<= 0.02 and at most one greedy disagreement in eight. It also reports raw logit relative RMSE and the original 0.02 target; passing profiles meet that target. KL was added transparently during bring-up, not applied to relax existing operator tests. Prefill/capacity prompts repeat synthetic seed tokens; short continuation checks are teacher-forced rather than a free-running quality or perplexity suite.

Wider offload remains unqualified: n_gpu_layers=4 reached KL 0.00259452 and n_gpu_layers=8 reached KL 0.00374072. Earlier all-layer execution failed near layer 13 after 80 preparations/79 executions, with transport graph-creation error6020 followed by execute6001. The exact resource limit is unknown; the subsequent 1 GiB cache reduction does not establish a fix.

The SDK selects its HTP user-driver path after traditional transport fails and reports architecture 73. Unsupported-option/profiling warnings remain. Successful hybrid execution does not prove full driver compatibility or a performance gain. Softmax accuracy, residual Add accuracy, RoPE accuracy and HTP attention recipes, broad model quality, all-layer resource stability, persistent contexts/caches, installed packages and GPU/CPU execution remain open.

## Projection-only build verification (historical)

On November 15, 2026, the full native ARM64 Release build completed successfully, including CLI, server, examples and tests. QAIRT/tensor CTests passed 11/11; the focused hardware executable passed 543 checks with zero failures. Module/shared and fully static QAIRT configurations each built and passed 10/10 CTests. The SDK-disabled GGML/CPU configuration built successfully. The final support snapshot contains 127 representative rows from 23,594 live probes. Build output retains existing MSVC duplicate-object and missing embedded-UI warnings; no QAIRT compilation error remains.

## Normalization validation before SiLU/residual changes (November 15, 2026)

These results supersede the projection-only baseline above. M2 remains incomplete because full transformer/attention offload and broad quality/resource qualification are still open.

| Check | Current result |
| --- | --- |
| Full native ARM64 Release build | Passed, including CLI/server/examples/tests |
| QAIRT and backend-tensor CTests | 12/12 in native, module/shared and fully static configurations |
| SDK-disabled GGML/CPU build | Passed |
| Default hardware executable | 1,337 checks, zero failures |
| Experimental row executable, including large in-place LayerNorm | 1,570 checks, zero failures |
| M2 host admission tests | 42 checks, zero failures |
| Existing strict contiguous F32 RMS_NORM / NORM comparisons | 52/52 passed, including 65,536 rows |
| Existing strict unmasked/sink-free F32 Softmax comparisons, experimental setting | Latest run 11/23 passed; not qualified for default admission |
| Default capability snapshot | 129 representative rows from 23,594 live probes; experimental Softmax disabled |

The local row fixture uses an explicit FP16 boundary error budget of 0.004; its passing result does not substitute for the existing strict operator tests. Those global thresholds are unchanged. Strict Softmax failures include FP16 numerical error and large-channel cases; shape remapping alone does not resolve accuracy.

The supplied Q4_K_M model passed the current limited n_gpu_layers=2 profile with default Softmax disabled:

| Fixture | Current evidence |
| --- | --- |
| Decode/reset | Maximum decode KL 0.00091876; greedy agreement 7/8; reset KL 0.00001993; 8 preparations / 142 hits / 150 QNN executions |
| Prefill128 plus tail13/reset | Maximum decode KL 0.00010859; greedy agreement 8/8; 17 preparations / 92 hits / 109 executions |
| Capacity1024/reset | Final prompt KL 0.00011444; reset KL 0.00001993; 12 preparations / 70 hits / 82 executions |

These are bounded teacher-forced fixtures, not full-model quality or performance qualification. Model execution counts are aggregate and do not identify each operator individually. Existing transport/profile and MSVC duplicate-object/UI warnings remain. All M2 changes are local and uncommitted.

## SiLU decomposition and residual Add

GGML represents SiLU as GGML_OP_UNARY with GGML_UNARY_OP_SILU. The backend admits matching contiguous F16/F32 input/output tensors up to rank four, within the existing bounded staging/shape profile. F32 is an application representation; QNN tensors and HTP arithmetic remain FP16.

Each SiLU compiles into two QNN nodes in one finalized graph: s = Sigmoid(x), followed by y = ElementWiseMultiply(x, s). op_spec::sigmoid creates the parameter-free unary packet. The sigmoid result is a backend-private NATIVE tensor with the same dimensions/type as x; only y is APP_READ. No CPU Sigmoid or SiLU arithmetic is used. QNN may optimize these nodes during finalization.

```mermaid
flowchart LR
    X[APP_WRITE x: FP16] --> S[QNN Sigmoid]
    S --> T[NATIVE sigmoid result: FP16]
    X --> M[QNN ElementWiseMultiply]
    T --> M
    M --> Y[APP_READ y: FP16]
```

SiLU currently starts a separate prepared run, preserving the existing rank-two projection/MUL planner. The cache key includes the GGML unary opcode, tensor types, shapes and strides. Fresh execution bindings and deferred output commits support changed inputs and in-place GGML SiLU. The private intermediate is not exported or allocated as a GGML tensor.

Residual Add compiles to one QNN ElementWiseAdd node: y = main_branch + skip. Matching contiguous F16/F32 representations are required. For Add, QNN dimensions are [ne[3],ne[2],ne[1],ne[0]] for every operand/output; outer dimensions are not flattened. Each skip dimension must equal the output dimension or be one. Arbitrary repeat tiling, mismatched types, extra sources and gapped strides are rejected.

For example, main GGML [32,3,2,2] and skip [32,1,1,2] map to QNN [2,2,3,32] and [2,1,1,32]. This retains each original broadcasting axis. Add starts a separate run and follows the same cache, finite-value validation and transactional output rules. A SiLU-to-Add chain consumes staged predecessors; failure in Add leaves both GGML outputs unchanged.

Residual Add is implemented but remains disabled by default. The unchanged strict ADD suite passed 51/57 admitted cases, with FP16 and F32 boundary rounding failures above 1e-7. Examples include FP16 error 2.88e-7 and chained F32 error 5.01e-7. Set GGML_QAIRT_EXPERIMENTAL_ADD=1 before backend registration to test it. The captured 0|1 setting gates both device admission and direct computation. No global tolerance was relaxed and no native FP32 HTP arithmetic was introduced.

Native validation: SiLU passed 2,353 hardware checks and all 4 admitted existing strict operator comparisons. The residual broadcasting/SiLU-chain fixture passed 2,339 checks, including failure after a staged predecessor. QAIRT/tensor CTests passed 14/14 in the native configuration; configuration and model results are recorded separately below.

SDK references are the local MasterOpDef/HtpOpDefSupplement Sigmoid, ElementWiseMultiply and ElementWiseAdd sections. Sigmoid is parameter-free and preserves shape/type. The FP16 Sigmoid input rank limit is four; the integration stays within that bound. Elementwise Add/Multiply support the required FP16 configurations and bounded broadcasting shapes.

### Final SiLU/residual verification

On November 15, 2026, the full native ARM64 Release build and module/shared/static QAIRT test-target builds completed successfully. QAIRT/tensor CTests passed 14/14 in each enabled configuration. The SDK-disabled GGML/CPU build also passed. Default hardware tests passed 3,701 checks; host admission tests passed 58 checks. Native normalization retained 52/52 strict comparisons, and native SiLU passed 4/4 strict comparisons. Experimental residual Add passed 51/57 strict comparisons; the 2,339-check exact-broadcast/SiLU-chain fixture does not override those failures.

The supplied model's default decode/reset fixture passed with experimental Add and Softmax disabled: maximum decode KL 0.00091876, greedy agreement 7/8, reset KL 0.00001993, and 8 preparations / 142 hits / 150 QNN executions. These aggregate counts match the prior normalization profile; this model run does not prove SiLU offload. Focused hardware tests separately establish execution of the SiLU decomposition.

The conservative default support snapshot contains 130 representative rows from 23,594 live probes, including four supported SiLU cases and no default ADD admission. Source/newline checks covered 20 changed/new repository files; git diff --check passed. Existing SDK transport/profile and build warnings remain. Changes remain local and uncommitted; M2 attention/RoPE accuracy and wider quality/resource work remain outstanding.

## Genie-style cached RoPE (experimental)

GGML ROPE now uses Genie's preparation and inference split: compute reusable sine/cosine tables on the host, then select rows using the current GGML position IDs. The selected rows are QNN application inputs. Rotation executes as QNN RotaryEmbedding on HTP; there is no host rotation or QNN Gather in this path.

The local Genie reference is genie/src/core/engines/qnn-htp/nsp_model.cpp under C:/QAIRT code copy/QAISW/FirstParty/Genie-standalone. calculate_rope_embeddings() at line 5510 allocates context-sized tables and finishes by setting m_ropeInitialized. The inference loop at lines 8478-8486 copies rows indexed by posIds into sine/cosine graph input buffers. This backend follows that host selection mechanism, while deriving frequencies from GGML parameters rather than Genie's model configuration.

### Admission, types and layouts

- Contiguous matching F16 or F32 activation/output tensors, up to four dimensions, with an even head size no greater than 4096.
- GGML NORMAL and NEOX modes, even rotary dimension from 2 through the head size, full or partial rotation beginning at offset zero.
- Contiguous I32 position vector of length ne[2], at most 4096 tokens. The same token-position vector applies to each GGML batch, matching the CPU reference. IDs must be nonnegative and below 2^20; the table byte limit can impose a smaller bound.
- Optional contiguous F32 frequency factors, with at least rotary_dimension/2 entries and at most 4096 entries. Used factors must be positive and finite. Only the used entries affect cached frequencies.
- Positive finite frequency base and scale; finite extension, attention and beta values. Active YaRN extension additionally requires a positive original context, base greater than one, and positive beta values. Truncated and continuous correction dimensions use ggml_rope_yarn_corr_dims().

MROPE, IMROPE, VISION, ROPE_BACK, nonzero rotation offsets, gapped activation layouts, quantized/BF16 activations and unsupported parameter profiles remain on CPU through capability rejection. Negative/out-of-range IDs, invalid factors and nonfinite/overflowing boundaries fail direct computation before any partition output is published.

The GGML physical layout is [B,T,H,D]. Host packing changes it to QNN [B,H,T,D]; output unpacking reverses the change. QNN activation, cosine, sine and output tensors are FP16. GGML F32 boundaries are converted to FP16 for the rotated channels. Partial RoPE copies unrotated channels from original GGML bytes, preserving them exactly even for F32.

Selected cosine/sine inputs have shape [B,T,rotary_dimension/2]. QNN position_ids is omitted because selection already occurred on the host. rotary_embedding_dim is UINT32; interleaved is BOOL8, true for NORMAL and false for NEOX. The released SDK documents the interleaved name but does not provide a header macro for it.

### Cache and lifecycle

prepared_run owns an optional rope_tables object containing FP16 cosine/sine vectors, the number of prepared rows and selected-table input IDs. Each new profile precomputes at least 1024 positions. The table grows in powers of two when larger IDs arrive and computes only added rows. Changing position values, including irregular/repeated IDs, does not rebuild or finalize the QNN graph.

Profile keys include activation/position shapes and strides, all RoPE operation parameters and the exact used frequency-factor bit patterns. Changed factors or scaling parameters create a distinct profile. The key excludes position values and table capacity. No application tensor pointer or selected-row buffer is retained between calls.

Each profile's two host tables are limited to 64 MiB. They count in prepared_run.static_bytes alongside the existing logical retained-payload accounting, within the shared 1 GiB and 512-profile LRU bounds. Cache growth evicts other profiles while retaining the executing entry. Invocation packing remains under the existing 256 MiB staging budget. Cache invalidation and backend destruction release tables with their graph profiles.

```mermaid
sequenceDiagram
    participant G as GGML scheduler
    participant B as backend_context
    participant C as Host RoPE table cache
    participant Q as QNN HTP
    G->>B: ROPE activation, positions, optional factors
    B->>B: Validate layout, parameters, factors and IDs
    B->>C: Find parameter/factor/shape profile
    alt New profile or larger positions
        C->>C: Precompute missing FP16 sine/cosine rows
    end
    alt New graph profile
        B->>Q: Register x, selected cosine/sine and y
        B->>Q: Add RotaryEmbedding and finalize
    end
    B->>C: Copy rows indexed by current position IDs
    B->>B: Pack activation from B,T,H,D to B,H,T,D
    B->>Q: Bind fresh activation and selected rows; execute
    Q-->>B: FP16 rotated activation
    B->>B: Unpack; preserve partial suffix; stage output
    B-->>G: Publish only after every partition operation succeeds
```

### Accuracy and validation

RoPE is implemented but disabled by default: set GGML_QAIRT_EXPERIMENTAL_ROPE=1 before backend registration to enable it. This strict 0|1 setting is captured once and enforced by both supports_op and direct graph_compute. Native FP32 HTP arithmetic and global test thresholds remain unchanged.

The focused CPU-reference fixture passed 80,313 checks with zero failures. It covers both modes, F16/F32 boundaries, multiple heads/tokens/batches, partial rotation with exact suffix preservation, YaRN, factors and their mutation, irregular/repeated IDs, table growth beyond position 2048, graph reuse, in-place execution, a SiLU-to-RoPE predecessor chain, and failure atomicity. Its per-element comparison uses the existing local 0.004 relative/absolute tolerance and does not qualify the stricter generic suite.

The unchanged generic test-backend-ops ROPE suite passed 28/66 admitted cases in the final measured run. Failed NMSE values were approximately 1.00e-7 through 1.87e-7 against the unchanged 1e-7 limit, including F16 inputs. Earlier runs passed 30/66; cases near the threshold vary with randomized inputs. FP16 table/boundary/native arithmetic precision remains unqualified, so the experimental gate stays closed by default.

Run the focused fixture with:

```powershell
$env:GGML_QAIRT_DEVICE = 'HTP'
$env:GGML_QAIRT_LIBRARY_DIR = 'C:\Qualcomm\AIStack\QAIRT\2.52.0.261104\lib\aarch64-windows-msvc'
$env:GGML_QAIRT_EXPERIMENTAL_ROPE = '1'
& C:\Github\qairt-m2-arm64\bin\Release\test-qairt.exe --rope
```

The default support CSV continues to mark ROPE unsupported because it records default capability settings. Hardware CTest qairt-m2-rope enables the gate for its own process. Model decode/reset validation with the gate disabled passed with maximum decode KL 0.00091876, greedy agreement 7/8, reset KL 0.00001993 and 8 preparations / 142 hits / 150 QNN executions. That default hybrid model test does not establish model-level RoPE offload or complete HTP attention.

## Initial FP16 accuracy budget (November 15, 2026)

The user-approved initial QAIRT HTP operator budget is NMSE <= 1e-5, where NMSE = sum((reference - actual)^2) / sum(reference^2). This corresponds to normalized RMS error <= sqrt(1e-5), approximately 0.3162%. It is a squared aggregate error budget, not an elementwise absolute or relative tolerance and not an IEEE-mandated accuracy bound.

The budget applies to the current QAIRT HTP adapter's FP16 execution, including F32 GGML inputs/outputs converted at FP16 graph boundaries. It does not add native FP32 HTP arithmetic. The generic test-backend-ops backend-specific tolerance hook uses max(existing operation tolerance, 1e-5) for QAIRT; existing operation-specific overrides remain in force. Other backends retain their existing tolerances. Future QAIRT GPU/CPU adapters must select a precision-appropriate policy before being enabled.

Focused test-qairt tensor comparisons additionally enforce NMSE <= 1e-5 for nonzero reference tensors and retain abs(actual-reference) <= 0.004 * max(1, abs(reference)), or the fixture's explicitly tighter elementwise bound. Zero-reference tensors use that existing absolute bound. Finite-value checks, exact layout/suffix checks, failure atomicity checks and cache checks remain unchanged.

Model acceptance remains KL(CPU||HTP) <= 0.001, maximum absolute logit error <= 0.75, intermediate relative RMSE <= 0.02 and at least 7/8 teacher-forced greedy agreement. Logit relative RMSE remains diagnostic. These model metrics are separate from the operator NMSE budget.

Earlier strict results at 1e-7 are historical baseline measurements, not results under the new budget. Experimental Add, Softmax and RoPE gates remain disabled by default pending qualification under this policy and relevant model-level validation. The budget alone does not qualify wider offload, attention, perplexity or long free-running generation.

### Validation under the 1e-5 budget

Native ARM64 Release builds of test-backend-ops, test-qairt and test-qairt-model succeeded. Existing MSBuild warnings remain. Individual generic operator runs used test-backend-ops.exe test -b QAIRT0 -o <operation> -j 1, enabling each relevant experimental gate for its process.

| Operator or test group | Measured result |
|---|---|
| Residual Add | 57/57 passed |
| RoPE, isolated process | 66/66 passed |
| SiLU | 4/4 passed |
| RMSNorm and LayerNorm | 52/52 passed |
| Softmax | 20/23 passed |
| QAIRT and backend-tensor CTests, including focused NMSE checks | 15/15 passed |
| Default supplied-model decode/reset | Zero failures; maximum decode KL 0.00091876; greedy agreement 7/8; reset KL 0.00001993; 8 preparations / 142 hits / 150 executions |

Remaining Softmax failures have channel counts 200000 and 643251, with measured NMSE approximately 1.0697e-5, 1.0772e-5 and 9.5517e-5. They exceed the approved budget and remain unqualified.

The combined six-operator process passed 198/202 cases: three Softmax numerical failures and one RoPE QNN transport/graph-execution failure. RoPE subsequently passed all 66 cases in a fresh isolated process; this does not resolve the combined-run runtime failure. Default model validation kept experimental Add, Softmax and RoPE disabled and therefore does not qualify their model-level offload. No experimental gate was enabled by default and no commit was created.

### Deferred large-channel Softmax tests

At the user's request, the unmasked F32 Softmax fixtures with shapes [200000,1,1,1], [200000,4,1,1] and [643251,3,1,1] have been removed from the shared test-backend-ops fixture list for now. Their scale is 1, max_bias is 0 and execution is not in-place. Other backend runs also omit these three shared fixtures. After removal, the native ARM64 Release test-backend-ops build succeeded and the remaining admitted QAIRT Softmax cases passed 20/20 with the experimental gate enabled and the 1e-5 NMSE budget. Changes remain local and uncommitted.

This is a test-coverage deferral, not an accuracy fix or qualification of large-channel HTP Softmax. CPU outputs rounded only to FP16 measured approximately 9.05e-6 NMSE for 200000 channels and 9.40e-5 for 643251 channels. All output probabilities in those measurements were below the FP16 minimum normal value; subnormal output rounding materially limits accuracy. The initial 1e-5 budget, current capability checks and experimental Softmax gate remain unchanged. Restore these three fixtures when revisiting output precision, large-channel support or the corresponding numerical policy.

### Latest combined operator suite after Softmax test deferral

The combined native ARM64 hardware suite was rerun on November 15, 2026 after removing the three large-channel Softmax fixtures. It passed 198/199 admitted cases and exited with status 1. Add, Softmax and RoPE experimental gates were enabled for this test process; default backend configuration remains unchanged.

| Operator group | Combined-process result |
|---|---|
| Residual Add | 57/57 passed |
| Unmasked Softmax, retained coverage | 20/20 passed |
| SiLU | 4/4 passed |
| RMSNorm and LayerNorm | 52/52 passed |
| RoPE | 65/66 passed; one QNN execution failure |

Failing case: ROPE(type=f32,ne_a=[80,32,2,1],n_dims=32,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0,inplace=0,n_offs=0,n_ctx_orig=0,freq_base=10000.000000,beta_fast=1.000000,truncate=1).

QNN reported "Failed to prepare graph 392. Error 6020 through transport layer", followed by "Graph rope_run failed in execution with err 6001" and graphExecute failure. The test then reported NMSE 1.931298187 against the 1e-5 limit; this value follows a failed execution and does not qualify the numerical accuracy of a successfully executed RoPE graph.

The combined-run runtime failure is now reproduced after removal of the large-channel Softmax cases. The earlier combined run failed near graph preparation 391 on a different RoPE case, while isolated RoPE previously passed 66/66. The precise cause and SDK/driver resource limit remain unknown. Large Softmax fixture removal therefore leaves a separate cumulative runtime stability gap.

Remaining M2 gaps are HTP grouped attention, masked Softmax and KV writeback, full-block/model qualification of experimental operations, wider-layer model accuracy and runtime/resource stability. The qualified one-layer hybrid decode/prefill scope and CPU attention/KV boundary remain unchanged. The removed Softmax coverage remains deferred. M2 is incomplete and uncommitted.

Reproduction command, with GGML_QAIRT_DEVICE=HTP, the installed SDK runtime directory configured, and GGML_QAIRT_EXPERIMENTAL_ADD, GGML_QAIRT_EXPERIMENTAL_SOFTMAX and GGML_QAIRT_EXPERIMENTAL_ROPE all set to 1:

```powershell
test-backend-ops.exe test -b QAIRT0 -o ADD,SOFT_MAX,ROPE,SILU,RMS_NORM,NORM -j 1
```

### Rank-4 RoPE table experiment

At the user's request, the exact failing combined-suite RoPE case [80,32,2,1], rotary dimension 32 and NORMAL mode was tested with QNN cosine/sine descriptors changed from [1,2,16] to [1,1,2,16]. The matching application-side packet validation was changed for the experiment; activation/output descriptors remained [1,32,2,80], and tensor bytes and operation parameters were unchanged.

The experimental ARM64 build succeeded, but the isolated case failed 0/1. QNN rejected the graph node during backendValidateOpConfig, before finalization/execution:

```text
qairt: Input[1] has incorrect Rank 4.
qairt: Op specific validation failed.
qairt:  <E> validateNativeOps master op validator rope:qti.aisw:RotaryEmbedding failed 3110
qairt:  <E> Op validation failed in Htp User Driver! err = 3110
qairt:  <E> QnnBackend_validateOpConfig failed 3110
qairt: backendValidateOpConfig: QNN call failed
```

The combined rank-4 suite was not run because its exact isolated case was rejected. Both source files were restored exactly to their pre-experiment rank-3 content and rebuilt successfully. The restored exact isolated case then passed 1/1. The official SDK master contract requires rank-3 selected sine/cosine tables when QNN position_ids is omitted; the HTP supplement requires rank-4 activation input/output. The trial confirms that the proposed rank-4 table change is rejected by this SDK. The separate combined-suite runtime failure 6020/6001 remains unresolved. No implementation change or commit was retained from this experiment.

## Standard isolation for QAIRT operator correctness tests

QAIRT cases in `test-backend-ops test` now use a fresh backend/runtime for each supported case. The harness releases its outer QAIRT backend owner, applies operation/parameter filters and device support checks, and then initializes the selected case's backend. RAII destroys the case's ggml buffers and contexts before its QAIRT backend, prepared QNN graphs/contexts, device, backend, and logger are released.

QAIRT correctness cases run with one worker even if `-j` requests more workers. Concurrent backend objects share the registry runtime and would prevent complete teardown between cases. This policy requires no additional environment flag. Other backends and performance/support/gradient test modes retain their existing lifecycle. Experimental operator gates and the 1e-5 FP16 NMSE budget still apply.

The original combined ADD, SOFT_MAX, ROPE, SILU, RMS_NORM, NORM run passed 198/199 cases and failed a RoPE graph with internal preparation error 6020 followed by execution error 6001. The full-teardown experiment passed 199/199 cases and verified 201 complete runtime teardowns, with zero active contexts, matching acquired/released handle counts, and zero release failures. That verified lifecycle is now standard in the correctness harness.

Runtime caches remain active within each case and in normal model execution. Cross-case cache accumulation is deliberately excluded from this operator suite; cache reuse is covered by focused cache tests and model execution checks. Repeated initialization increases correctness-suite runtime. This test policy does not establish the cause of the SDK preparation failure or qualify production model execution under accumulated graph resources.
Standard lifecycle verification on November 15, 2026: native ARM64 test-backend-ops built successfully. The combined QAIRT ADD, SOFT_MAX, ROPE, SILU, RMS_NORM, NORM suite passed 199/199 with no reset flag or temporary teardown instrumentation. The previously failing F32 RoPE case [80,32,2,1], rotary dimension 20, NORMAL mode passed 1/1 when -j 4 was requested; the QAIRT harness applied its standard one-worker isolation policy. CPU SiLU passed 12/12 with two workers. No preparation/execution/cleanup failure signatures were observed. Other-backend fixture counts differ from QAIRT admission counts; the CPU verification helper was corrected to expect its 12 supported SiLU cases.

The permanent change is in tests/test-backend-ops.cpp. The production QAIRT runtime and its model-execution cache policy were not changed. Repository QAIRT documentation and the M2 detailed design record the lifecycle. Changes remain uncommitted.
## FP16 masked Softmax implementation and model validation (November 15, 2026)

This section supersedes earlier statements that masked model Softmax is unsupported or necessarily stays on CPU. The implemented FP16 QNN graph is ElementWiseAdd(logits, additive_mask), followed by ordinary Softmax(axis=3, beta=1). It preserves [batch, heads, queries, keys]. Host preparation applies GGML scale before the mask, centres each valid row before FP16 conversion, and crops unused padded mask rows. Exact negative-infinity mask entries use -65504 and a zero source logit. Both application inputs are rebound on every invocation, including cache hits and shared-input aliases. Invalid values and entirely excluded rows fail before publishing outputs.

Native ARM64 Release builds succeeded, including the full build. Generic Softmax passed 64/64 admitted cases, including 44 masked cases; the combined ADD, SOFT_MAX, ROPE, SILU, RMS_NORM, NORM suite passed 243/243. QAIRT/backend-tensor CTests passed 16/16, including the dedicated masked fixture. Operator comparisons retain the approved NMSE <= 1e-5 budget.

The model fixture now sets offload_kqv=true when GGML_QAIRT_EXPERIMENTAL_SOFTMAX=1. With offload_kqv=false, actual attention Softmax stayed on CPU even after masked conversion was implemented. Weight placement remains n_gpu_layers=2 and op_offload=false, selecting one repeating layer while retaining the vocabulary projection on CPU. Executed-tensor callbacks, rather than reserved-graph scheduler messages, verify actual placement and require kq_soft_max-15 to use QAIRT.

Using C:\Models\Llama-3.2-1B-Instruct-Q4_K_M.gguf with all three experimental gates enabled, decode/reset, prefill128 plus tail13/decode/reset, and context1024/rejected append/reset each exited 0 with zero failures. All five observed tensors (Qcur-15, Kcur-15, ffn_inp-15, l_out-15, kq_soft_max-15) used QAIRT in each profile. Each profile passed 34 sampled intermediate comparisons.

| Profile | Maximum prompt/decode KL(CPU||HTP) | Maximum prompt/decode absolute logit error | Greedy agreement | Preparations / hits / executions |
| --- | --- | --- | --- | --- |
| Decode/reset | 0.00058681 | 0.219574 | 8/8 | 12 / 228 / 240 |
| Prefill128 + tail13, decode/reset | 0.00015755 | 0.197966 | 8/8 | 28 / 146 / 174 |
| Context1024, rejected append/reset | 0.00015767 | 0.079888 | Not applicable | 22 / 106 / 128 |

Reset KL was 0.00003911 in all three profiles. Acceptance remains KL <= 0.001, maximum absolute logit error <= 0.75, intermediate relative RMSE <= 0.02, and greedy agreement >= 7/8. Graph caches remain active throughout model execution; per-case runtime teardown belongs only to the operator correctness harness.

Live DEBUG diagnostics validated FP16 Add and Softmax nodes and successful QnnGraph_finalize (status 0x0). No MaskSoftmax or fusion-selection messages were observed; this does not prove that no internal optimization occurred. The functional contract depends on regular FP16 operations, not quantized fusion.

Default experimental gates remain disabled. Wider layer offload, complete HTP attention/KV functionality, context shifting, multiple sequences, perplexity, and long free-running generation remain outside this qualification. The full build retained existing duplicate-object and unavailable embedded-UI warnings. SDK startup/driver-option warnings remain. Changes are local and uncommitted.

## BF16 boundary conversion implemented (November 16, 2026)

BF16 MatMul weights and F32/BF16 activation operands now execute through the existing QNN FP16 recipe, with F32 GGML results. Original BF16 bytes remain unchanged. BF16 storage is expanded with GGML helpers, checked for finite values and FP16 overflow, and converted to FP16. F32 activations feeding BF16 weights are first rounded to BF16, matching GGML CPU dot-product preparation. Numerical BF16 outputs, other BF16 numerical operators, F16 activations paired with BF16 weights, and native BF16 QNN arithmetic remain unsupported.

BF16 metadata views preserve their byte layout; tests cover transposed activations. GGML's MatMul constructor forbids transposed weights. The private input_preparation enum distinguishes ordinary, Q8_K and BF16 input preparation in cache/source deduplication. Connected MatMul runs split when an intermediate needs BF16 or Q8_K activation preparation. Dynamic inputs are rebound; static weights use existing allocation generations, retained graphs and LRU limits. Invalid input/weight values reject computation before any staged output is published.

Validation completed:
- Native ARM64 affected targets and the full Release build passed.
- Module/shared and static QAIRT targets built successfully; SDK-disabled GGML/CPU targets built successfully. The SDK-disabled configuration does not define test-backend-tensor, so no OFF CTest result is claimed.
- BF16 host admission: 184 checks, zero failures.
- Dedicated HTP BF16 fixture: 4,369 checks, zero failures; maximum matched-input CPU-reference NMSE 9.47822424e-8.
- Existing BF16 MatMul operator suite: 99/99 admitted cases passed after applying the approved 1e-5 NMSE limit specifically to QAIRT BF16 MatMul. Other backend/operator tolerances are unchanged.
- QAIRT/backend-tensor CTests: 18/18 passed in each of native, module/shared and static configurations.
- The representative default BF16 MatMul support CSV row now reports supported. Other representative rows remain unchanged.

### Preserved full-chain precision limitation

The connected identity-MatMul/BF16-MatMul fixture also measures the original CPU chain separately. The FP16 QNN predecessor turns 1.00395 into 1.00390625, a BF16 rounding midpoint. The subsequent BF16 preparation can therefore select a different value from the CPU chain. Measured full-chain NMSE was 6.30041424e-5, above 1e-5; the fixture explicitly reports full_chain_qualified=0.

Conversion acceptance compares CPU and QNN with identical actual boundary inputs and separately checks the predecessor's FP16 error. The original chain drift is retained as a diagnostic, not removed or treated as qualified. These passing component tests do not qualify arbitrary BF16 model chains.

GGML_PREC_BF16 permits BF16 or F32 accumulation. This change does not admit explicit BF16/F32 accumulator requests or prove QNN's optimized accumulation precision. Such attention operations remain on CPU. The QNN reference FP16 MatMul kernel uses qf32 accumulation, but that alone does not identify the released SDK's optimized kernel. The supplied Q4_K model has not provided BF16-weight model qualification.

Existing SDK startup and MSVC duplicate-object/embedded-UI warnings remain. Source review checked type/layout bounds, buffer ownership, original-byte preservation, cache preparation identity, connected-run conversion and output atomicity. No GGML core, public C ABI or scheduler change was made. Changes remain on Add-ggml-qairt-backedn, locally uncommitted.

Verification commands:
```powershell
test-qairt.exe --bf16-host
test-qairt.exe --bf16
test-backend-ops.exe test -b QAIRT0 -o MUL_MAT -p 'type_a=bf16'
# Older CPU-reference suites require their original compatibility preparation.
$env:GGML_QAIRT_DIRECT_FP16 = '0'
ctest --test-dir C:\Github\qairt-m2-arm64 -C Release -R 'qairt-|test-backend-tensor' --output-on-failure
# default-fp16 and direct-fp16 cases explicitly select the direct path.
$env:GGML_QAIRT_DIRECT_FP16 = '1'
```

## Grouped attention and KV writeback - November 17, 2026

Set `GGML_QAIRT_EXPERIMENTAL_ATTENTION=1` before backend registration to enable the new attention and KV writeback profiles. The default remains disabled. This setting does not relax explicit GGML accumulator precision.

`GGML_OP_FLASH_ATTN_EXT` has a conservative causal grouped-attention conversion. Query storage can be F32/F16; K/V and mask storage are F16. Query/KV heads must divide evenly, key/value head widths must match, head width must be a multiple of 32 and at most 256, queries at most 128, keys at most 4096, and query heads at most 64. The initial profile has one sequence. Masks must contain a populated causal prefix with zero for visible keys and negative infinity elsewhere; padded query rows are allowed. ALiBi, softcap, sinks, sparse KV lists, sliding windows, arbitrary masks and explicit nondefault accumulator/source precision remain rejected. Mask values are checked at execution time; unsupported data fails without publishing results.

The preferred QNN packet uses `GroupQueryAttention`, INT32 sequence metadata, ten legacy input positions with explicit NULL optional tensors, and three outputs. Fresh Q/K/V are BSH; past/present K/V are BNSH. Already rotated Q/K are passed with `do_rotary=0`. Each invocation packs real GGML strides, separates the populated past and fresh K/V slices, and validates QNN present-cache outputs against the populated GGML cache before publishing attention. This native path compiles but is not hardware-qualified on this machine.

The installed Windows HTP driver loads `QnnHtpPrepareDrv.dll` from `DriverStore/FileRepository/qcnspmcdm8380.inf_arm64_6a2c30a0aec80e34/HTP`. Its master validator rejects the native GQA packet at `backendValidateOpConfig` with 4005 (`QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND`). Canonical RAW NULL descriptors still fail; reference source accepts the packet contract. Native GQA therefore remains unavailable on this installed driver. No driver files were changed.

When native validation specifically returns op-package-not-found, the backend remembers this per stream and executes grouped attention as QNN MatMul, ElementWiseAdd, Softmax, and MatMul. One finalized single-head graph is reused across query heads; query head h uses KV head floor(h / (num_heads / kv_num_heads)). Host work only packs layouts, scales query inputs and encodes the mask. Both attention multiplications and Softmax execute through QNN. The graph key covers shapes/strides, populated KV length and scale; inputs are rebound for every execution. Other QNN failures are returned, not hidden by this fallback.

Compatible rank-two `GGML_OP_SET_ROWS` now uses QNN ScatterNd with reduction NONE. Cache storage is F16, update storage is F32/F16, and contiguous I32/I64 indices are checked before narrowing to QNN INT32. Duplicate, negative, overflowing and out-of-range indices are rejected. QNN receives separate cache/update/index buffers and produces a staged full-cache output. Untouched rows are verified bit-exact. The cache is published only after every operation in the GGML partition succeeds. Quantized, strided and multi-stream cache writes retain CPU fallback; llama.cpp still owns cache allocation, indexing, capacity and reset.

Validation on ARM64:

- Focused attention/cache fixture: 12,565 checks, zero failures. Eighteen attention cases cover MQA/GQA/MHA, prefill, single/multiple-token decode and permuted query storage. Maximum isolated attention NMSE is 2.3402265e-6 against CPU; connected KV-writeback/attention NMSE is 6.78879923e-7, both below 1e-5. I32/I64 cache writes match CPU bit-exactly. Invalid-index, duplicate-index, failed-successor and invalid-mask cases preserve prior cache/output bytes.
- Native QAIRT CTests: 20/20 passed, including the two new attention tests and existing operator regressions.
- The one-layer model decode/reset fixture passes with all experimental gates enabled. Both K and V writes for layer 15 execute in QAIRT buffers. Prompt KL is 0.00034499; maximum observed decode KL is 0.00058681; greedy disagreement is 0/8; reset KL is 0.00003911. Totals: 13 preparations, 242 cache hits, 255 QNN executions.
- Prefill128 plus tail13 and eight decode steps passes with actual QAIRT K/V writes. Prompt KL is 0.00000038; maximum observed decode KL is 0.00015755; greedy disagreement is 0/8; reset KL is 0.00003911. Totals: 31 preparations, 154 cache hits, 185 QNN executions.

These model fixtures retain conventional attention. Their score MatMul requests BF16 accumulation and remains on CPU. llama.cpp's fused attention graph explicitly requests F32 accumulation; the new default-precision GQA profile does not silently override it. End-to-end model attention offload still requires a verified QNN accumulation contract or an explicitly approved relaxed-precision policy. Wider-layer quality/resource qualification and broader architectures remain open. The native GQA path requires a compatible installed driver and hardware validation. No local commit was created.
## M2 revalidation and remaining gaps - November 17, 2026

Sparse attention is rejected using GGML op-parameter slot 4 (`n_kv_max`), independently of accumulator precision in slot 3. A regression sets `n_kv_max=1`, requires rejection, restores zero and requires admission. Before the correction, the host fixture reported 18 failures; after correction, the host fixture and all 18 hardware attention cases passed. The hardware attention/cache fixture now reports 12,565 checks and zero failures, maximum isolated attention NMSE 2.3402265e-6, and connected KV-writeback/attention NMSE 6.78879923e-7. The existing 1e-5 budget remains unchanged.

Full native, module/shared and static ARM64 Release builds passed after this correction. The SDK-disabled full build previously passed with `LLAMA_BUILD_APP=OFF`; no SDK-disabled runtime QAIRT result is claimed. Existing unavailable embedded-UI and duplicate-object build warnings remain.

| Configuration | Fresh validation | Result |
| --- | --- | --- |
| Native ARM64 | QAIRT tests plus backend tensor test | 20/20 passed |
| Module/shared ARM64 | QAIRT tests plus backend tensor test | 20/20 passed |
| Static ARM64 | QAIRT tests, backend tensor test, batch allocator and server component tests | 22/22 passed |
| Native ARM64 combined operators | ADD, SOFT_MAX, ROPE, NORM, RMS_NORM, SILU | 243/243 admitted QAIRT cases passed; exit 0 |

Run CTests with experimental environment variables unset: individual experimental fixtures set their required gates through CTest properties. An initial run with all four gates enabled globally failed the M1 check that expects Add to be disabled. Repeating the suite with its intended default environment passed; this was a test invocation error, not a numerical tolerance change.

The one-repeating-layer model profile (`QAIRT_TEST_N_GPU_LAYERS=2`, output projection on CPU) was rerun with all four experimental gates enabled. Decode/reset completed with zero failures, prompt KL 0.00034499, maximum decode KL 0.00058681, reset KL 0.00003911 and greedy disagreement 0/8. Actual QAIRT K/V writes were observed. Runtime totals were 13 graph preparations, 242 cache hits and 255 executions.

### Wider model accuracy remains unqualified

The four-repeating-layer profile (`QAIRT_TEST_N_GPU_LAYERS=5`) completed execution but failed 15 accuracy checks. Prompt KL was 0.00072964 and maximum decode KL was 0.00179832, exceeding the unchanged 0.001 limit. Layer-14 and layer-15 V-cache relative RMSE were 0.030411 and 0.082521, exceeding 0.02. Greedy disagreement was 1/8, within its limit. Runtime totals were 31 preparations, 854 hits and 885 executions. These are accuracy failures, not graph-finalization or execution failures.

Disabling experimental Add, RoPE and Softmax while retaining experimental KV writeback did not resolve wider accuracy: 13 checks failed, maximum decode KL was 0.00187927, and layer-15 V-cache relative RMSE was 0.061234. The remaining error therefore cannot be attributed solely to those three experimental operations. The ARM CPU `quantize_row_q8_K` wrapper calls `quantize_row_q8_K_ref`, which is also used by QAIRT activation preparation; inspection did not establish an ARM/reference quantizer mismatch. The cause of accumulated model drift remains under investigation. No threshold was relaxed and no passing wider-model qualification is claimed.

### Remaining M2 acceptance work

- Determine and correct wider-layer numerical drift while retaining operator and model budgets; extend model qualification only after passing it.
- Verify the installed HTP implementation's accumulation contract before admitting explicit BF16/F32 accumulator requests. Standard llama.cpp conventional attention requests BF16 accumulation, and fused attention requests F32 accumulation; both remain on CPU. The passing grouped-attention fixture uses default precision and the QNN decomposition path, so it does not establish full model attention offload.
- Native QNN GroupQueryAttention remains unavailable on the installed driver: `backendValidateOpConfig` returns 4005 (`QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND`). The backend falls back to QNN decomposition only for that validation error. Native GQA requires a compatible driver/op package and separate validation.
- Qualify broader prefill/decode, context-state behavior and model-quality coverage beyond the currently passing limited hybrid profile. Multiple sequences, context shifting and long free-running/perplexity coverage remain unqualified.

M2 remains incomplete. Changes remain local and uncommitted on `Add-ggml-qairt-backedn`.
### Same-input projection diagnostic implemented

`QAIRT_TEST_LOCAL_REFERENCE=1` enables CPU replay of every observed contiguous F32 QAIRT MatMul output using the same live operands. A separate output allocation and a single-node graph preserve model tensors and do not recompute input producers. Manually added reference nodes carry `GGML_TENSOR_FLAG_COMPUTE`; `ggml_graph_add_node` does not set it, and the CPU executor skips nodes lacking it. Overlapping input/output storage is skipped. The setting accepts only 0 or 1 and is disabled by default.

On the four-repeating-layer profile, 420 projection replays passed the unchanged 1e-5 local operator budget, with no skipped cases. Maximum local NMSE was 2.98905289e-7 at `Vcur-14`. Prompt/decode/reset KL and greedy disagreement matched the run without CPU replay, while the same 15 model-level accuracy checks failed. The diagnostic changes scheduler partition boundaries (34 preparations, 896 cache hits, 930 HTP executions), so those counts must not be used as ordinary-model performance measurements. Small same-input projection error does not qualify a complete chain whose activation inputs already differ; normalization, other operations and subsequent quantization boundaries still need investigation.

Reproduction:

```powershell
$env:GGML_QAIRT_DEVICE = 'HTP'
$env:GGML_QAIRT_LIBRARY_DIR = 'C:\Qualcomm\AIStack\QAIRT\2.52.0.261104\lib\aarch64-windows-msvc'
$env:GGML_QAIRT_EXPERIMENTAL_ATTENTION = '1'
$env:GGML_QAIRT_EXPERIMENTAL_ROPE = '1'
$env:GGML_QAIRT_EXPERIMENTAL_SOFTMAX = '1'
$env:GGML_QAIRT_EXPERIMENTAL_ADD = '1'
$env:QAIRT_TEST_N_GPU_LAYERS = '5'
$env:QAIRT_TEST_LOCAL_REFERENCE = '1'
& C:\Github\qairt-m2-arm64\bin\Release\test-qairt-model.exe C:\Models\Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

This four-layer diagnostic is expected to fail the model acceptance gates until wider numerical drift is resolved. Its passing local projection comparisons do not change that result. The corrected diagnostic built successfully in native, module/shared and static ARM64 configurations.

The one-repeating-layer profile also passed with local CPU replay enabled: 105 projection comparisons, no skipped cases, maximum NMSE 2.32083294e-7 at Vcur-15, and zero model failures. Its prompt/decode/reset KL and greedy agreement matched the ordinary validation run. The diagnostic introduced no changes to the accepted precision or model budgets.
