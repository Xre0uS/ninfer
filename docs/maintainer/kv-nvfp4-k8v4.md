你的任务是在 NInfer 当前 paged causal attention 中，为 KV cache 完整增加并优化两种新的量化模式：

1. `nvfp4`
2. `k8v4`

不要只给出设计方案。需要完成实现、独立正确性验证、benchmark 接入、性能优化、产品配置接入，以及所有受影响的活动文档更新。

两个模式必须作为两条独立路线实现、测试、调优和验收。可以共享数学语义完全相同的底层 primitive，例如 Hadamard、E2M1 codec 和页地址计算，但不得：

- 在热点循环中通过运行时分支混合两种模式；
- 用一个 mega-kernel 承载两套不同的数据格式和 MMA 路径；
- 让一种模式静默 fallback 到另一种模式、FP8 或 BF16；
- 用一个模式的正确性或性能结果替代另一个模式的独立验证。

建议先完整实现并验收 `nvfp4`，再实现 `k8v4`。

# 一、首先确认项目现状和已有习惯

开始实现前，只检查与本任务直接相关的当前代码和活动文档：

- `AGENTS.md`
- `include/ninfer/types.h`
- `include/ninfer/ops/softmax_attention.h`
- `include/ninfer/ops/kv_cache_append.h`
- `src/core/paged_kv_cache.h`
- `src/ops/kv_cache/d256_profile.h`
- `src/ops/kv_cache/hadamard_d256.cuh`
- `src/ops/common/mma.cuh`
- `src/ops/linear/nvfp4/nvfp4_codec.cuh`
- 当前 FP8 prompt/small-T attention 和 KV append 实现
- `tests/ops/softmax_attention/causal_cache.cpp`
- `tests/ops/test_kv_cache_append.cpp`
- `bench/ops/causal_softmax_attention_bench.cu`
- KV append benchmark
- `bench/README.md`
- `docs/maintainer/op-development.md`
- `docs/maintainer/paged-kv-cache.md`
- `docs/maintainer/tensor-formats.md`

可以通过以下历史提交了解 FP8 KV 和 Hadamard 路线的拆分方式，但当前代码和活动文档才是权威：

- `6183c9be feat(attention): add fp8 kv cache support`
- `17a7275f feat(attention): add int8 kv hadamard rotation`

在开始修改前，先在工作更新中简要说明：

1. 当前 FP8 KV 的 codec、standalone/fused append、prompt、small-T、test 和 benchmark 如何组织；
2. 当前 KV 结构中哪些 K/V 对称假设阻碍 `k8v4`；
3. 两个新模式准备采用的物理布局；
4. QK、softmax、PV 和 Hadamard 的计算精度；
5. 两个模式的源码、dispatch、测试和性能归属边界。

不要为此创建永久 roadmap 或并行设计文档。

# 二、两个模式的精确数学语义

使用项目已有的归一化 Hadamard：

    R = H256 / 16

R 是正交且对称的，因此：

    RᵀR = I
    Rᵀ = R

## 2.1 `nvfp4`

定义：

    Kc = Q4(RK)
    Vc = Q4(RV)
    Qc = Q4(RQ)

其中 `Q4` 是本任务定义的 NVFP4 group-16 E2M1 cache codec。

QK 计算使用：

- Q：NVFP4 group-16；
- K：NVFP4 group-16；
- native NVFP4 block-scaled Tensor Core；
- UE4M3 block scale；
- FP32 accumulator。

当前项目 `mma_nvfp4_e4m3` 使用的 PTX 形式为：

    ...f32.e2m1.e2m1.f32.ue4m3

因此，本任务中的 NVFP4 QK accumulator 固定为 FP32，不把 accumulator dtype 留作未决选项。

逻辑计算为：

    logits = dequant(Qc) · dequant(Kc)ᵀ
    P = softmax(mask(logits * attention_scale))
    Z = P · dequant(Vc)
    O = RᵀZ

最终输出在完整 FP32 merge、normalization 和 inverse Hadamard 后转换为 BF16。

## 2.2 `k8v4`

定义：

    Kc = Q8(RK)
    Vc = Q4(RV)
    Qc = Q8(RQ)

其中：

- `Q8` 完全复用当前 FP8 cache 的 row-scaled E4M3、group-256 codec；
- K 的格式、scale、rounding 和 Hadamard 语义与当前 FP8 cache 完全一致；
- Q 完全复用当前 FP8 attention 的量化和 Tensor Core 路径；
- V 使用与 `nvfp4` 相同的 NVFP4 group-16 codec。

QK 使用：

- FP8 Q；
- FP8 K；
- native FP8 Tensor Core；
- FP32 accumulator。

逻辑计算为：

    logits = dequant(Qc) · dequant(Kc)ᵀ
    P = softmax(mask(logits * attention_scale))
    Z = P · dequant(Vc)
    O = RᵀZ

## 2.3 Hadamard 的明确范围

“两种模式都应用 Hadamard”明确包含 K、Q 和 V：

- Q 和 K 使用同一个 R，以保持旋转前后的点积语义；
- V 在写入 cache 前计算 `R·V`；
- attention 对解码后的旋转 V 做加权求和；
- split merge 和 normalization 完成后，对结果执行 `Rᵀ`；
- inverse Hadamard 必须在 FP32 中执行；
- 虽然 R 对称，可以复用同一实现，但契约和代码应表达为 inverse/transpose 语义。

不得只旋转 Q/K 而保持 V 不变。

# 三、NVFP4 cache codec

NVFP4 cache 复用项目已有的 group-size-16 E2M1 数值格式，但不能直接把权重 artifact 的矩阵布局当成 KV cache 布局。

需要为 KV cache 定义唯一、精确、可独立测试的 codec：

- group size 固定为 16；
- 每两个 E2M1 code 打包到一个 U8；
- 较小的 d index 使用低 nibble；
- 每组保存一个已有格式的 E4M3/UE4M3 scale；
- 使用当前 NVFP4 codec 对应的 RNE 和 finite saturation 语义；
- 明确定义全零组、scale 编码为零、最小非零 scale、rounding tie 和 saturation 行为；
- production codec 与测试 oracle 不得共享同一份量化实现。

KV cache 不应继承 NVFP4 weight artifact 的矩阵级全局 divisor。cache 的逻辑值由每个 group 的 E2M1 code 和 E4M3 scale 唯一决定。若当前 native MMA helper 对 scale representation 有额外布局要求，应只在物理 staging 中完成转换，不改变 cache 的逻辑 codec。

不要为 FP4 增加需要 `dtype_size = 0.5` 的通用标量 DType：

- E2M1 code plane 使用 packed U8；
- scale plane 使用当前已有的 E4M3 表示；
- codec 语义由封闭的 KV profile 表达；
- 不增加任意 bit width、任意 group size 的开放式量化框架。

# 四、KV cache 结构和物理布局

当前结构使用一个 dtype、一个 quant_group 或一个 vector-byte 值共同描述 K/V，这不能表达 `k8v4`。

需要将其重构为封闭的、分别描述 K 和 V 的 codec/layout profile。该 profile 至少能够明确：

- K code dtype/layout；
- K scale dtype/layout；
- K group size；
- K 每个 token/head 的逻辑和物理字节数；
- V code dtype/layout；
- V scale dtype/layout；
- V group size；
- V 每个 token/head 的逻辑和物理字节数。

不要用字符串、任意 bit width 或任意 group size 驱动 kernel。公开 storage mode 仍然是有限枚举，内部映射到有限的已注册 profile。

对于 D=256：

## `nvfp4`

每个 K 或 V vector：

- 256 个 E2M1 code；
- 两个 code/byte，共 128 bytes；
- 16 个 group；
- 每组一个一字节 E4M3 scale，共 16 bytes；
- 合计 144 bytes。

每个 token/head 的 K+V：

    144 + 144 = 288 bytes

## `k8v4`

K 复用当前 FP8 row-256：

- 256 bytes FP8 codes；
- 一个 FP16 scale，共 2 bytes；
- K 合计 258 bytes。

V 使用 NVFP4：

- 128 bytes codes；
- 16 bytes scales；
- V 合计 144 bytes。

每个 token/head 的 K+V：

    258 + 144 = 402 bytes

这些字节数必须一致反映到：

- page plane layout；
- capacity 计算；
- workspace 计算；
- target Program memory 报告；
- host/device context-cache replica；
- continuation/checkpoint；
- benchmark 的物理 cache bytes；
- append 和 attention 的地址计算。

K/V 的所有 plane 仍共享同一个 page-group identity、生命周期和逻辑 token frontier，但允许具有不同 code/scale plane geometry。

普通 paged causal attention 的 Main Text、包含 Vision prompt 的 continuation 和 MTP 路线应一致接入。DFlash 当前 BF16 full/local KV 状态保持不变，除非现有代码证明它直接依赖必须修改的公共结构。

# 五、计算精度

除非独立 oracle 和代表性真实 shape 数据证明该选择不可接受，采用以下唯一生产精度路径。

## 5.1 Hadamard

- 正向 Q/K/V Hadamard：FP32；
- inverse Hadamard：FP32；
- 不在 BF16、FP16、FP8 或 E2M1 中执行 Hadamard reduction；
- 最终仅在公开输出边界转换为 BF16。

## 5.2 QK

`nvfp4`：

- Q/K operand：E2M1；
- block scale：UE4M3；
- native NVFP4 Tensor Core；
- accumulator：FP32。

`k8v4`：

- Q/K operand：当前 FP8 E4M3；
- scale：当前 FP8 route 的既有格式；
- native FP8 Tensor Core；
- accumulator：FP32。

以下操作保持 FP32：

- attention scale；
- mask；
- online-softmax max；
- exp；
- sum；
- running-max rescale；
- split m/l/o state；
- split merge；
- normalization。

## 5.3 P 和 V

“P 不量化”表示 P 不转换为 FP8 或 FP4，但允许在 PV Tensor Core 输入边界进行一次 FP32→FP16 RNE cast。

采用：

- P 在 softmax、split state 和 merge 过程中保持 FP32；
- 只在送入 PV MMA 前转换为 FP16；
- NVFP4 V 的 E2M1 code 和 scale精确扩展并形成 FP16 operand；
- PV 使用 FP16 Tensor Core；
- PV accumulator 使用 FP32；
- 完整 merge 和 normalization 后执行 FP32 inverse Hadamard；
- 最终存储为 BF16。

优先使用 FP16 而不是 BF16 作为 P/V MMA operand，因为：

- P 位于 `[0,1]`；
- FP16 的尾数精度高于 BF16；
- 当前 FP8 attention 已采用相同的 FP16-PV、FP32-accumulator 路线；
- 关键归约、merge、normalization 和 inverse rotation 仍保持 FP32。

如果独立 oracle 和代表性真实 K/V 数据证明 FP16 的指数范围导致实际溢出、下溢或不可接受误差，可以改用 BF16，但必须提交对应证据。不要同时保留两个没有明确适用域的生产路线。

# 六、Kernel 和执行路线

两个模式分别提供所需的：

- prompt/prefill kernel；
- small-T/decode kernel；
- append-and-attend 路线；
- read-only cached-attention 路线；
- standalone KV append；
- fused append；
- launcher；
- workspace query；
- 有限 shape dispatch；
- CUDA Graph capture/replay 支持。

建议使用独立的模式级实现，例如：

- `prompt_nvfp4`
- `small_t_nvfp4`
- `prompt_k8v4`
- `small_t_k8v4`

具体文件名服从当前目录习惯，但实现边界必须保持独立。

允许共享：

- Hadamard primitive；
- E2M1 pack/decode；
- E4M3 scale decode；
- 页地址计算；
- 数学语义完全相同的 FP8 QK helper。

不得共享包含模式判断的热点主循环。

Prefill 的 QK 必须分别使用 native NVFP4 或 native FP8 Tensor Core。small-T 可以采用适合其形状的专用分块，但必须保持相同的 codec 和精度语义，并独立测量。

实现不得：

- 将完整 K、V 或 P 反量化到全局临时 buffer；
- gather paged cache 到连续 buffer；
- 对完整 K/V 做额外的第二次遍历；
- 因 GQA query-head 数量重复加载同一 KV；
- 在热点循环里按 cache mode 分支；
- 引入隐藏 device allocation；
- 破坏 CUDA Graph 地址稳定性；
- 为了复用旧 kernel 而长期保留无性能竞争力的 fallback。

目标是：

- K/V 和 scale 按 KV head/GQA group 流式读取；
- 在寄存器或 shared-memory tile 中完成解码、MMA 和归约；
- Q 量化尽可能与 QK 消费融合；
- append 的 K/V Hadamard、scale 计算、量化和存储尽可能融合；
- inverse Hadamard 尽可能与最终 FP32 输出处理融合；
- 避免任何完整中间张量的全局写回。

# 七、测试

为 `nvfp4` 和 `k8v4` 分别添加独立命名的测试。不得只通过 `all` 聚合入口间接覆盖。

## 7.1 Codec 和 KV append 精确测试

使用独立 CPU exact oracle，至少覆盖：

- E2M1 编码表；
- nibble 顺序；
- E4M3 scale bytes；
- 全零组；
- 最小非零 scale；
- rounding ties；
- 正负 saturation；
- group 边界；
- K Hadamard 后的精确 cache representation；
- V Hadamard 后的精确 cache representation；
- standalone append 与 fused append 的 byte-for-byte 一致性；
- page 63/64/65 边界；
- 非连续、带 offset 和 fragmented physical-page mapping；
- 未写区域保持不变；
- 输入 tensor 不被修改。

`k8v4` 必须分别验证：

- K 与当前 FP8 codec 完全一致；
- V 与 `nvfp4` 的 V codec 完全一致；
- 非对称 K/V plane 地址和字节数正确。

## 7.2 Attention 数值测试

使用独立 FP64 oracle验证完整逻辑公式：

1. 从公开 BF16 Q/K/V 输入开始；
2. 由独立 oracle 执行正向 Hadamard；
3. 执行对应模式的精确量化；
4. 精确解码 represented Q/K/V；
5. 使用 FP64 计算 QK、attention scale、mask 和 softmax；
6. 使用 FP64 计算 PV；
7. 执行 inverse Hadamard；
8. 与生产 kernel 输出比较。

生产 kernel 不得作为 oracle，两个生产模式也不得互相充当 oracle。

同时保留两类数值证据：

- 生产 kernel 相对“精确量化后”的 FP64 oracle，用于验证实现正确性；
- 量化模式相对未量化 BF16/FP64 attention 的误差统计，用于描述量化质量。

每个模式使用统一、由误差统计支持的 criterion，不为单个失败 case 定制 tolerance。

至少覆盖：

- append-and-attend；
- cached-attention；
- prompt；
- small-T；
- 当前支持的两组 GQA geometry；
- batch 1、2、4、8 的代表点；
- tile/route 边界的前一点、边界点和后一点；
- context 0/1/63/64/65/127/128；
- 代表性 2K、8K、16K context，以及至少一个 64K cached-attention case；
- masked/empty row；
- split merge；
- 连续和 fragmented page mapping；
- CUDA Graph capture/replay；
- workspace exact high-water；
- Main 与 MTP 的容量和状态事务。

最终输出看起来合理不能代替 codec、append 状态转换和 attention operator 的独立验证。

## 7.3 Perplexity 端到端数值验收

完成两个模式的 Engine 接入并通过聚焦正确性测试后，必须扩展
`ninfer-perplexity` 的 `--kv-dtype` 解析、`--help`、报告名称和 JSON 字段，使其支持精确名称
`nvfp4` 与 `k8v4`。

开发过程中可以使用固定语料的 `--quick` 模式做 smoke test；最终验收必须使用 64K context、同一个
明确指定的 registered `.ninfer` artifact 和完整 `ninfer-ppl-1m-v1` 语料，分别重新运行 `bf16`、
`fp8`、`nvfp4` 和 `k8v4`。不得用历史报告充当本次基线，也不得在四次运行之间改变 artifact、语料、
stream 顺序、context、stride、device 或其他执行设置。64K 协议固定为 `context=65536`、
`stride=32768`，保持与现有默认协议相同的半窗口步长关系。

基准命令采用以下协议；省略 `--quick` 表示运行 full corpus：

```bash
NINFER_PPL_MODEL=/absolute/path/to/registered-model.ninfer
NINFER_PPL_CORPUS=eval/corpora/perplexity-1m/manifest.json

./build/apps/ninfer-perplexity "$NINFER_PPL_MODEL" \
  --corpus "$NINFER_PPL_CORPUS" --context 65536 --stride 32768 --kv-dtype bf16
./build/apps/ninfer-perplexity "$NINFER_PPL_MODEL" \
  --corpus "$NINFER_PPL_CORPUS" --context 65536 --stride 32768 --kv-dtype fp8
./build/apps/ninfer-perplexity "$NINFER_PPL_MODEL" \
  --corpus "$NINFER_PPL_CORPUS" --context 65536 --stride 32768 --kv-dtype nvfp4
./build/apps/ninfer-perplexity "$NINFER_PPL_MODEL" \
  --corpus "$NINFER_PPL_CORPUS" --context 65536 --stride 32768 --kv-dtype k8v4
```

比较四份 `report.json` 时，先确认 artifact identity、tokenizer、corpus ID、stream/input/scored-token
数量、context 和 stride 完全一致，再分别报告：

- overall 和每个 domain 的 `mean_nll` 与 perplexity；
- `nvfp4` 相对 BF16、FP8 的绝对变化和相对变化；
- `k8v4` 相对 BF16、FP8 的绝对变化和相对变化；
- 是否存在集中在某个 domain 或长窗口中的异常退化。

Perplexity 是端到端数值质量证据，不能替代 operator oracle；报告中的 scored-token rate 也不能替代
attention benchmark。不得预先编造通用 PPL 阈值，但任何实质且未解释的退化都会阻止数值验收：
需要先定位量化、rotation、scale、append/cached 一致性或状态事务问题，修复后重新运行同一协议。

# 八、Benchmark 和性能目标

性能验收以完整公开 Op 的端到端 latency 为第一指标，不以理论 Tensor Core 峰值利用率作为硬门槛。

已有 INT8 和 FP8 attention 的经验表明，即使使用更低精度 Tensor Core，实际硬件利用率也可能只略高于 BF16。主要额外成本来自 kernel 内部的：

- Q 在线 Hadamard 和量化；
- append 时 K/V Hadamard、量化和 scale 生成；
- scale 加载；
- V 反量化；
- 数据重排和 shared-memory staging；
- 分页地址计算；
- softmax、merge 和 inverse Hadamard。

因此，不假设 NVFP4 理论 Tensor Core 吞吐能够按比例转化为 attention 端到端加速。

Tensor Core utilization、mixed roofline 和 contraction-only microbenchmark 只用于归因，不直接作为实现是否合格的标准。

## 8.1 Benchmark 接入

扩展现有公开 benchmark：

- `causal_softmax_attention_bench` 支持精确字符串：
  - `--kv-dtype nvfp4`
  - `--kv-dtype k8v4`
- KV append benchmark 同样支持两个模式；
- `all` 可以聚合运行，但两个模式必须分别输出；
- 非对称布局必须分别计算 `key_vector_bytes` 和 `value_vector_bytes`；
- 不得继续使用 `2 * cache_vector_bytes` 表达 `k8v4`。

分别报告：

- 完整公开 Op latency；
- 相对 BF16/当前 FP8 的加速或回退；
- logical cache bytes；
- 实际物理 cache bytes；
- effective cache bandwidth；
- QK FLOPs；
- PV FLOPs；
- append 与 cached 路线结果；
- 不同 context 下的性能交叉点；
- 量化与 Hadamard 成本的诊断性拆分；
- QK/PV Tensor Core throughput，作为解释指标而不是验收门槛。

## 8.2 比较基线

`nvfp4`：

- 主要与当前 FP8 KV 和 BF16 KV 比较。

`k8v4`：

- 主要与当前 FP8 KV 比较；
- 同时保留 BF16 作为参考。

所有比较必须使用相同的：

- B；
- W；
- context；
- GQA geometry；
- page mapping；
- append/cached 状态；
- CUDA Graph 条件；
- timing 方法。

至少测量：

- prompt 64、65、128、1024；
- small-T 的当前实际热区间；
- 所有 dispatch seam；
- context 128、2048、8192、16384、65536，以及当前支持上限中与 64K 不同的必要 anchor；
- 两种 GQA geometry；
- append-and-attend；
- read-only cached attention；
- standalone append；
- 连续和 fragmented page mapping；
- cold CUDA Graph 作为主要公开结果。

## 8.3 合理的性能完成标准

性能完成标准是：

1. 生产 dispatch 在每个相关区域选择已验证候选中最快或在测量误差内等价的完整实现；
2. 量化、Hadamard 和反量化尽可能融合，不产生完整 K/V/P 全局临时量；
3. 不增加额外的 cache 全量遍历；
4. 实际 cache 流量与声明的物理布局一致；
5. 不因 GQA、分页或非对称布局产生非必要重复加载；
6. 给出相对 BF16/FP8 的真实收益区间和性能交叉点；
7. 不要求所有 shape 都比现有路线更快；
8. 小 shape 因固定量化成本而回退是允许的，但必须测量并解释；
9. 长 context 或目标 bulk workload 未从更小 cache 流量中获得收益时，必须定位量化、Hadamard、反量化、同步或访存中的主导成本；
10. 不设置任意的 Tensor Core utilization 百分比；
11. 不要求 NVFP4 按理论峰值相对 BF16 获得相同倍数的端到端加速；
12. 不得把 contraction-only 的高吞吐当成完整 attention kernel 已优化的证据。

开发期间可以使用小规模私有候选 sweep 比较：

- 在线量化与其他 staging 方式；
- Hadamard 融合位置；
- tile/warp 布局；
- scale 加载和解码方式；
- prompt 与 small-T 的不同执行机制。

候选选择必须依据完整 kernel latency。单独的 codec、量化或 MMA benchmark 只用于解释成本。

选型完成后：

- 删除失败候选；
- 删除私有强制 route；
- 删除无生产用途的 sweep 入口；
- 保留公开 benchmark 和简洁结果。

只有当一个具体 kernel 仍存在未解释的性能差距时，才使用 ncu 检查：

- 实际 DRAM bytes；
- Tensor Core 活跃度；
- SM throughput；
- occupancy；
- shared-memory 行为；
- 同步；
- warp stalls。

不要为了达到预设利用率数字而进行无目标 profiling。只有完整推理中的归因仍然不清楚时才使用 nsys。

如果两个模式在所有代表性目标 workload 上都没有延迟、带宽或容量方面的可用收益，可以认为功能实现正确，但不能宣称性能优化已经完成。

# 九、产品接入

将两个模式作为正式 `KvCacheStorage` 选项接入。

外部名称严格为：

- `nvfp4`
- `k8v4`

不得添加旧名称、别名或兼容 fallback。

这里的接入范围是整个仓库内所有主动消费、解析、校验、展示、序列化或列举 Main KV storage 的
活动表面，不能把下面的文件列表当作穷举清单。开始实现时先使用 `rg` 建立完整 inventory；完成后
重新搜索 `KvCacheStorage`、`kv-dtype`、KV mode name、parser、formatter 和相关 switch，逐项确认：

- 通用 Main KV 入口完整支持 `nvfp4` 与 `k8v4`；
- 所有 enum switch、name formatter、JSON/CSV/report serializer 都有两个新模式；
- 所有 CLI/app/tool parser、`--help`、错误信息和 schema 都列出准确名称；
- 所有测试辅助器、benchmark matrix、measurement/perplexity 工具和公开报告都能保留两个模式的
  独立身份；
- 所有活动用户文档和 model card 中的允许值列表都已更新；
- 只有产品契约明确固定为其他存储的专用路线可以不接受新模式，并且必须是显式约束，不能由遗漏、
  default 分支或异常 fallthrough 形成；
- 不修改仅仅选择某个固定 KV 模式且没有枚举或声明支持集合的无关实验脚本。

已知必须更新的位置包括但不限于：

- `KvCacheStorage`；
- Engine option；
- CLI parser 和 `--help`；
- serving parser/schema；
- benchmark parser；
- perplexity/measurement 工具；
- request log、load/memory summary 和 JSON/CSV formatter；
- benchmark matrix、serve campaign 和其他传递 `--kv-dtype` 的通用工具；
- 对应 parser、formatter、report 和真实 Engine 路线测试；
- 输出报告字段；
- target KV profile；
- Program memory/capacity 计算；
- storage mode 的正向和反向映射；
- Main/MTP cache 构造；
- context-cache host/device replica；
- 显式 CMake source ownership。

现有 BF16、INT8 和 FP8 模式保持其当前语义，不进行无关重构。

# 十、文档

更新受影响的活动文档，包括：

- attention Op contract；
- KV append Op contract；
- `docs/maintainer/paged-kv-cache.md`；
- 相关 model/runtime 文档；
- `docs/cli.md`；
- `docs/serving.md`；
- `bench/README.md`；
- README 或 model card 中实际公开 cache 模式的位置。

文档必须明确说明：

- `nvfp4` 与 `k8v4` 的准确含义；
- Q/K/V 的格式；
- Hadamard 应用于 K/Q/V；
- V 输出需要 inverse Hadamard；
- QK accumulator 为 FP32；
- P 不进行 FP8/FP4 量化；
- P/V 使用 FP16 Tensor Core、FP32 accumulator；
- 两个模式的每 token/head cache 字节数；
- 性能结果以完整公开 Op 为准，而不是理论 Tensor Core 峰值。

不要保留完成后的 roadmap、临时设计文档或 superseded path。

# 十一、执行顺序

按两个独立的 vertical transaction 完成：

## 阶段一：`nvfp4`

1. codec 和 profile；
2. paged-cache layout；
3. standalone/fused append；
4. prompt 和 small-T attention；
5. independent tests；
6. benchmark；
7. 性能选型和优化；
8. 产品及文档接入；
9. 完整验收。

## 阶段二：`k8v4`

1. 非对称 profile/layout；
2. FP8 K + NVFP4 V append；
3. prompt 和 small-T attention；
4. independent tests；
5. benchmark；
6. 性能选型和优化；
7. 产品及文档接入；
8. 完整验收。

不得在第一个模式仍处于临时候选状态时，把两个模式混入同一个未定型实现。

# 十二、完成条件

使用：

    cmake --build <build-dir> -j

不得指定数值 job limit。

只有满足以下条件才算完成：

- `nvfp4` 和 `k8v4` 分别端到端可用；
- 两个模式具有独立 kernel/dispatch 和测试结果；
- standalone append 与 fused append byte-for-byte 一致；
- prompt、small-T、append-and-attend 和 cached 路线均通过；
- CUDA Graph capture/replay 通过；
- 独立 codec oracle 通过；
- 独立 FP64 attention oracle 通过；
- workspace 和状态事务正确；
- benchmark 正确计算非对称物理字节数；
- 已完成 RTX 5090、sm_120a 上的代表性完整 Op 测量；
- 已报告量化/Hadamard 开销和性能交叉点；
- 已使用同一 artifact、64K context 和完整固定语料重新运行 BF16、FP8、NVFP4、K8V4
  perplexity；
- 已报告两个新模式相对 BF16/FP8 的 overall、domain-level NLL/PPL 变化，且无未解释的实质退化；
- 已完成仓库级 KV storage surface inventory，并确认不存在遗漏新模式的 active parser、formatter、
  serializer、report、help、schema、benchmark、measurement、perplexity 或 Engine integration；
- 没有已知且未解释的正确性问题；
- 没有明显可避免的额外 cache pass、全局中间量或重复加载；
- 活动文档、CLI、serving 和报告名称一致；
- 没有遗留 superseded path、临时候选或兼容 fallback。

# 十三、最终报告

最终回复只包含对结果有用的信息：

1. 两个模式最终采用的 KV 物理布局；
2. codec、Hadamard 和精度选择；
3. NVFP4 QK 使用 FP32 accumulator 的确认；
4. 两种模式的源码和 ownership 边界；
5. 关键测试命令和结果；
6. 代表性 benchmark 表；
7. 相对 BF16/FP8 的收益、回退和交叉点；
8. 同一 artifact、64K context、同一 full corpus 下四种 KV 模式的 overall/domain NLL 与
   perplexity 对比；
9. 量化、Hadamard、反量化等主要性能成本；
10. 使用的硬件、CUDA/toolchain 和 workload；
11. 无法运行的检查或仍存在的实质限制。

不要输出原始日志、完整 profiling inventory、临时候选流水账或无关的仓库审计结果。
