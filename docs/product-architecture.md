# Radeon Kernels 产品架构

状态：MILESTONE 6.4 IN PROGRESS  
日期：2026-09-17  
最近更新：2026-09-18，Milestone 6.4 SDPA / Flash Attention 已固定 contract、workload、公共 API、fallback-only dispatch 与首个 online-softmax HIP 候选族

## 实施状态

| Milestone | 状态 | 当前结果 |
| --- | --- | --- |
| 1. Schema + Runtime skeleton | COMPLETE | 九份公共 schema、环境指纹、manifest registry、区域冲突检测、dispatcher、artifact 校验与 explain API 已落地 |
| 2. GEMM precompiled kernel packs | COMPLETE | gfx1151/gfx1201 私有候选包已按固定 ABI 打包，并在对应目标机中完成零编译加载和三种子正确性验证 |
| 3. GEMM static dispatch and fallback | COMPLETE | 内置精确 dispatch、pack discovery、稳定 GEMM API、预编译执行、torch fallback 和 explain 已通过两种架构验证 |
| 4. Promotion pipeline | COMPLETE | gfx1151 已完成人工审批、Ed25519 签名、公开 evidence/compatibility/Pack Index、最小 pack 发布和目标机验签执行；gfx1201 因计时不稳定被正确拦截 |
| 5. 基础 LLM 算子 | COMPLETE | Pack Installer、通用 runtime、五算子 API、80-case 双架构评估、10 个签名 release 和两机最终验收全部完成 |
| 6. GEMV / KV cache / Attention | IN PROGRESS | M6.1-M6.3 已完成；M6.4 SDPA / Flash Attention 已进入候选构建与双架构评估阶段 |
| 7. Quantization / MoE / Sampling | PLANNED | 后续阶段 |

### Milestone 5 实施清单

| Step | 状态 | 验收内容 |
| --- | --- | --- |
| M5.1 Pack Installer | COMPLETE | `rk pack list/install/verify` 已落地；先验签 index，再校验 archive 大小/SHA-256、归档路径、manifest 签名与 artifact 身份，最后原子安装且拒绝覆盖 |
| M5.2 通用算子 Runtime | COMPLETE | 非 GEMM 算子共享静态 dispatch、签名 pack 解析、预编译入口加载、确定性 fallback 和 explain；fallback-only manifest 合法且不触发搜索 |
| M5.3 五算子契约与 API | COMPLETE | 五份版本化 contract/workload、独立 FP32 reference、输入验证、稳定公共 API 和 fallback-only dispatch 已随 wheel 交付 |
| M5.4 Native 候选与双架构评估 | COMPLETE | 同一 HIP/C++ 源码分别在 gfx1151/gfx1201 构建；80 个 case 全部完成三种子正确性，65 个过 gate，15 个作为负面知识保留 |
| M5.5 Promotion 与签名发布 | COMPLETE | evidence 已去除 GEMM 硬编码；10 个代表 winner 完成 proposal、审批、签名 evidence/compatibility/Pack Index 和 release 发布 |
| M5.6 目标机验收 | COMPLETE | 最终 wheel 与每架构 5 个签名 pack 已验证全部 winner/fallback；用户侧零搜索、零 benchmark、零编译 |

Milestone 5 完成标准：

- 五个算子都有稳定、文档化的 Python API，且不向用户暴露 variant。
- 五个算子都有独立 reference、正确性测试、静态 dispatch manifest 和确定性 PyTorch fallback。
- 用户可以从签名 Pack Index 安装并验证兼容 pack，不需要手工理解 ABI 组合。
- gfx1151 与 gfx1201 都完成五算子候选评估；性能不稳定或不超过门槛的结果作为负面知识保存，不伪装成 winner。
- 通过 gate 的 winner 复用 Milestone 4 的人工审批和签名发布链，并在目标机完成零编译、零搜索复验。

Milestone 5 实施记录：

- M5.1：新增签名 Pack Installer 和 `rk pack list/install/verify`。安装器只接受已签名 Pack Index 中与本机精确兼容的 release，限制归档成员为 index 声明的 manifest、签名和 artifacts，拒绝路径穿越、链接、额外文件、篡改、错误环境和覆盖安装。5 个专项测试已通过。
- M5.2：新增共享 `OperatorRuntime` 和通用 native entrypoint 执行器；dispatch schema/runtime 允许空 entries，使尚未产生合格 winner 的算子仍可发布稳定 API 和显式 PyTorch fallback。runtime、GEMM 回归和安装器共 39 个相关测试通过。
- M5.3：固定 RMSNorm、AddRMSNorm、split-half RoPE、SwiGLU (`silu_mul`) 和最后一维 Softmax 的 1.0 语义；新增五份 contract、五份 workload、独立 FP32 reference、公共 API、输入验证和 fallback-only dispatch。完整本地测试 91 项通过，构建 wheel 已确认包含 contract、dispatch 与 API。
- M5.4：同一候选源码在目标镜像内分别编译为 gfx1201/gfx1151 预编译扩展，执行 5 算子 × 4 规模 × 2 dtype × 2 架构矩阵。80 个 case 均完成三种子正确性；gfx1201 32/40、gfx1151 33/40 同时通过 CV 和 3% 提升门槛。其余 15 项保留完整 raw samples 作为负面知识；两架构的 `rows=32,width=65536` Softmax 明确慢于 PyTorch，不进入 dispatch。
- M5.5：evidence schema 的 oracle、baseline 和 workload 已泛化且保持旧 GEMM evidence 兼容。从每个算子/架构的稳定结果中确定一个最大提升代表 case，生成 10 个候选 pack；10 份 proposal 的 95% bootstrap 区间下界均超过 3%，随后完成 `reviewer=zmu` 审批、Ed25519 签名和公开发布。所有 34 份 M4/M5 公开文档均通过 schema 与签名校验。
- M5.6：最终 wheel SHA-256 为 `8b64d308b2a1ff555db03896732b9d2c5c3aaceefd5671b939c2dbbe3c82275d`。gfx1201 与 gfx1151 分别从 wheel 内签名 Pack Index 安装 5 个 release，五个精确 winner 与五个未覆盖 shape fallback 均通过数值和 dispatch 验证；两端均记录 `compiled_during_verification=false`、`searched_during_verification=false`。私有验收记录 SHA-256 分别为 gfx1201 `5ca4007170060a76242b06573b875d9a49bac8b569a1db393819063230a3c312`、gfx1151 `23143f1eba7f43fdb9deefdf812a3b07775347fcf8421290bcfe5009954d374a`。

Milestone 1-5 已落地的核心文件：

```text
schemas/operator-contract.schema.json
schemas/dispatch.schema.json
schemas/compatibility.schema.json
schemas/kernel-pack.schema.json
schemas/evidence.schema.json
schemas/approval.schema.json
schemas/pack-index.schema.json
schemas/signature.schema.json
schemas/trusted-keys.schema.json
schemas/__init__.py

dispatch/gemm-1.0.json
dispatch/rms_norm-1.0.json
dispatch/add_rms_norm-1.0.json
dispatch/rope-1.0.json
dispatch/swiglu-1.0.json
dispatch/softmax-1.0.json
dispatch/__init__.py

contracts/rms_norm-1.0.json
contracts/add_rms_norm-1.0.json
contracts/rope-1.0.json
contracts/swiglu-1.0.json
contracts/softmax-1.0.json

src/radeon_kernels/runtime/fingerprint.py
src/radeon_kernels/runtime/compatibility.py
src/radeon_kernels/runtime/registry.py
src/radeon_kernels/runtime/dispatcher.py
src/radeon_kernels/runtime/loader.py
src/radeon_kernels/runtime/explain.py
src/radeon_kernels/runtime/pack.py
src/radeon_kernels/runtime/packs.py
src/radeon_kernels/runtime/resources.py
src/radeon_kernels/runtime/signing.py
src/radeon_kernels/runtime/installer.py

src/radeon_kernels/providers/native.py
src/radeon_kernels/providers/torch_fallback.py
src/radeon_kernels/ops/gemm/api.py
src/radeon_kernels/ops/gemv/
src/radeon_kernels/ops/_runtime.py
src/radeon_kernels/ops/rms_norm/
src/radeon_kernels/ops/add_rms_norm/
src/radeon_kernels/ops/rope/
src/radeon_kernels/ops/swiglu/
src/radeon_kernels/ops/softmax/

dispatch/README.md
compatibility/README.md
pack-index/README.md
evidence/README.md
approvals/README.md
keys/trusted-keys.json
tests/test_runtime.py
tests/test_kernel_pack.py
tests/test_gemm_dispatch.py
tests/test_signing.py
tests/test_promotion.py

developer/promotion/build_kernel_pack.py
developer/promotion/verify_kernel_pack.py
developer/promotion/verify_runtime_dispatch.py
developer/promotion/create_proposal.py
developer/promotion/record_approval.py
developer/promotion/publish_approved.py
developer/promotion/signing.py
developer/promotion/prepare_llm_candidates.py
developer/promotion/prepare_operator_candidates.py
developer/promotion/prepare_operator_replacement.py
developer/promotion/create_operator_proposal.py
developer/promotion/verify_llm_release.py
developer/promotion/verify_gemv_release.py
benchmarks/llm_ops/benchmark_softmax_online.py
benchmarks/llm_ops/native_llm_ops.hip
benchmarks/gemv/native_gemv.hip
benchmarks/gemv/benchmark.py
```

当前实现保证：

- Runtime 导入不会主动导入 PyTorch，也不会触发 GPU 检测或编译。
- Dispatch manifest 拒绝未知字段、未知 schema major/minor 和重复 entry ID。
- 同优先级重叠 region 被拒绝。
- 高优先级 region 只有在严格包含于低优先级 region 时才允许覆盖。
- 没有匹配项时只返回 manifest 声明的 fallback，不触发用户侧搜索。
- 预编译 artifact 必须位于 pack 根目录内并通过 SHA-256 校验。
- Python extension winner 必须同时匹配 Python ABI 和精确 PyTorch build。
- 公共 wheel 强制包含九份 schema、可信公钥和已发布的 approval、compatibility、evidence、pack index；runtime 导入仍不引入开发端依赖。
- 九份 schema 可通过标准 Python resource API 从构建后的 wheel 读取。
- 公共 `gemm(a, b, out=None)` 不接受 variant；variant 只来自已审核 dispatch 的静态 launch 参数。
- Pack discovery 依次检查 `RADEON_KERNELS_PACK_PATH`、用户数据目录和 Python 环境数据目录。
- Pack 必须同时匹配环境、算子语义版本、evidence ID、artifact 路径、入口、格式和 SHA-256。
- 精确 winner 缺失、损坏或无法加载时执行 manifest 声明的 `torch.mm` fallback，并记录实际执行原因。
- 输入启用 autograd、shape 不匹配或不连续时不会误用当前 inference-only winner。
- Runtime 默认拒绝无签名、签名错误、哈希变化、未知密钥或已吊销密钥签出的 kernel pack。
- 签名覆盖规范化文件名和 SHA-256，防止把一个有效签名挪给另一个发布文件。
- 人工 approval 绑定 proposal ID 与 proposal 文件的精确 SHA-256；proposal 在审批后发生任何变化都会阻止发布。
- 发布归档只包含 manifest、manifest 签名和 manifest 引用的 artifact，不包含私有验证日志。
- `rk pack list/install/verify` 只接受签名 index 中精确匹配环境的 release，安装过程拒绝不安全归档、篡改与覆盖。
- RMSNorm、AddRMSNorm、RoPE、SwiGLU 和 Softmax 公共 API 不暴露 variant；未匹配调用执行 manifest 声明的 PyTorch FP32 reference fallback。
- 当前测试总数为 112，全部通过。

Milestone 2 已生成的私有候选包：

| Architecture | Pack | 固定环境 | Workload | 验证结果 |
| --- | --- | --- | --- | --- |
| gfx1151 / wave32 | `rk-gemm-gfx1151-rocm715-cp312-torch2140a0` | ROCm ABI 7.15、CPython 3.12、PyTorch `2.14.0a0+rocm7.15.0a20260719` | FP16 256x4096x4096、variant 8 | 三种子通过；最大绝对误差 0.0009765625；验证期零编译 |
| gfx1201 / wave32 | `rk-gemm-gfx1201-rocm724-cp312-torch2110` | ROCm ABI 7.2、CPython 3.12、PyTorch `2.11.0+rocm7.2.4.git5fbd98f3` | FP16 512x512x512、variant 9 | 三种子通过；最大绝对误差 0.000244140625；验证期零编译 |

候选包和验证凭据保存在 `lab-private/exports/kernel-packs/`，不进入公共 Git。构建器要求源 artifact 的预期 SHA-256 完全一致、拒绝覆盖已有 pack，并在原子移动前重新解析 manifest 和校验复制后的二进制。目标机验证器只加载 wheel 与预编译扩展，不调用编译或搜索路径。

Milestone 3 目标机端到端验证：

| Architecture | 公共 API winner | 未覆盖 shape | 结果 |
| --- | --- | --- | --- |
| gfx1151 | `rk.gemm` 自动选择 `native/gemm_out`、variant 8 | 64x64x64 自动选择 `torch/mm` | 两条路径均通过；零搜索、零编译 |
| gfx1201 | `rk.gemm` 自动选择 `native/gemm_out`、variant 9 | 64x64x64 自动选择 `torch/mm` | 两条路径均通过；零搜索、零编译 |

用户可将单个 pack 目录或包含多个 pack 的父目录写入 `RADEON_KERNELS_PACK_PATH`。默认还会检查 `~/.local/share/radeon-kernels/packs/` 和 `<python-prefix>/share/radeon-kernels/packs/`。运行时在首次调用时加载这些目录；安装新 pack 后可调用 `radeon_kernels.ops.gemm.reset_runtime()` 刷新。

Milestone 5 公开代表 winner：

| Architecture | RMSNorm | AddRMSNorm | RoPE | SwiGLU | Softmax |
| --- | --- | --- | --- | --- | --- |
| gfx1201 | 128×4096 FP16，87.84% | 128×4096 FP16，88.69% | decode d128 BF16，92.60% | 1024×11008 FP16，90.34% | 128×4096 BF16，77.60% |
| gfx1151 | 128×4096 FP16，84.24% | 128×4096 FP16，79.02% | decode d128 FP16，88.89% | 512×28672 FP16，87.59% | 512×8192 FP16，74.73% |

表中提升均相对各算子的 PyTorch FP32 reference，且仅对 dispatch 中精确环境与 workload 生效；其他 shape、dtype、autograd 或非连续输入确定性 fallback。完整 80-case raw samples、15 项负面知识和机器配置只保存在 `lab-private/`。

### Online Softmax 演进（等待批准）

Softmax 候选已从传统三遍实现演进为数值稳定的 Online Softmax。每个线程在读取输入时维护 `(running_max, running_denominator)`，工作组用相同的 associative combine 合并局部统计，然后执行一次归一化输出；因此普通行从三次输入遍历降为两次。`width >= 16384` 时使用 `4096` 元素分块，先并行生成分块统计，再合并行统计并并行写回，避免超宽行只有一个工作组。

评估在同一进程交错测量 Online 候选、已发布三遍内核和 PyTorch FP32 reference。晋级除原有正确性、候选/基线 CV 和相对 PyTorch 至少 3% 的 gate 外，还要求旧三遍内核 CV 不超过 3%，且 Online 候选至少再快 3%。两次独立进程复测结果如下：

| Architecture | Dispatch workload | Run | Online | 三遍内核 | PyTorch | 相对三遍提升 | 决策 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| gfx1201 | 128×4096 BF16 | 首测 | 0.009587 ms | 0.010228 ms | 0.045580 ms | 6.27% | promote |
| gfx1201 | 128×4096 BF16 | 复测 | 0.009610 ms | 0.010205 ms | 0.045657 ms | 5.83% | promote |
| gfx1151 | 512×8192 FP16 | 首测 | 0.061320 ms | 0.076631 ms | 0.319011 ms | 19.98% | promote |
| gfx1151 | 512×8192 FP16 | 复测 | 0.062533 ms | 0.076526 ms | 0.316100 ms | 18.29% | promote |

超宽 `rows=32,width=65536` 的结论按架构保留：gfx1201 Online 已比三遍内核快约 56%，但仍比 PyTorch 慢 3%–5%，继续 fallback；gfx1151 Online 比三遍内核快约 67%、中位数比 PyTorch 快约 30%–36%，但两次 PyTorch baseline CV 都超过 3%，暂不晋级。这些结果证明 Online 算法显著修复了旧实现的超宽行并行度，但不会绕过稳定性门槛。

两架构替换 pack、候选 dispatch 和 promotion proposal 已生成，proposal 的全部自动 gate 均通过。公开 `dispatch/softmax-1.0.json`、签名 evidence 和 Pack Index 在人工批准前保持不变；批准后才执行签名发布、原子替换 dispatch 和双机零编译验收。候选 artifact SHA-256 为 gfx1201 `5869f5581ea6463117d653e82ba3a34896143c2a9b3844f0faca0915c3b197e3`、gfx1151 `21d8b7ff5503d72e96ec17ab784c5a6df1a7531d867ff21042b57f2c19c6bc43`。

当前明确边界与后续内容：

- gfx1201 外部复测的 candidate CV 为 24.55%、`torch.mm` CV 为 18.04%，不满足 3% 稳定性上限，必须重测，当前不得正常晋级。
- 原有 `ops.gemm.native` JIT 路径仅保留为开发者工具，不再是公共 `gemm` 默认路径。
- 当前正式 fallback adapter 只有 PyTorch；hipBLASLt、CK 和 AITER adapter 尚未发布。
- HSACO 已声明为 artifact 类型，但 provider-specific launcher 尚未实现。

## 1. 项目定义

`radeon-kernels` 是一个面向 Radeon GPU 的 LLM 算子知识发布与执行系统。

项目开发者在私有多机实验室中完成候选实现的搜索、编译、正确性验证、性能测试和晋级；普通用户只安装经过验证的运行时、dispatch 数据和预编译 kernel，直接复用开发者已经获得的知识。

项目遵循以下原则：

- 开发者负责昂贵的搜索，用户只负责查询和执行。
- 用户侧不运行 autotune、benchmark 或 AI 搜索。
- 用户侧默认不进行 JIT 编译，不要求安装编译器、Triton、Docker 或实验环境。
- 已覆盖的环境和 workload 直接加载预编译 winner。
- 未覆盖或不兼容的环境使用确定性 fallback，不在用户机器上临时尝试候选。
- 私有机器配置和原始实验数据不进入公共仓库。
- 所有性能结论都必须绑定算子语义、硬件、软件环境、workload 和证据。

项目的核心资产不是某一个 kernel，而是：

```text
算子语义
+ 架构和软件兼容性知识
+ 最佳实现及其适用范围
+ 预编译二进制
+ 正确性与性能证据
+ 确定性 fallback
```

## 2. 产品边界

### 2.1 开发者侧

开发者侧负责：

- 维护私有目标机器和执行环境。
- 收集真实 LLM workload 的 shape 分布。
- 搜索 Native HIP、AMDGPU ASM、Triton 和外部 provider 实现。
- 使用独立 reference 执行正确性验证。
- 保存完整 timing samples、静态信息和环境身份。
- 通过确定性 gate 决定候选是否晋级。
- 对晋级结果进行脱敏、审核、签名和发布。
- 保存失败实验和负面知识，避免未来重复搜索。

### 2.2 用户侧

用户侧只负责：

- 安装 runtime 和与本机兼容的 kernel pack。
- 调用稳定的公共算子 API。
- 由 runtime 识别硬件与 ROCm 环境。
- 查找 dispatch manifest 并加载 winner。
- 在不匹配时走已经声明的 fallback。
- 使用 explain API 查看选择原因。

用户侧明确不执行：

- 候选生成。
- 性能搜索。
- 首次运行校准。
- 自动 benchmark。
- AI 或远程服务调用。
- 动态下载并执行未经签名的代码。

## 3. 总体架构

```text
                    开发者侧
+------------------------------------------------+
| lab-private                                    |
|                                                |
| targets.yaml -> 多机执行 -> 候选搜索           |
|                       |                        |
|         correctness / performance / stability |
|                       |                        |
|                verified lineage                |
|                       |                        |
|                promotion proposal              |
+-----------------------+------------------------+
                        | 人工审核、脱敏、签名
                        v
                   公共发布层
+------------------------------------------------+
| Operator Contracts                             |
| Dispatch Database                              |
| Compatibility Database                         |
| Precompiled Kernel Packs                       |
| Sanitized Evidence                             |
+-----------------------+------------------------+
                        v
                      用户侧
+------------------------------------------------+
| rk.ops.*                                       |
|      |                                         |
| runtime fingerprint                            |
|      |                                         |
| manifest lookup                                |
|      |                                         |
| load precompiled kernel / official fallback    |
+------------------------------------------------+
```

## 4. 公共仓库结构

```text
radeon-kernels/
|-- src/radeon_kernels/
|   |-- ops/
|   |   |-- gemm/
|   |   |-- rms_norm/
|   |   |-- rope/
|   |   |-- swiglu/
|   |   `-- softmax/
|   |
|   |-- runtime/
|   |   |-- fingerprint.py
|   |   |-- registry.py
|   |   |-- dispatcher.py
|   |   |-- loader.py
|   |   |-- compatibility.py
|   |   `-- explain.py
|   |
|   |-- providers/
|   |   |-- native.py
|   |   |-- triton.py
|   |   |-- hipblaslt.py
|   |   |-- ck.py
|   |   |-- aiter.py
|   |   |-- aotriton.py
|   |   `-- torch_fallback.py
|   |
|   `-- integrations/
|       `-- pytorch/
|
|-- developer/
|   |-- benchmark/
|   |-- search/
|   |-- asmevo/
|   |-- promotion/
|   `-- evidence/
|
|-- schemas/
|   |-- operator-contract.schema.json
|   |-- dispatch.schema.json
|   |-- compatibility.schema.json
|   |-- kernel-pack.schema.json
|   `-- evidence.schema.json
|
|-- dispatch/
|-- compatibility/
|-- pack-index/
|-- evidence/
|-- benchmarks/workloads/
|-- lab/
|-- tests/
`-- docs/
```

只有 runtime、公共算子契约、dispatch、compatibility、pack index 和脱敏 evidence 属于用户产品面。搜索和 benchmark 工具属于开发者工具面，不得成为 runtime 的必需依赖。

## 5. 私有工作区

```text
radeon-kernel-workspace/
|-- .radeon-workspace.yaml
|-- radeon-kernels/
`-- lab-private/
    |-- config/
    |   `-- targets.yaml
    |-- state/
    |-- runs/
    |-- cache/
    |-- logs/
    `-- exports/
```

私有工作区保存：

- SSH alias 和 host fingerprint。
- 机器工作目录、GPU 分配和容器镜像。
- 原始 timing samples 和 telemetry。
- 候选源码、编译日志、反汇编和 code object。
- AsmEvo lineage、失败实验和未晋级结果。
- 尚未经过审核和脱敏的 promotion proposal。

公共仓库不得包含真实主机名、IP、SSH 信息、远程路径、私有 registry 地址或原始机器数据。

## 6. 算子契约

每个算子必须先定义版本化语义契约，再添加实现。

算子契约至少包含：

- `operator_id` 和 `semantic_version`。
- 输入、输出和可选参数。
- dtype、accumulation dtype 和输出 dtype。
- shape、layout、stride 和 alignment 约束。
- 输入输出 alias 规则。
- workspace 要求。
- forward、backward 和 inference-only 能力。
- deterministic 和 HIP Graph capture 能力。
- NaN、Inf、subnormal 和 signed zero 语义。
- reference implementation。
- 正确性 seeds、输入分布和误差阈值。

目录示例：

```text
ops/rms_norm/
|-- api.py
|-- spec.py
|-- reference.py
|-- workloads.yaml
|-- candidates/
|   |-- torch.py
|   |-- triton.py
|   |-- ck.py
|   |-- aiter.py
|   |-- native/
|   `-- asm/
`-- tests/
```

## 7. Provider 模型

Provider 表示一种 kernel 来源。官方实现和自研实现都作为候选参与统一验证，而不是被特殊处理。

首批 provider 包括：

- Native HIP/C++。
- 独立 AMDGPU ASM 或 HSACO。
- Triton。
- hipBLASLt 和 rocBLAS。
- Composable Kernel。
- AITER。
- AOTriton。
- PyTorch ROCm fallback。

每个 provider adapter 负责：

- 声明自身版本、ABI 和架构兼容性。
- 将统一算子契约转换为 provider 调用。
- 报告 graph capture、determinism 和 workspace 能力。
- 暴露稳定的 benchmark 入口。
- 对不支持的 workload 返回结构化原因。

## 8. Dispatch Key

Dispatch 不能只按算子名或 GPU 名称选择。完整 key 至少包含：

```text
operator semantic version
+ architecture / wave size / architecture features
+ ROCm ABI / compiler compatibility
+ Python ABI / PyTorch build compatibility
+ provider and provider version
+ dtype / accumulation dtype / quantization scheme
+ layout / stride / alignment
+ bounded shape region
+ functional flags
+ deterministic requirement
+ graph capture requirement
```

不同算子还需要额外维度：

- Attention：causal、head dimension、Hq/Hkv、sequence length、page size、KV dtype。
- Norm：epsilon、residual fusion、hidden dimension。
- GEMM：M/N/K、transpose、batch mode、epilogue、quantization。
- MoE：expert count、top-k、token distribution 和 routing layout。

Dispatch manifest 使用不重叠的 shape region、guard predicate 和显式 priority。找不到匹配项时必须进入 fallback，不允许在用户环境触发搜索。

## 9. Dispatch Record

以下是已发布 `gfx1151` entry 的核心字段；完整记录位于 `dispatch/gemm-1.0.json`：

```json
{
  "operator": "gemm",
  "semantic_version": "1.0",
  "priority": 100,
  "environment": {
    "architecture": "gfx1151",
    "rocm_abi": "7.15",
    "wavefront_size": 32,
    "python_abi": "cp312",
    "pytorch_version": "2.14.0a0+rocm7.15.0a20260719"
  },
  "workload": {
    "dtype": "fp16",
    "layout": "nn",
    "m": [256, 256],
    "n": [4096, 4096],
    "k": [4096, 4096],
    "contiguous": true,
    "requires_grad": false
  },
  "winner": {
    "provider": "native",
    "artifact": "lib/radeon_kernels_wmma_gfx1151.so",
    "entrypoint": "gemm_out",
    "sha256": "c8fb5969147b2e47dd8857e547a5fbd2a9d166bb58b1ad0871aa78ffcc8c28b6"
  },
  "launch": {
    "variant": 8
  },
  "fallbacks": [
    "torch"
  ],
  "evidence_id": "asmevo-gemm-20260917t050538z-gfx1151-k0"
}
```

公开知识既包括 winner，也包括：

- 已验证的备选实现。
- 不支持的环境和 workload。
- 已知错误或不稳定组合。
- provider 和 ROCm 版本限制。
- 正确性范围和未测试范围。
- 性能置信区间和测量环境。

## 10. 用户 Runtime

公共 Python API：

```python
import radeon_kernels as rk

c = rk.ops.gemm(a, b)
y = rk.ops.rms_norm(x, weight, eps=1e-6)
z, residual_out = rk.ops.add_rms_norm(x, residual, weight, eps=1e-6)
q, k = rk.ops.rope(q, k, cos, sin)
y = rk.ops.silu_mul(gate, up)
probabilities = rk.ops.softmax(logits)

print(rk.explain_last_dispatch())
```

Runtime 调用链：

```text
API input validation
-> environment fingerprint
-> operator signature
-> manifest lookup
-> compatibility and guard validation
-> precompiled artifact load
-> kernel launch
```

`rk.explain_last_dispatch()` 至少返回：

```text
operator: gemm
architecture: gfx1151
selected: native/gemm_out
artifact: lib/radeon_kernels_wmma_gfx1151.so
evidence: asmevo-gemm-20260917t050538z-gfx1151-k0
fallbacks: torch
reason: matched exact workload-region and loaded compatible pack
```

Runtime 必须保持纯查表行为。它不测量候选、不改变 winner，也不向外发送遥测。

## 11. Fallback

每条 dispatch entry 必须声明有序 fallback。例如：

```text
预编译自研 winner
-> 已验证的 AITER / CK / hipBLASLt 实现
-> PyTorch ROCm reference
```

以下情况必须 fallback：

- 没有匹配的 architecture 或 ROCm ABI。
- kernel pack 缺失、哈希错误或签名无效。
- shape、stride、alignment 或功能参数不满足 guard。
- 用户要求 deterministic 或 graph capture，但 winner 不支持。
- provider 初始化失败。

Fallback 必须可解释，不能静默执行用户侧搜索。

## 12. 发布产物

建议将用户产物拆分为：

```text
radeon-kernels-runtime
radeon-kernels-pack-gfx1151-rocm715
radeon-kernels-pack-gfx1201-rocm724
```

Runtime 包含：

- 公共 API。
- environment fingerprint。
- dispatcher、registry、loader 和 explain。
- manifest schema validator。
- 官方 provider fallback adapter。

Kernel pack 包含：

```text
lib/
|-- rk_gemm.so
|-- rk_rms_norm.so
`-- rk_rope.so

manifest.json
compatibility.json
SHA256SUMS
```

Kernel pack 可以通过 wheel 或 Release artifact 发布。安装完成后，正常执行不得依赖网络。

当前开发阶段不要求每轮实验或 promotion 都重新构建 wheel。日常目标机验收直接同步 `src/` 与公开的 dispatch、Pack Index 和可信公钥，release pack 仍通过签名索引安装；wheel 只在里程碑封版、公开版本或需要验证最终分发物时构建。源码模式和 wheel 模式必须使用同一套 runtime、签名校验与预编译 pack，均不得在用户路径触发编译或搜索。

## 13. 搜索去重与知识复用

开发者侧也必须避免重复消耗算力和 AI token。实验缓存的逻辑唯一键为：

```text
operator contract hash
+ candidate source/artifact hash
+ architecture
+ environment/toolchain hash
+ workload hash
+ measurement policy hash
```

组合已经存在有效证据时，不再重复执行。实验数据库永久保留：

- 编译失败。
- 正确性失败。
- 正确但更慢。
- 性能不稳定。
- ABI 或架构不兼容。
- 已接受候选和 verified lineage。
- 优化假设、修改窗口和反汇编证据。
- 已经耗尽的搜索方向。

新一轮搜索必须从 verified node 和历史负面知识开始，不能重新探索相同候选。

## 14. 下一批算子

GEMM 产品化完成后，首批新增算子为：

| 优先级 | 算子 | 首批 workload |
| --- | --- | --- |
| P0 | RMSNorm | tokens 1-4096，hidden 512-16384，FP16/BF16 |
| P0 | Add + RMSNorm | 同 RMSNorm，增加 residual fusion |
| P0 | RoPE | head dimension 64/80/96/128/256，decode/prefill |
| P0 | SiLU x Gate / SwiGLU | tokens 1-4096，hidden 512-32768 |
| P0 | Softmax | rows 1-8192，width 32-65536 |
| P1 | GEMV / skinny GEMM | M=1-32，大 N/K，LLM decode |
| P1 | KV-cache operations | append、copy、paged layout、quantize |
| P1 | Paged Attention Decode | batch、Hq/Hkv、context、page size |
| P1 | SDPA / Flash Attention | causal/noncausal、GQA、head dimension |
| P2 | Quantization | INT8/FP8，per-token/per-channel |
| P2 | MoE | top-k、routing、grouped GEMM |
| P2 | Sampling | logits softmax、top-k/top-p、argmax |

### 固定实施顺序

Milestone 6 及后续算子按以下顺序实施：

1. **GEMV / Skinny GEMM**
2. **KV-cache append/copy/paged-layout**
3. **Paged Attention Decode**
4. **SDPA / Flash Attention**
5. 再进入 **Quantization、MoE 和 Sampling**

该顺序是项目依赖约束，而不只是优先级建议：先建立 decode 线性层和 KV-cache 数据移动的可靠基线，再评估 Paged Attention 与完整 SDPA，避免把底层 GEMV 或 KV layout 的瓶颈错误归因给 Attention。除非文档经过明确评审和更新，后续 milestone 不跨过尚未完成的前置阶段。

### Milestone 6.1 实施清单

| Step | 状态 | 验收内容 |
| --- | --- | --- |
| M6.1.1 Contract 与 workload | COMPLETE | 固定 `x[M,K] @ weight[N,K].T -> output[M,N]`、FP16/BF16、FP32 oracle 和 10 个 decode 代表规模 |
| M6.1.2 Reference、API 与 fallback | COMPLETE | `gemv(x, weight)` 不暴露 variant；两台目标机从 wheel 验证 fallback-only dispatch 确定性执行 `torch.mm`，零编译、零搜索 |
| M6.1.3 Native 候选 | COMPLETE | wave32 FP32 dot-product，开发侧比较每 wave 计算 1/2/4 个输出的三个 entrypoint |
| M6.1.4 双架构 benchmark | COMPLETE | 40/40 组合通过三种子正确性；20 次 warmup、30 个交错样本、CV 与 3% 提升 gate 已执行 |
| M6.1.5 候选结论 | COMPLETE | gfx1201 4/20 过 gate；gfx1151 0/20，全部负面知识保留；尚未进入 promotion |
| M6.1.6 第二轮复测与专项优化 | COMPLETE | gfx1201 仅用原三 entrypoint 独立复测，3/4 过 gate；另 1 项候选稳定但基线两次超 CV；gfx1151 九候选的 270 次三种子检查全部正确，0/10 过 gate |
| M6.1.7 Promotion preparation | COMPLETE | 三项 gfx1201 winner 已生成独立不可变 pack、私有 staged dispatch 和 proposal；全部自动 gate 与 schema 预检通过 |
| M6.1.8 Approval、发布与验收 | COMPLETE | reviewer `zmu` 批准三项 proposal；公开 dispatch、evidence、compatibility、Pack Index 和 release 均已签名发布，gfx1201 源码模式验收确认 3 个 winner 与 1 个 fallback，零编译、零搜索 |

M6.1 不实现 KV-cache、Paged Attention 或 SDPA。正确性 oracle 使用 FP32 累加 reference；性能 baseline 使用同 dtype 的 `torch.mm(x, weight.T)`，避免将 reference 转换开销计入候选收益。

M6.1 首轮结果：

| Architecture | Workload | Dtype | Native | `torch.mm` | 提升 | Entrypoint | 决策 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| gfx1201 | M1 N4096 K11008 | FP16 | 0.1532 ms | 0.2043 ms | 25.01% | `gemv_wave2` | candidate |
| gfx1201 | M1 N4096 K11008 | BF16 | 0.1534 ms | 0.2094 ms | 26.75% | `gemv_wave2` | candidate |
| gfx1201 | M1 N8192 K8192 | FP16 | 0.2928 ms | 0.3963 ms | 26.13% | `gemv_wave1` | candidate |
| gfx1201 | M1 N8192 K8192 | BF16 | 0.2904 ms | 0.3976 ms | 26.97% | `gemv_wave1` | candidate |

上述四项的 candidate/baseline CV 均不超过 3%，三种子最大绝对误差不超过 BF16 的 `0.00390625`。gfx1201 的 M1 N4096 K4096 虽有约 42% 中位数优势，但两侧 CV 都超标；M1 N28672 K8192 的 baseline CV 超标，均不得晋级。M>=2 时当前候选开始落后，M>=4 后差距快速扩大，说明独立 wave-dot 不应覆盖小批量 GEMM 区域。

gfx1151 的 20 个组合全部正确且大部分计时稳定，但候选在 M1 已比 `torch.mm` 慢 34%-155%，随着 M 增大退化至数十倍，因此本轮不产生 gfx1151 winner。该结果作为架构特定负面知识保存，下一轮 gfx1151 优化应研究向量化加载、输入/权重缓存复用或矩阵指令路径，而不是直接扩大当前 kernel 的 dispatch 范围。

公开 `dispatch/gemv-1.0.json` 仍为 fallback-only。首轮 benchmark 只证明四个 gfx1201 workload 具备进入独立复测的资格；在复测、proposal、人工批准和签名发布完成前，用户调用始终确定性执行 `torch.mm` fallback。

M6.1 第二轮执行边界固定如下：

- gfx1201 使用首轮源码中的 `gemv_wave1/2/4` 三个 entrypoint，对 M1 N4096 K11008 和 M1 N8192 K8192 的 FP16/BF16 四个候选进行独立复测；新增候选不得参与选择，以免改变复测对象。
- gfx1151 只覆盖五个 M1 decode 规模，同时比较原始 wave-dot、32/64-bit packed load 和 block 级 LDS 输入复用，共九个 entrypoint。
- packed 候选仅在 `K` 保持每一权重行的向量对齐时使用向量读取，其他 `K` 自动走标量路径；不得扩大公开 API 的输入假设。
- 本轮不测试 M>=2，不实现 KV-cache、Paged Attention 或 SDPA。结果必须通过三种子正确性、candidate/baseline CV 不超过 3% 和至少 3% 中位延迟改善，才可进入 proposal。

M6.1 第二轮 gfx1201 有效复测结果：

| Workload | Dtype | Entrypoint | Native | `torch.mm` | 提升 | CV candidate/baseline | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| M1 N4096 K11008 | FP16 | `gemv_wave2` | 0.1433 ms | 0.1690 ms | 15.20% | 0.55% / 3.95% | baseline 连续失稳，不晋级 |
| M1 N4096 K11008 | BF16 | `gemv_wave2` | 0.1416 ms | 0.1691 ms | 16.24% | 0.59% / 0.66% | 通过 |
| M1 N8192 K8192 | FP16 | `gemv_wave1` | 0.2504 ms | 0.3275 ms | 23.52% | 0.14% / 0.46% | 通过 |
| M1 N8192 K8192 | BF16 | `gemv_wave1` | 0.2511 ms | 0.3286 ms | 23.57% | 0.13% / 0.55% | 通过 |

首次复测所在 GPU 出现系统级抖动，四项 candidate CV 为 18.82%-23.13%，该证据完整保留但不参与晋级。换至空闲 GPU 后三项完整过 gate。M1 N4096 K11008 FP16 的 candidate 在两次有效复测中保持约 0.1432 ms 和不超过 0.56% CV，但 `torch.mm` baseline CV 分别为 5.02% 和 3.95%，因此即使中位数收益超过 15%，也不得绕过 baseline stability gate。

gfx1151 专项搜索中，`gemv_packed4_wave1` 成为全部十个 dtype/workload 组合的最快候选；相对原始 wave-dot 最快实现提升约 21%-50%，证明 64-bit packed load 是有效方向。它在 M1 N4096 K11008 FP16 达到 0.3944 ms、比 `torch.mm` 快 0.95%，在 M1 N28672 K8192 BF16 达到 2.3760 ms、快 2.29%，仍未达到 3% 门槛；其余组合落后 0.49%-34.95%。LDS 输入复用没有成为任何组合的 winner，应作为负面知识停止继续扩展。九个 entrypoint、十个组合、三种子的 270 次检查全部正确，最大绝对误差为 `0.00390625`。

第二轮源码 SHA-256 为 `45299ccc2f9fd5b0953e584635673278808a4f1f52b7d992ba81eee3a7f9ba7f`；gfx1201 和 gfx1151 artifact SHA-256 分别为 `768491638fc4de0ba7827e1841de9ab918bdec5107a98778c9d13a67db789af2` 与 `17f77a999d6d2764fc58b6d48288945155f0452bcb52cc2d2acdaaa98b82133c`。公开 dispatch 继续保持 fallback-only；仅三项 gfx1201 结果具备生成 promotion proposal 的资格，未获人工批准前不得发布。

M6.1 promotion preparation 已为同一 GEMV 算子的多个精确 workload 增加通用 staging 工具。它只选择 `decision=promote` 且所有 worker gate 为真的结果，为每项建立独立 pack，并将所有 entry 写入私有 staged dispatch；公开 `dispatch/gemv-1.0.json` 不在此阶段修改。三份 proposal 结果如下：

| Workload | Dtype | Proposal | 提升 | 95% bootstrap CI | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| M1 N4096 K11008 | BF16 | `proposal-m6-20260918-gemv-gfx1201-decode-ffn-down-h4096-bf16-381585303463e2fd` | 16.24% | [15.95%, 16.59%] | 已批准并发布 |
| M1 N8192 K8192 | BF16 | `proposal-m6-20260918-gemv-gfx1201-decode-h8192-bf16-a7431d842ab97ac0` | 23.57% | [23.42%, 23.71%] | 已批准并发布 |
| M1 N8192 K8192 | FP16 | `proposal-m6-20260918-gemv-gfx1201-decode-h8192-fp16-0e7e40c9f0a239e6` | 23.52% | [23.45%, 23.75%] | 已批准并发布 |

三份 proposal 的 artifact、环境、workload、三种子正确性、candidate/baseline CV、3% 中位改进、95% bootstrap 下界和 worker gate 全部通过。proposal evidence 已正确记录 oracle 为 `torch_fp32_reference`、计时 baseline 为 `torch_mm`。reviewer `zmu` 的三份 approval 均与 proposal 的精确 SHA-256 绑定，未使用 gate override。公开 dispatch 已原子更新为 staged 内容，SHA-256 为 `88ebed631076d10df754f8c997638e0a2acfd472648e8765dada21d0b78fa784`。

三份 release archive 的 SHA-256 分别为 `adb37f097707f722b8e68c1d095a18215245b2a370106553ae6569042c85b369`、`948d0965948e63d09df9cb2f5256052a687cd791330f977b550d36c3b2d6f647` 和 `f751f27d36cfea49490ffa64747d27a7e6c567b6c5d7b6eadca8b82c4d386bbb`。所有公开文档签名与 archive manifest 签名均通过验证。

gfx1201 最终验收采用源码同步模式，没有为本轮重新构建或安装 wheel。固定镜像从签名 Pack Index 安装三个 release，三个精确 workload 分别命中 `native/gemv_wave2`、`native/gemv_wave1`、`native/gemv_wave1`，未覆盖的 M2 N64 K128 FP16 调用回退到 `torch/mm`。最大绝对误差依次为 `0.001953125`、`0.001953125`、`0.00048828125` 和 `0`；记录明确为 `compiled_during_verification=false`、`searched_during_verification=false`。私有验收记录 SHA-256 为 `e75163f93d9b27dd92d4359c8cf632f12e1666c345a6cbcd52d1d47dad903243`。

M6.1 wheel SHA-256 为 `57fff4d3a17bbee82d20a89023c354492720084fdce6c7b137ae9b28e8228676`。该 wheel 已在 gfx1201/ROCm 7.2 和 gfx1151/ROCm 7.15 固定镜像中分别验证 FP16、BF16 fallback，四次调用的实际选择均为 `torch/mm`，相对 FP32 oracle 的最大绝对误差为 0，并记录 `compiled_during_verification=false`、`searched_during_verification=false`。

### Milestone 6.2 实施清单

| Step | 状态 | 验收内容 |
| --- | --- | --- |
| M6.2.1 Contract 与 workload | COMPLETE | 固定 paged layout 为 `[num_blocks, block_size, num_kv_heads, head_dim]`；定义物理 slot append、源/目标 block copy 和 22 个双 dtype 代表组合 |
| M6.2.2 Reference、API 与 fallback | COMPLETE | 新增 `kv_cache_append`、`kv_cache_copy`、PyTorch `index_copy_` reference 和初始 fallback-only dispatch；公共 API 不暴露 variant |
| M6.2.3 Native 候选 | COMPLETE | append/copy 各实现 scalar、64-bit vec4、128-bit vec8 三种搬运入口；边界检查留在设备端 |
| M6.2.4 双架构 benchmark | COMPLETE | 44/44 组合通过三种子逐元素正确性；gfx1151 22/22 过 gate，gfx1201 空闲 GPU 复测 11/22 过 gate，其余结果作为负面知识保留 |
| M6.2.5 Promotion preparation | COMPLETE | 33 个精确 winner 已生成不可变 pack、私有 staged dispatch 和 proposal；全部自动 gate 与 bootstrap 置信区间通过，并已进入 M6.2.6 审批链 |
| M6.2.6 Approval、发布与验收 | COMPLETE | reviewer `zmu` 批准 33 项 proposal；两个公开 dispatch、33 份 evidence/approval/release 和 Pack Index 已签名发布，两架构源码模式验收全部通过，零编译、零搜索 |

`kv_cache_append(key, value, key_cache, value_cache, slot_mapping)` 将 `[tokens, kv_heads, head_dim]` 写入展平的物理 page slot，其中 `block = slot // block_size`、`offset = slot % block_size`。有效 slot 必须位于 cache 容量内且彼此唯一。

`kv_cache_copy(key_cache, value_cache, block_mapping)` 使用 `[pairs, 2]` 的 source/destination block 映射复制完整 K/V page。目标 block 必须唯一，并与所有 source block 不相交；该约束保证并行实现与源快照语义一致。两项操作都是 inference-only 原地操作，返回的 K/V cache 与输入 cache alias，数值要求逐元素精确一致。

M6.2 不实现 Paged Attention、block table 的逻辑序列管理、cache 分配器、淘汰策略或量化 cache。这些属于后续层；本阶段只提供 Paged Attention 可直接消费的数据布局与数据移动原语。

M6.2 首轮与复测结果：

| Architecture | Operator | 过 gate/总数 | 延迟改善范围 | 最低 95% bootstrap 下界 | 最大有效 CV |
| --- | --- | ---: | ---: | ---: | ---: |
| gfx1151 | append | 12/12 | 62.38%-68.09% | 62.35% | 1.66% |
| gfx1151 | copy | 10/10 | 65.94%-88.62% | 65.81% | 2.92% |
| gfx1201 | append | 2/12 | 74.67%-74.81% | 74.64% | 1.58% |
| gfx1201 | copy | 9/10 | 80.67%-87.03% | 80.62% | 1.84% |

gfx1201 初测的 22 项 candidate/baseline CV 全部超标，因此完整保留但不参与晋级；换至空闲 GPU 后有 11 项通过全部 gate。未晋级项中，GQA 128-token append 比 PyTorch 慢 2.52%-3.18%，GQA d256 BF16 append 慢 36.69%；d256 FP16 append 和 8-pair BF16 copy 虽有正收益，但 baseline CV 分别为 3.56% 和 6.37%，不得绕过稳定性门槛。

33 份 proposal 均为 `recommended_decision=promote`。reviewer `zmu` 的 approval 全部与 proposal 精确 SHA-256 绑定，未使用 gate override。公开 dispatch 已原子更新为 staged 内容，SHA-256 分别为 append `308bc388ce859b3d76bc0589182a873ba4a66500fac7bc82b9da58f7e0b80fa7`、copy `2eaa70cb5a460353bb55aaa26b3a0499748d00736625f773ce16df1996ddf1df`。

签名发布后，append Pack Index 包含 14 个 release，SHA-256 为 `25939b21d1ac60632596e45da50f6329af98d31ae9ce7f544a89f0a30085205e`；copy Pack Index 包含 19 个 release，SHA-256 为 `47c8e3dd53748e0dde6e5275ce7392aa9c8da55ae4d823ee621561aaa9a92ac0`。33 份 signed evidence、approval、compatibility record、manifest 和 release archive 均通过发布器及目标 Pack Installer 验证。

最终验收继续采用源码同步模式，没有构建 wheel。gfx1151 从签名索引安装 22 个 release，执行 22 个 exact winner 与 2 个 fallback；gfx1201 安装 11 个 release，执行 11 个 exact winner 与 2 个 fallback。所有 K/V cache 逐元素最大绝对误差为 0，返回张量均 alias 输入 cache，两边均记录 `compiled_during_verification=false`、`searched_during_verification=false`。私有验收记录 SHA-256 分别为 gfx1151 `b2f12ade1794d9d6d478d154c75a8c7698caf0eba0c1381a773c2f2d0b57981d`、gfx1201 `498479140794682e853c6072daa5dd7c7c94660f61fd6d95e9fc3539ca22fffe`。

### Milestone 6.3 实施清单

| Step | 状态 | 验收内容 |
| --- | --- | --- |
| M6.3.1 Contract 与 workload | COMPLETE | 固定单 query decode、paged block table、MHA/GQA head 映射、FP32 累加与 page 16/32；定义 8 个代表规模、双 dtype 共 16 项组合 |
| M6.3.2 Reference、API 与 fallback | COMPLETE | 新增 `paged_attention_decode`、向量化 FP32 PyTorch reference 和 fallback-only dispatch；双架构源码模式 FP16/BF16 验收均为零编译、零搜索 |
| M6.3.3 Online-softmax native 候选 | COMPLETE | 单遍读取 K/V，在寄存器维护 FP32 running max/sum/output；比较每 block 1/4/8 个 wave 的调度，不物化 score 矩阵 |
| M6.3.4 双架构 benchmark | COMPLETE | 32/32 组合通过三种子正确性；gfx1151 首轮 14/16、gfx1201 空闲 GPU 复测 12/16 过 gate，其余结果作为负面知识保留 |
| M6.3.5 独立复测与 promotion preparation | COMPLETE | gfx1151 13 项、gfx1201 12 项再次通过全部 gate；25 个不可变 pack、合并 staged dispatch 和 proposal 已生成并进入审批链 |
| M6.3.6 Approval、发布与验收 | COMPLETE | reviewer `zmu` 批准 25 项 proposal；dispatch 已原子更新，evidence、approval、compatibility、Pack Index 和 pack manifest 已签名发布，双架构源码模式验收证明零编译、零搜索 |

`paged_attention_decode(query, key_cache, value_cache, block_tables, context_lengths, scale=None)` 固定 query 为 `[batch, query_heads, head_dim]`，cache 延续 M6.2 的 `[num_blocks, block_size, kv_heads, head_dim]`，block table 为 `[batch, max_blocks_per_sequence]`。GQA 映射固定为 `kv_head = query_head // (query_heads // kv_heads)`，默认 scale 为 `1/sqrt(head_dim)`，输出与 query 同 shape、dtype 和 device。

M6.3 只实现 causal decode 的单 query attention，不包含 prefill、滑动窗口、ALiBi、RoPE、稀疏 mask、量化 cache 或 cache 管理。native 候选必须使用 FP32 online softmax 状态，不能分配与 context length 成正比的 score 临时张量；PyTorch fallback 可物化 score，作为独立正确性与通用执行路径。

M6.3 双架构评估与独立复测结果：

| Architecture | 最终过 gate | 延迟改善范围 | 最低 95% bootstrap 下界 | 最大 candidate/baseline CV |
| --- | ---: | ---: | ---: | ---: |
| gfx1151 | 13 | 45.58%-98.45% | 43.89% | 2.59% / 0.72% |
| gfx1201 | 12 | 28.08%-95.98% | 26.86% | 1.38% / 1.50% |

两架构共 32 个首轮 dtype/workload 组合均完成三种子正确性，最大绝对误差为 `6.103515625e-05`。gfx1201 首次计时受到系统级抖动影响，16 项 candidate/baseline CV 全部超标；切换到空闲 GPU 3 后 12 项恢复稳定，并在独立复测中 12/12 再次通过。gfx1151 对首轮 14 项候选独立复测后有 13 项通过；MHA d64 FP16 的 candidate CV 为 3.81%，因此即使收益超过 90% 也不晋级。

gfx1201 的四项明确负面边界为 batch=1、GQA 32x8、d128、context 1024/4096 的 FP16/BF16。候选本身 CV 不超过 1.32%，但分别比 PyTorch reference 慢 12.67%-42.98%；这些形状不写入 staged dispatch。gfx1151 的 batch=1、context 4096 两项首轮收益约 47%，但 candidate CV 为 4.58%-4.79%，同样不绕过稳定性 gate。

25 份 proposal 均为 `recommended_decision=promote`，全部 artifact、环境、workload、正确性、CV、3% 改善、bootstrap 下界和 worker gate 通过。native 源码 SHA-256 为 `4520dcc62fcd55a94e6190d606995aad08cd00dde4f9078fa8946b73d98e62d1`；gfx1151/gfx1201 artifact SHA-256 分别为 `d768014f7238080b137aabeb6a0f29be0f899af62751fbd66821014986d2534d` 与 `edcc05947b50cf88994d6a345f796a29ebdd817f8c5ce75314f6470b672a6b2c`。

reviewer `zmu` 批准全部 25 项，approval 均与 proposal 的精确 SHA-256 绑定，未使用 gate override。公开 dispatch 已原子更新为 25 entries，SHA-256 为 `aa749fec864b156be05e2a8087337c3d84d6db668c43195aec960c1346592f6a`；签名 Pack Index 含 25 个 release，SHA-256 为 `d1a66948f09963cdf1c5fe05aabb6520dadbcb6cea800bffd2db37436e2e3101`；compatibility 文档 SHA-256 为 `5d52747ef4983d13b2e3db49999bf714886e0c434f1cd9d37b609c94e75f9e50`。

最终验收继续采用源码同步模式，没有构建 wheel。gfx1151 从签名 Pack Index 安装并执行 13 个 exact winner，gfx1201 安装并执行 12 个 exact winner；两边各执行 2 个未覆盖调用并确定性回退到 `torch/paged_attention`。所有 winner 均命中审批指定的 native entrypoint，最大绝对误差为 `1.52587890625e-05`，两边均记录 `compiled_during_verification=false`、`searched_during_verification=false`。私有验收记录 SHA-256 分别为 gfx1151 `7bc82544823b1fd83a08949178b262f3f8ad7261ff255f524f058ab0905ad79b`、gfx1201 `6168a12a88c386337c4a7ffee90472eede5ba116be23b2f95a64ab797800665c`。

### Milestone 6.4 实施清单

| Step | 状态 | 验收内容 |
| --- | --- | --- |
| M6.4.1 Contract 与 workload | COMPLETE | 固定连续 `[batch, heads, sequence, head_dim]`、MHA/GQA、causal/noncausal、FP16/BF16、FP32 累加和 8 个代表 workload；首版不含 dropout、任意 mask、backward 或 packed variable length |
| M6.4.2 Reference、API 与 fallback | COMPLETE | 新增 `sdpa(query, key, value, is_causal=False, scale=None)`、显式 FP32 oracle、PyTorch SDPA fallback 和 fallback-only dispatch；公共 API 不暴露候选 variant |
| M6.4.3 Flash-like native 候选 | COMPLETE | 新增每 query row 一个 wave 的单遍 online-softmax HIP 内核，比较每 block 1/4/8 waves；不物化 `sequence x sequence` score 矩阵 |
| M6.4.4 双架构 benchmark | IN PROGRESS | 先以显式 FP32 oracle 验证 native 与 PyTorch SDPA，再执行交错计时、CV 和 3% 提升 gate |
| M6.4.5 AsmEvo controller refinement | PENDING | 为各架构最有希望的 K0 冻结 contract、preflight 与 environment，通过 `controller-v1` correctness receipt 后才允许候选计时 |
| M6.4.6 Promotion preparation | PENDING | 仅独立复测和 AsmEvo provenance 满足策略的精确 workload 可生成不可变 pack、staged dispatch 与 proposal |
| M6.4.7 Approval、发布与验收 | PENDING | 人工批准前公开 dispatch 保持 fallback-only；批准后再签名发布并执行双架构零编译、零搜索验收 |

M6.4 的性能 baseline 是目标 PyTorch build 的 `torch.nn.functional.scaled_dot_product_attention`，正确性 oracle 则显式执行 FP32 `QK^T -> causal mask -> softmax -> V`。两者故意分离，避免把候选与同一个实现自比较。native 候选使用 FP32 running max、normalization sum 和输出累加器，属于 Flash Attention 的融合 online-softmax 路径，但首轮不宣称完整 FlashAttention-2 分块或 formal equivalence。

## 15. 下一步落地点：GEMM 产品化

在继续搜索新算子前，先把已有 GEMM 转换为第一个完整用户产品。

### 15.1 Schema

新增：

```text
schemas/operator-contract.schema.json
schemas/dispatch.schema.json
schemas/compatibility.schema.json
schemas/kernel-pack.schema.json
schemas/evidence.schema.json
```

完成 schema validator，并为冲突 region、未知字段、错误 ABI 和无 fallback 等情况增加测试。

### 15.2 Runtime

新增：

```text
src/radeon_kernels/runtime/fingerprint.py
src/radeon_kernels/runtime/registry.py
src/radeon_kernels/runtime/dispatcher.py
src/radeon_kernels/runtime/loader.py
src/radeon_kernels/runtime/compatibility.py
src/radeon_kernels/runtime/explain.py
```

Runtime 必须支持：

- 检测 architecture、wave size、ROCm ABI 和 PyTorch ABI。
- 加载并校验 dispatch manifest。
- 根据算子 signature 执行确定性匹配。
- 验证 artifact 哈希和兼容性。
- 加载预编译 `.so` 或 HSACO。
- 执行 fallback。
- 记录并解释最后一次 dispatch。

### 15.3 GEMM Kernel Pack

将已经验证的 K0 artifact 制作为：

```text
radeon-kernels-pack-gfx1151-rocm715
radeon-kernels-pack-gfx1201-rocm724
```

每个 pack 必须包含：

- 不可变 artifact。
- artifact SHA-256。
- 对应 dispatch entries。
- operator semantic version。
- toolchain 和 ROCm ABI 范围。
- workload guard。
- evidence ID。
- fallback 顺序。

### 15.4 GEMM API 改造

当前 `torch.utils.cpp_extension.load()` 路径保留为开发者构建工具，但从用户默认执行路径移除。

用户调用：

```python
from radeon_kernels.ops.gemm import gemm

result = gemm(a, b)
```

新的默认行为为：

```text
dispatch lookup
-> load precompiled winner
-> execute
-> deterministic fallback when unmatched
```

### 15.5 Promotion

开发者 promotion 工具负责：

```text
private run evidence
-> verify hashes and lineage
-> sanitize private fields
-> validate confidence and compatibility
-> generate dispatch proposal
-> human approval
-> public manifest and pack index
```

Promotion 不得自动修改公开 winner。

Milestone 4 的实际发布链为：

```text
private raw evidence
-> create_proposal.py（哈希、lineage、正确性、稳定性、置信区间、脱敏）
-> immutable proposal
-> record_approval.py（人工 approve/reject，绑定 proposal SHA-256）
-> publish_approved.py（再次核对 proposal、approval、pack、dispatch）
-> Ed25519-signed minimal pack archive
-> signed evidence / approval / compatibility / pack index
-> target runtime verification
```

发布器遵守以下硬约束：

- 没有 `approved` approval 不发布。
- approval 指向的 proposal 哈希不一致不发布。
- 正常晋级要求所有 gate 为真；未通过的 proposal 同时需要 approval 中的显式 override 和发布命令的二次 override，避免单点误触。
- 发布器不增加、删除或修改 dispatch entry，只验证 proposal 中的 entry 与已有公共文件逐字段一致。单 entry 的 approval 不对同一 dispatch 文件中的其他 entry 背书，因此 dispatch 文件不在这次 detached-signature 链内。
- 候选 pack 中的 `verification.json`、主机路径和原始实验记录不会进入 release archive。
- 私钥保存在公共仓库和共享 workspace 之外；仓库只包含可轮换、可吊销的可信公钥表。

当前 proposal 结果：

| Architecture | Proposal | Gate result | Promotion state |
| --- | --- | --- | --- |
| gfx1151 | `proposal-asmevo-gemm-20260917t050538z-gfx1151-k0-817bd5fbe4e49024` | 正确性、lineage、pack/runtime、CV、3% 改进和 bootstrap 下界全部通过 | reviewer `zmu` 已批准；已签名发布并完成目标机验证 |
| gfx1201 | `proposal-asmevo-gemm-20260917t050538z-gfx1201-k0-75ca9b04fb86ae3d` | `external_timing_stable` 失败 | `needs_review`，必须复测 |

gfx1151 的首个正式 release：

| Item | Value |
| --- | --- |
| Approval | `approval-proposal-asmevo-gemm-20260917t050538z-gfx1151-k0-817bd5fbe4e49024-approved` |
| Signing key | `rk-release-2026-09` |
| Pack | `rk-gemm-gfx1151-rocm715-cp312-torch2140a0` `0.1.0` |
| Release SHA-256 | `f01bf64b5b258589e953044974e09f1e6f2f4355f821fe25433d4e8688e9002b` |
| Final wheel SHA-256 | `109b0786950773ba8cb2021648f85920d8bcb2c15204da00dbda98128e634780` |
| Exact winner | FP16 NN `256x4096x4096` -> `native/gemm_out`, variant 8 |
| Winner correctness | cosine `0.9999999403953552`，最大绝对误差 `0.0009765625` |
| Fallback | FP16 NN `64x64x64` -> `torch/mm`，最大绝对误差 `0.0` |
| Runtime gate | signed index verified；signed manifest verified；`compiled=false`；`searched=false`；overall `passed=true` |
| Runtime verification SHA-256 | `ca9a3e9d230da4e01027a92bffdac92f8484041881fc047915019983f1af0342` |

这次 release 同时证明了两条 promotion 路径：稳定且置信区间越过 3% 门槛的候选可以被人工批准后发布；表面加速明显但测量不稳定的候选不会进入正常发布链。

## 16. 第一阶段验收标准

第一阶段完成必须同时满足：

1. 全新环境只安装 runtime 和匹配的 kernel pack 即可运行 GEMM。
2. 用户调用不会触发 JIT、benchmark、autotune 或 AI。
3. gfx1151 和 gfx1201 可以选择各自的已验证 artifact。
4. 不匹配的 shape、dtype、ROCm 或架构可以稳定 fallback。
5. artifact 在加载前完成哈希和兼容性验证。
6. `rk.explain_last_dispatch()` 可以说明选择和 fallback 原因。
7. 用户 wheel 不依赖 SSH、Docker、实验配置或私有目录。
8. 搜索工具和原始 evidence 不进入 runtime 依赖。
9. dispatch schema 能拒绝重叠或含糊的 workload region。
10. GEMM 的正确性、dispatch、fallback 和安装流程都有自动化测试。

## 17. 实施顺序

```text
Milestone 1: Schema + Runtime skeleton
Milestone 2: GEMM precompiled kernel packs
Milestone 3: GEMM static dispatch and fallback
Milestone 4: Promotion pipeline
Milestone 5: RMSNorm / AddRMSNorm / RoPE / SwiGLU / Softmax
Milestone 6: GEMV / KV cache / Attention
Milestone 7: Quantization / MoE / Sampling
```

在 Milestone 1-4 完成前，不扩大临时 benchmark 或新增大量特殊加载路径。所有新算子都必须复用同一套 contract、provider、evidence、promotion、dispatch 和 kernel-pack 模型。

## 18. 最终承诺

`radeon-kernels` 对用户的产品承诺是：

> 用户安装一次，即可使用项目维护者已经搜索和验证过的 Radeon LLM 算子；正常运行不需要搜索、不消耗调优算力、不调用 AI，也不要求用户理解 ROCm 算子库之间的兼容性差异。

项目开发者负责把分散、易变的 Radeon 算子实现转换为可复现、可审计、可直接执行的公共知识。
