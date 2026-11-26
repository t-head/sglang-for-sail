# GLM-5.3 MegaMoE 最佳实践：0.5.21 → 0.5.23 参数迁移对照与缺口说明

这份说明配合同目录的 `glm-5-3-megamoe-mxfp4-best-practice.md` 使用。前者讲「怎么跑」，这份讲「从 0.5.21 那份文档迁移到 0.5.23 时，哪些参数改了、哪些能力这版没有、以及验证阶段要补什么」。所有结论都按 0.5.23 迁移分支 `glm_dev_0523` 的实际代码核对（核对基线：head 999b365124588d9aa41982186ce49aa3f3745f37，即 0.5.23 主线 f04a60e 加本批迁移提交）。

## 一、一句话结论

0.5.21 那套 MegaMoE 长序列 prefill 配置在 0.5.23 上可以直接沿用，但有三处必须改：**① 前缀缓存开关改名（旧名彻底失效、没有兼容别名）；② 开前缀缓存后页大小要求从固定 64 变成 64 × index_kpool（GLM-5.3 是 256）；③ DeepGEMM 缓存目录以 `SGLANG_DG_CACHE_DIR` 为准，`DG_JIT_CACHE_DIR` 会被程序自己覆写，手工设了不生效**。另外 0.5.23 新增了一条起服校验：LayerSplit 与 HiCache 同时开时必须是 layer_first 布局＋kernel I/O 后端。

## 二、逐参数对照表

「变化」列的四种标注：**同名同义**（照抄即可）／**改名**（必须改脚本）／**新增**（0.5.21 没有）／**依赖侧**（sglang 代码不解析，透传给 DeepGEMM、Mooncake、NCCL）。

| 0.5.21 文档里的写法 | 0.5.23 写法 | 变化 | 依据（0.5.23 代码位置） |
| --- | --- | --- | --- |
| `SGLANG_ENABLE_DSA_CP_PREFIX_CACHE` | `SGLANG_DSA_CP_PREFIX_CACHE`（默认 False） | **改名，旧名失效且无别名** | `python/sglang/srt/environ.py:2069`（`EnvBool`，不是带别名的类型）；`disaggregation/utils.py:166-168` |
| `--page-size 64`（开前缀缓存时） | `--page-size 64 × index_kpool`（GLM-5.3 为 256） | **语义变化** | `mem_cache/kv_cache_builder.py:27-56`（`require_dsa_cp_page_size`）、`:425-427` 调用点 |
| `DG_JIT_CACHE_DIR=<目录>` | `SGLANG_DG_CACHE_DIR=<目录>` | **手工设置无效**：导入时被 `SGLANG_DG_CACHE_DIR` 覆写 | `layers/deep_gemm_wrapper/compile_utils.py:57-58`；`DG_CACHE_DIR` 仍被调优表侧单独读取（`layers/deep_gemm_tuner/utils.py:28-29`） |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` | 同名，但 0.5.23 默认是 True | 同名不同默认值 | `environ.py:1414` |
| `SGLANG_SAIL_DEEPEP_RECV_HOOK=0` | 同名，0.5.23 默认 True | 同名不同默认值 | `environ.py:1496` |
| `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1` | 同名，默认 False | 同名同义 | `environ.py:796` |
| `--moe-a2a-backend megamoe` | 同名 | 同名同义 | `arg_groups/fields/exec_.py:850`、`:866` |
| `--enable-w4a4-mxfp4-megamoe` | 同名，默认 False；新增 MTP 侧配套开关 | 同名＋新增配套 | `arg_groups/fields/exec_.py:875-880`；`arg_groups/fields/spec.py:201-215`（`--speculative-enable-w4a4-mxfp4-megamoe`） |
| `--enable-prefill-cp` / `--cp-strategy interleave` / `--attn-cp-size` | 同名（choices 仍是 zigzag／interleave） | 同名同义 | `arg_groups/fields/parallel.py:157-163` |
| `--attention-backend dsa`、`--dsa-prefill-backend flashmla_sparse`、`--dsa-decode-backend flashmla_kv` | 同名；prefill 侧另有 `flashmla_sparse_q8` 可选 | 同名＋新取值 | `arg_groups/fields/exec_.py:328`（choices 列表） |
| `--enable-dsa-cache-layer-split` | 同名 | 同名同义 | `arg_groups/fields/parallel.py:176` |
| （无） | `--hicache-mem-layout layer_first` ＋ HiCache I/O 后端 kernel：LayerSplit 打开时为硬性要求 | **新增起服校验** | `disaggregation/utils.py:197-225`（`require_dsa_cp_hicache_layout`）、调用点 `kv_cache_builder.py:428-436` |
| （无） | `SGLANG_DSA_LAYER_SPLIT_BATCH_PAGES`（默认 "1"）、`..._VERIFY`（默认 "0"） | **新增**（0.5.23 迁移批带入） | `mem_cache/dsa_cache_layer_split.py:218`、`:222` |
| `--weight-loader-disable-mmap` | 同名 | 同名同义 | `arg_groups/fields/model.py:299-302` |
| `--deepep-mode auto` | 同名（choices auto／normal／low_latency，默认 auto） | 同名同义 | `arg_groups/fields/exec_.py:903-906` |

## 三、这版没有带过来的东西（缺口，验证阶段暴露需求再单开）

1. **ChunkCache 机制整块不在 0.5.23**：0.5.21 那批改动里有 6 条生产码与一个专项单测（16 条用例）是给 ChunkCache 用的；0.5.23 的工作树里既没有 `chunk_cache*.py` 也没有 `ChunkCache` 类名（grep 命中为 0，留痕见归档）。结论：**该能力这版不存在，不迁**。如果验证阶段确实出现「需要 ChunkCache 才跑得通」的场景，再按 opt-in 单开一项，不预先塞进默认路径。
2. **0.5.21 的部分前缀缓存实现在 0.5.23 已有等价版本**，所以迁移时没有再合一遍代码，具体对应关系（供评审对照，避免误认为漏迁）：
   - 开关读取：0521 的带参判断 → 0523 的无参 `supports_dsa_cp_prefix_cache()`（`disaggregation/utils.py:171-195`），并且多要求 CP 宽度大于 1；
   - 「错位命中只警告一次」：0521 用模块级全局量 → 0523 挂在运行上下文 `resources.dsa_cp_misaligned_warned`（`mem_cache/base_prefix_cache.py:441-463`）；
   - 复用长度：0521 只算设备命中 → 0523 的 `dsa_cp_prefix_reuse_length()` 把 HiCache 命中段与整 KV 命中都算进去（`:425-438`）；
   - 对齐值：0521 直接用 page_size → 0523 用 `lcm(page_size, index_kpool)`（`:388-423`）；
   - 请求侧调用点：0521 散在三个入口 → 0523 收敛到 `mem_cache/common.py:260`、`:268` 一处，因此旧入口不重复打闸门；
   - 传输侧的策略复查：0521 在 PD 连接层再加一次 `assert cp_strategy == interleave`，0523 有意不做（理由写在 `disaggregation/common/conn.py:529-536` 的说明里：策略检查已经在前缀缓存谓词里，重复检查会让「没要这个特性、只是把开关设进环境」的配置反而中断传输）。
3. **TopK 相关的两枚 opt-in 开关只作用于 ragged 分发路径**：`SGLANG_DSA_RAGGED_BF16_TOPK`、`SGLANG_DSA_DEEPSELECT` 都是默认 False，paged 变体继续走 0.5.23 自己的自动分派，不扩门控范围（有无收益要等实测，不用推断代替）。
4. **一处约定未定（按裁决保持现状）**：packed MTP draft 的传输层号，0521 把装配处从「设备池总层数」改成「host 池 live target 层数」，0.5.23 的对应装配点仍用设备池总层数（`mem_cache/hybrid_cache/hybrid_pool_assembler.py:309`，而 host 池自己的 `target_layer_num` 已是分片感知值，`:206-212`）。静态读码无法判定在 LayerSplit＋packed draft 组合下哪个才是对的，因此**本次不迁、保持 0.5.23 现状**，把它和「host 池 draft 传输号用 live target 层数」这条改动一起放到验证窗口里用上机证据定。文档层面：如果你的部署同时用到 LayerSplit 与 packed MTP draft，请在看板里关注 draft 层命中与 KV 一致性，出现异常先怀疑这一处。

## 四、验证阶段的要求（交付口径）

- **分场景各按各自最优配置验证**：prefill 主导与 decode 主导两类场景分别做 A/B 对照（两臂同一份 whl、同一台机器、只差 MegaMoE 两枚开关），每场景多轮取中位数，分别报告收益并给出「推荐开启场景」。禁止用单一配置或总体均值下结论。
- **每场景硬门禁**：error=0、无异常 fallback、命中标记齐全（MegaMoE 命中证据、前缀缓存的 `DSA prefill CP prefix cache is ENABLED (experimental)`）、精度与对照持平。
- **计时纯净性**：确认没有留下会污染计时的调试变量（例如 bypass 执行、打印参数之类的开关）并留证。
- **完整 case**：一个报错修完不算完，要跑通该场景对应的完整测试 case（全量评测或全流程），并保留原始日志与校验值。
- **0.5.21 的性能数字只能当参考**，不能写进 0.5.23 的结论；本文表中的数字已按此标注。

## 五、迁移过程中的合规与身份口径（评审需要）

- 0.5.23 这条线的迁移只读参考 0.5.21 分支，不改写它的历史，也不动在途分支本体；提交与 PR 由本仓库指定的唯一提交执行人操作。
- 本仓库（t-head 侧）提交信息与正文不出现外部组织域名与仓库名、不出现内部工单号；制品与镜像地址按本仓库制品库填写。文档里的占位符（`<0.5.23 sglang whl 地址>`、`<PPU DOCKER NAME>`、`<YOUR_MODEL_PATH>`）在出包后由提交执行人或验证窗口回填真实地址，不凭记忆编造链接。
- 容器命名沿用本群规则（自建容器名必须带归属后缀）。
