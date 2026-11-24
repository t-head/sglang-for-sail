# GLM-5.3 MXFP4 长序列 Prefill 开启 MegaMoE 最佳实践（sglang 0.5.23）

本文是《GLM-5.3 推理最佳实践》里「MXFP4 长序列 Prefill Only CP+EP」一章的补充，讲的是**在 0.5.23 这条线上怎么把 MegaMoE 打开、怎么验证真的打开了、以及哪些参数和 0.5.21 不一样**。配置方法来自 0.5.21 那份最佳实践（同一套 case），正文按 0.5.23 的实际参数与开关名重写：老写法里在 0.5.23 上已经不存在的参数、被改名或被默认值反转的开关，都在对应位置写清楚，另外单独有一份逐参数的对照与核对表（见文末链接）。

## 一、环境准备

| 项目 | 说明 |
| --- | --- |
| 硬件 | 真武 890P 单机 4 卡（或 1P1D 两台 8 卡） |
| 软件 | 0.5.23 版 PPU 镜像；镜像与出包地址以本批 CI 产出为准（本仓库不使用外部组织的制品域名，请按 t-head 侧制品库地址填写） |
| 模型权重 | GLM-5.3-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken |

容器内装包（只换 sglang，其余依赖保持镜像自带版本；换包前后各留一份 `pip freeze` 便于回溯）：

```shell
# sglang 0.5.23 本批交付件（whl 地址以 CI 产出为准，安装时固定 --no-deps）
pip install <0.5.23 sglang whl 地址> --force-reinstall --no-deps

# 校验：版本与关键后端能 import
python3 -c "import sglang; print(sglang.__version__)"
python3 -c "import deep_gemm, flash_mla; print('deep_gemm / flash_mla ok')"
```

依赖版本要点（0.5.23 与 0.5.21 的差异会直接决定起服成败）：

- DeepGEMM：0.5.23 这版 MXFP4 MegaMoE 走的是 SM89+ 的使能路径，对 DeepGEMM 的接口要求比 0.5.21 更严。**不要用 0.5.21 那套依赖包直接配 0.5.23 的 sglang**；装完必须先做下面的「预检」再起正式服务。
- FlashMLA 同理，以本批交付说明列出的版本为准。

## 二、预检（算预检，不算正式起服）

装机后先做参数级冒烟，确认两件事：一是 MegaMoE 这组参数在 0.5.23 上能被解析，二是没有把不兼容组合带进去。

```shell
python3 -m sglang.launch_server --help 2>&1 | grep -E -- '--moe-a2a-backend|--enable-w4a4-mxfp4-megamoe|--cp-strategy|--enable-dsa-cache-layer-split|--dsa-prefill-backend'
```

上面每一行都应有输出（0.5.23 参数名与 0.5.21 一致的部分保持原名；0.5.21 用过但在 0.5.23 不存在的参数，本文不再写）。

## 三、单机 4 卡 Prefill Only（推荐配置）

Attn CP + MoE EP，开启 MegaMoE：

```shell
# 进到容器（示例命令；容器名按本群规则必须带归属后缀）
sudo docker run --privileged -it -e HOST_HOSTNAME=$(hostname) \
    --network=host --device=/dev/infiniband --ipc=host --device=/dev/alixpu_ctl --device=/dev/alixpu \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --init --pid=host \
    -v <path to your workspace> \
    -w <path to your workspace> \
    --name <your docker_name> \
    <PPU DOCKER NAME>

# 关掉代理（代理会污染下载与计时）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# DeepGEMM 缓存目录：0.5.23 上以 SGLANG_DG_CACHE_DIR 为准。
# 原因：sglang 在导入时会用 SGLANG_DG_CACHE_DIR 覆写 DG_JIT_CACHE_DIR
# （见 python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py），
# 所以只设 DG_JIT_CACHE_DIR 不生效；DG_CACHE_DIR 是调优表侧另一个变量，仍单独读。
export SGLANG_DG_CACHE_DIR=/path/to/dg_cache/
export DG_CACHE_DIR=${SGLANG_DG_CACHE_DIR}

# 0.5.23 的默认值是 True（会做全量预编译），本文这套叠加态要求显式关掉，
# 否则首次起服会先跑数小时的 precompile warmup。
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# 以下三个开关在 0.5.23 上都是「默认关、本文显式开」，属于依赖库读取的变量，
# sglang 代码本身不解析它们，只透传给 DeepGEMM / Mooncake / NCCL。
export DG_USE_MOE_DYNAMIC_TILE=1
export MC_LOG_LEVEL=TRACE
export MC_NUM_QP_PER_EP=4
export MC_TE_METRIC=1

# PD/CP 相关：0.5.23 默认 False，本文这个场景需要置 1。
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
# 0.5.23 默认 True，本文按 0.5.21 实测口径设为 0（关掉 deepep 收包 hook）。
export SGLANG_SAIL_DEEPEP_RECV_HOOK=0

python3 -m sglang.launch_server \
  --model-path <GLM-5.3-MXFP4-权重路径> \
  --host 127.0.0.1 --port 31010 \
  --tp-size 4 --ep-size 4 --attn-cp-size 4 --enable-prefill-cp --cp-strategy interleave \
  --chunked-prefill-size 32768 --mem-fraction-static 0.9 \
  --attention-backend dsa --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv \
  --moe-a2a-backend megamoe --enable-w4a4-mxfp4-megamoe \
  --disable-custom-all-reduce --disable-cuda-graph --disable-overlap-schedule \
  --cuda-graph-max-bs-decode 16 --max-running-requests 16 --num-continuous-decode-steps 1 \
  --disable-radix-cache --trust-remote-code --disable-shared-experts-fusion --enable-metrics \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --watchdog-timeout 24000 --dist-timeout 24000 \
  --weight-loader-disable-mmap
```

对照臂（关 MegaMoE）：其它参数逐字不动，只把两枚开关换成 `--moe-a2a-backend deepep --deepep-mode auto`。做性能对照时两臂必须用同一份 whl、同一台机器、只差这两枚开关，否则结论不算正式口径。

**这两枚开关在 0.5.23 上的含义（0.5.21 文档没写，这里补齐）**：

- `--enable-w4a4-mxfp4-megamoe` 默认 False；0.5.23 上 MegaMoE 只在 SM89+ 生效，并且 GLM 架构走 MegaMoE 时按 token 预算做闸门（长序列 prefill 才吃得到收益，decode 小批不满足预算时不会走 MegaMoE）。
- 因此结论只能分场景给：prefill 主导场景可以按上面这套开；decode 主导场景要单独对照后再决定，不能拿一个总体均值说「开了更快」。
- 带 MTP/EAGLE 时还有一枚 `--speculative-enable-w4a4-mxfp4-megamoe`：不设时 draft 跟随 target 的 MMA 类型；如果想让 draft 留在 W4A8，就显式传 `--no-speculative-enable-w4a4-mxfp4-megamoe`，此时要注意 draft 可能在 KV 池定容之后再申请一块 MegaMoE 对称缓冲，需要下调 `--mem-fraction-static`。

## 四、client 命令（单机）

```bash
# In 100000 / Out 1，Concurrency 1
python3 -m sglang.bench_serving --max-concurrency 1 --dataset-name random \
  --random-input-len=100000 --random-output-len=1 --random-range-ratio 1 \
  --num-prompts 10 --host 127.0.0.1 --port 31010
```

## 五、PD 分离配置（P 节点开 MegaMoE，890P MXFP4 1P1D）

### P_Node（Prefill，开启 MegaMoE）

```shell
export SGLANG_DG_CACHE_DIR="/mnt/dg_cache0"
export DG_CACHE_DIR="/mnt/dg_cache0"
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# 前缀缓存（radix cache）开关：0.5.23 只认这个名字，默认关闭。
# 0.5.21 用的旧名 SGLANG_ENABLE_DSA_CP_PREFIX_CACHE 在 0.5.23 已失效且不保留兼容别名，
# 设了不报错也不生效——迁移时务必改名（详见「如何开启前缀缓存」一节）。
# export SGLANG_DSA_CP_PREFIX_CACHE=1

export DG_USE_MOE_DYNAMIC_TILE=1
export MC_LOG_LEVEL=TRACE
export MC_NUM_QP_PER_EP=4
export MC_TE_METRIC=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_SAIL_DEEPEP_RECV_HOOK=0

python3 -m sglang.launch_server --host 0.0.0.0 --port 8999 --disable-radix-cache --trust-remote-code \
  --model-path <YOUR_MODEL_PATH> --log-level info --enable-cache-report --enable-metrics \
  --dist-timeout 60000 --watchdog-timeout 60000 \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 --tp-size 8 \
  --page-size 64 --speculative-algorithm EAGLE --speculative-attention-mode decode \
  --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --disable-custom-all-reduce --moe-a2a-backend megamoe --enable-w4a4-mxfp4-megamoe \
  --disable-shared-experts-fusion --reasoning-parser glm45 --tool-call-parser glm47 \
  --disaggregation-mode prefill --mem-fraction-static 0.85 \
  --attention-backend dsa --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv \
  --enable-prefill-cp --cp-strategy interleave \
  --chunked-prefill-size 32768 --attn-cp-size 8 --disable-overlap-schedule --ep-size 8 \
  --enable-request-time-stats-logging --disable-cuda-graph \
  --weight-loader-disable-mmap
```

`--page-size 64` 只适用于**不开前缀缓存**的默认形态。一旦打开 `SGLANG_DSA_CP_PREFIX_CACHE`，页大小要求会变成「64 × index_kpool」，GLM-5.3 的 config 里 index_kpool=4 时就必须写 `--page-size 256`，否则起服直接被 `require_dsa_cp_page_size` 拒绝（报错里会同时给出你传的 page_size、index_kpool 和期望值）。这条是 0.5.21 与 0.5.23 最容易踩的参数差。

### D_Node EP（context 100k）

```shell
export SGLANG_DG_CACHE_DIR="/mnt/dg_cache0"
export DG_CACHE_DIR="/mnt/dg_cache0"
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

export DG_USE_MOE_DYNAMIC_TILE=1
export MC_LOG_LEVEL=TRACE
export MC_NUM_QP_PER_EP=4
export MC_TE_METRIC=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_SAIL_DEEPEP_RECV_HOOK=0

python3 -m sglang.launch_server --host 0.0.0.0 --port 8999 --disable-radix-cache --trust-remote-code \
  --model-path <YOUR_MODEL_PATH> --log-level info --enable-cache-report --enable-metrics \
  --dist-timeout 60000 --watchdog-timeout 60000 \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 --tp-size 8 \
  --page-size 64 --speculative-algorithm EAGLE --speculative-attention-mode decode \
  --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --disable-custom-all-reduce --moe-a2a-backend deepep --deepep-mode auto \
  --disable-shared-experts-fusion --reasoning-parser glm45 --tool-call-parser glm47 \
  --disaggregation-mode decode --mem-fraction-static 0.8 \
  --prefill-attention-backend fa3 --decode-attention-backend flashmla --cuda-graph-max-bs-decode 32 \
  --max-running-requests 1280 --enable-dp-lm-head --enable-dp-attention --stream-interval 64 \
  --dp-size 8 \
  --weight-loader-disable-mmap
```

D 节点保持 deepep，不套 MegaMoE 开关；前缀缓存开关只作用于 prefill 节点，decode 节点不适用（decode 侧的 radix cache 由 `--disaggregation-decode-enable-radix-cache` 自己管）。

### D_Node DP + EP（context 4k）

与上一段只差并行形态与 prefill 分块：`--chunked-prefill-size 2048` 加在启动参数里，其余同上。

### Router（部署在 P 节点）

```shell
export PREFILL_NODE=<prefill-ip>
export DECODE_NODE=<decode-ip>

python -m sglang_router.launch_router \
  --pd-disaggregation --host 0.0.0.0 --mini-lb --port 7000 \
  --prefill "http://<prefill-ip>:8999" \
  --decode "http://<decode-ip>:8999"
```

硬口径（router 地址必须可路由）：`--prefill/--decode` 要填对端真实可路由 IP，不能填 127.0.0.1 或 localhost。填成回环地址时 D 节点会把 router 当 prefill 握手，PD 之间的 KV 传输根本建立不起来，但 bench 仍会把 HTTP 200＋`finish_reason=abort` 的请求计入 Successful requests——表面全成功、数据全假。识别方法：P 节点日志 `ReqTimeStats` 里 `transfer_total>0` 且没有 AbortReq / handshake failed；D 节点日志没有「Error fetching prefill server info from bootstrap … Connection refused」。

### client（PD）

```bash
DATASET_PATH=<YOUR_LOCAL_DATASET_PATH>

# In 4096 / Out 1500，Concurrency 8
python3 -m sglang.bench_serving --max-concurrency 8 --dataset-name random \
  --random-input-len=4096 --random-output-len=1500 --random-range-ratio 1 \
  --num-prompts 80 --dataset-path ${DATASET_PATH} --host 0.0.0.0 --port 7000

# In 100000 / Out 1000，Concurrency 1
python3 -m sglang.bench_serving --max-concurrency 1 --dataset-name random \
  --random-input-len=100000 --random-output-len=1000 --random-range-ratio 1 \
  --num-prompts 10 --dataset-path ${DATASET_PATH} --host 0.0.0.0 --port 7000
```

合格判据：PD 链路可能静默失败（见上面 router 硬口径）。链路健康的特征是 `Total generated tokens (retokenized)` 非零；如果它是 0 而且每请求时延≈14 秒（bootstrap 15 次重试的常数），这轮数据全部作废。链路正常时 bench 的 TTFT/TPOT 才可信，并建议再用 `/generate` 流式直测交叉验证一次。

## 六、参考性能

0.5.21 那套配置实测过的两组数字（**同 case、同权重、同机型的 0.5.21 参考值，不能当成 0.5.23 的结论**；0.5.23 交付需要在本批件上重跑，prefill 与 decode 两类场景各按各自最优配置分别测）：

| 场景 | 并行配置 | 节点-卡数 | In / Out | Concurrency | Mean TTFT (ms) | Mean TPOT (ms) | 单卡 Total Throughput (tok/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 单机 Prefill Only | EP4 TP4 CP4（开 MegaMoE） | 1-4 | 100000 / 1 | 1 | 9216.67 | — | 2712.13 |
| 单机 Prefill Only | EP4 TP4 CP4（deepep 对照） | 1-4 | 100000 / 1 | 1 | 9640.83 | — | 2592.84 |
| PD 1P1D | P CP8 TP8 EP8（开 MegaMoE）＋D EP8 TP1 DP8 | 2-8 | 4096 / 1500 | 8 | 790.48 | 13.52 | 131.59 |
| PD 1P1D | P CP8 TP8 EP8（开 MegaMoE）＋D EP8 TP1 DP8 | 2-8 | 100000 / 1000 | 1 | 24723.00 | 17.85 | 148.33 |

表注：
- PD 两行为 1P1D 两台各 8 卡共 16 卡（P 开 MegaMoE、D 用 deepep），单卡＝总量÷16；两侧都开 MTP（EAGLE）。
- 100k 那行的 Mean TTFT 被 2 个 prefill 长尾请求拉高（中位 4904.11ms），长尾根因当时未取证，如实标注。

## 七、如何开启前缀缓存（radix cache）

- 配置方法：P 节点设置 `SGLANG_DSA_CP_PREFIX_CACHE=1`，并且启动命令**不带** `--disable-radix-cache`。两件事必须一起做：`--disable-radix-cache` 是显式关闭，带着它时环境变量无效。
- **旧开关名已失效**：0.5.21 用的是 `SGLANG_ENABLE_DSA_CP_PREFIX_CACHE`。0.5.23 上没有这个变量、也没有保留别名兼容，设了不报错但完全不生效。从 0.5.21 迁移脚本时只认新名字；如果现场确实有依赖旧名的脚本，请反馈具体场景，再按「凭证据单独开兼容别名」的流程评估。
- 生效条件（缺一不可）：PD 分离的 prefill 节点 ＋ `--enable-prefill-cp` ＋ `--cp-strategy interleave` ＋ attn CP 宽度大于 1 ＋ 页大小满足 `64 × index_kpool`（GLM-5.3 的 index_kpool=4 ⇒ `--page-size 256`）。CP 宽度为 1 时开关保持惰性（平台可能把 CP 宽度按 tp_size 重算成 1，那种情况下对齐保护并不生效，所以开关也不会把 radix cache 打开）。
- 如果同时打开 `--enable-dsa-cache-layer-split`（LayerSplit）与 HiCache：0.5.23 起服时会做布局前置校验，要求 HiCache 用 `layer_first` 布局＋`kernel` I/O 后端，不满足直接报错退出。这条保护来自 0.5.21 实测组合，0.5.23 上是否还允许别的布局尚未复测，因此按严格门执行；若验证阶段实测某布局可用，会另出放宽的改动并同步更新本节。
- 生效判据：server.log 出现 `DSA prefill CP prefix cache is ENABLED (experimental)`；如果仍出现 `Radix cache is disabled on this PD worker because DSA prefill CP requires CP-aware KV-cache resharding.`，说明没生效。
- 不打开时：保持默认（不设该变量、继续带 `--disable-radix-cache`）即可，行为与之前完全一致。
- 打开后若前缀命中长度不满足对齐要求，该请求的复用会被丢弃并重算（日志按进程只提示一次）——这是实验特性的既定降级契约，不是故障。

## 八、相关开关默认值一览（0.5.23）

| 开关 | 0.5.23 默认 | 说明 |
| --- | --- | --- |
| `SGLANG_DSA_CP_PREFIX_CACHE` | False | 前缀缓存实验开关（旧名失效，见第七节） |
| `--enable-w4a4-mxfp4-megamoe` | False | W4A4 MXFP4 MegaMoE，需配 `--moe-a2a-backend megamoe` |
| `SGLANG_DSA_TOPK_NO_SIMT` | False | no-SIMT BF16 prefill TopK 开关 |
| `SGLANG_DSA_RAGGED_BF16_TOPK` | False | ragged TopK 走 BF16（opt-in） |
| `SGLANG_DSA_DEEPSELECT` | False | DSA top-k 走 DeepSelect 内核（opt-in） |
| `SGLANG_DSA_LAYER_SPLIT_BATCH_PAGES` | 1（读 env 字符串） | LayerSplit 只广播活跃 batch 的 KV 页 |
| `SGLANG_DSA_LAYER_SPLIT_BATCH_PAGES_VERIFY` | 0 | 上一项的对拍校验 |
| `SGLANG_SAIL_BF16_INDEXER` | False | BF16 indexer |
| `SGLANG_DSA_USE_DSV4_BF16_TOPK` | True | DSA indexer 的 BF16 topk 家族（默认为开，别按「全默认关」理解） |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | True | 本文场景要求显式设 0 |
| `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` | False | 本文场景设 1 |
| `SGLANG_SAIL_DEEPEP_RECV_HOOK` | True | 本文按 0.5.21 实测口径设 0 |

逐参数对照、以及哪些 0.5.21 参数在 0.5.23 上已删改的核对表，见同目录的迁移对照说明。
