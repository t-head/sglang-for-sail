# PPU 迁移代码模式与历史案例

本文件是查找线索和历史案例，不是所有源/目标版本的固定布局。
先阅读本次目标定义、调用者、测试和构建文档，再决定是否适用。
专项命令及退出码见 [references/audits.md](references/audits.md)，
逐提交证据格式见 [references/ledger.md](references/ledger.md)。

## 1. 平台、SDK 与构建

部分社区版本已包含 `PlatformEnum.SAIL`、`SailSRTPlatform`、
`platforms/sail.py` 和 JIT `_sail_arch_suffix()`。如果本次目标确实如此，
将旧 `ppu.py` / `PPUSRTPlatform` 中仍需保留的增量合入现有平台入口，
不能仅新增一个没有被平台发现逻辑加载的文件。

SAIL 的 CUDA 兼容 torch 路径可能保持 `device_type="cuda"`、
`is_cuda() == True`。新增社区 CUDA 分支因此可能同时命中 PPU，需检查语义。
已有版本用设备名中的 `"ZW"` 检测 SAIL；roadmap 草稿也曾描述 `PPU_SDK`
环境变量检测。不要照抄草稿覆盖已合入实现，核实不同层的实际一致性。
检测方法不能证明具体硬件型号、显存、软件版本或算子能力。

构建文件随版本变化：曾使用 `setup_ppu.py` /
`common_extension_ppu.cc`，后续社区实现可能采用 `setup_sail.py` /
`common_extension_sail.cc`。曾有 `pyproject_other.toml` 的
`srt_sail/all_sail/dev_sail` extra 和 AOT `pyproject_sail.toml`。
按目标安装文档核对，不强制所有版本使用同一种 packaging 文件。
复用目标已提供依赖，避免恢复重复旧 pyproject 或无调用的算子注册。

SDK 有 CUDA 兼容与 CUDA Free 两类命名，编译前核实激活环境，
头文件、函数前缀和链接库一起对齐。PPU 算子库对接口和 dtype 有独立扩展，
不要因包名相同就替换成社区 NVIDIA wheel。

## 2. dtype、Hook 与能力守卫

社区可能只有某个 FP8 路径，而原 PPU patch 增加 BF16/INT8 等路由。
只迁入原需求中仍未覆盖的分支，保持模型原精度、量化和 scale/block 语义；
不为“硬件支持”自动改变用户配置或一律给全部 FP8 功能添加其他 dtype。

Hook 示例（路径和符号须从目标实际代码获取）：

```python
@plugin_hook(
    "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.__init__",
    type=HookType.AROUND,
)
def sail_init(original_fn, self, *args, **kwargs):
    original_fn(self, *args, **kwargs)
    # 此处只添加当前目标仍缺少的 SAIL 行为。
```

逐个核实 Hook 的目标、调用约定、参数透传、返回值、加载条件和注册次数。
社区已等价处理时可删除对应 Hook，但不能连带删除其仍独有的 dtype/metadata 增量。
纯 `*args, **kwargs` 只能说明 Python Hook 接受任意签名，不能证明转调底层库正确。

能力守卫是语义约束。历史案例中 `deep_gemm.tf32_hc_prenorm_gemm` 的 PPU 版本
无 split-k，旧补丁只在 `mhc_pre` 限制 `n_splits=1`；社区新增
`mhc_fused_post_pre` 融合路径后漏加限制。小 batch 走其他 kernel 掩盖了问题，
decode 图捕获的大 batch 才触发。对同一底层调用的全部新旧路径检查关联修复，
包括 buffer 初始化（如 zeros vs empty），再以真实版本/负载验证。

## 3. 六种 Python 基线漂移

| 形态 | 历史表现 | 检查方式 |
|---|---|---|
| import 缺失 | 新代码调用 einops/envs/index_buf_accessor，却未迁入原来范围外的 import | 目标 pre-commit Ruff 名称规则；核实排除路径 |
| envs 属性悬空 | 名字 envs 已绑定，但社区删除了其使用的属性 | 阅读实际定义与使用侧；相关 UT/启动验证 |
| from-import 目标悬空 | import 语句完整，但目标符号被删除/移动 | 核对目标定义、动态导出/子模块及替代实现 |
| 接口搬家并改签名 | get_req_to_token_extra_context_len 从 common 移到 allocation_sizing，并从带参变无参 | import 路径与调用实参同步更新，确认新内部配置来源 |
| 类成员落错类 | is_acext 被无冲突放进 DeepEPv2Fp8ScaleFormat 而非谓词 mixin | 显式源/目标 SHA 的类归属专项审计与调用验证 |
| 配置读取契约漂移 | server_args.page_size 从物化值变成声明值，读到 None/旧值 | config bag/访问器定义及配置读取专项审计 |

零冲突、py_compile、名字存在、原 patch 文本保真都不能证明后两种形态正确。
新建 PPU 模块/mixin 可能携带整份旧 import 和调用，同样属于接口审查范围。
删除接口时先查目标迁移路径和实际 PPU 行为，不机械“社区删除即所有私有需求废弃”，
也不盲目复活旧变量。合法 optional import、star re-export、
`__getattr__` 与别名需要按实际机制甄别。

类成员移位案例：源枚举里的 `is_acext` 紧跟 `is_aiter`；目标把谓词提取进
`_MoeRunnerBackendPredicates`，并插入其他类。合并以漂移的类边界为锚，
将方法放到 NamedTuple。修复应移入目标谓词 mixin，沿用新兄弟方法风格，
不是同时保留两个实现或放弃整条 patch。新归属与源类不同可以是合法重构，
要用继承和调用证据确认。

配置案例：源的 `materialize_declarations` 在解析结束写回 ServerArgs，
目标删除该步骤，改为声明投影到 config bag。相同的
`self.server_args.page_size` 字节序列因此变成 None 读取。
目标若确实有 `resolvable=True` 和分层访问器，应使用字段所在 bag 的 getter；
解析期 `resolving_view` 的合法读法不能机械替换。
同一函数兄弟字段已迁移但某个字段仍旧读法，是很强的漏改线索。

## 4. AOT / JIT 与真实执行

历史 AOT 源列表曾仍引用已经被社区移除的 `bmm_fp8.cu` 和 qserve kernel。
处理时同时核对源文件、注册引用、头文件声明、定义与 Python 使用侧；
某个算子在社区改走外部库不证明 PPU 同版本也有该替代。

JIT 漂移案例包括：宏 `SGL_TOPK`/常量改成 `TopKParams.topk` 运行时字段、
struct 增加字段导致结构化绑定数量不匹配、`radix_topk` helper 参数变化、
wrapper 与 kernel 命名空间不同。检查原 PPU 源码中仍依赖的每个共享定义。

同一份源码以不同宏生成多个 .so 时，随意删除匿名命名空间可能改变链接语义，
导致 function-local static 跨模块合并，第二个模块漏掉动态 smem 初始化。
根据实际 wrapper 作用域选择命名空间，不能机械搜索所有顶格 namespace 并删除。

从本次代码枚举只有 PPU 会编译/执行的 loader，逐个以实际参数在 PPU 上验证。
有的默认值让 PPU 走社区 CI 永不走的 kernel；有的模块在非 SAIL 直接 raise。
编译、单元测试、服务启动、真实 prefill/decode 和精度验证分别记录，
不能互相替代。平台共享测试与模型测试位置遵循目标版本 test/README.md。
