# 迁移审计：复用目标仓库检查

## 先使用 pre-commit

读取本次目标的 `.pre-commit-config.yaml`，确认 hooks、规则、files/exclude 与阶段，
不要把当前开发机或旧分支的配置视为所有版本的事实。Git 的 core.hooksPath 可能指向
全局工具；提交成功不能证明仓库 pre-commit 已运行。

每条提交对实际修改且仍存在的文件显式执行 `pre-commit run --files <files...>`；
终态用 `pre-commit run --from-ref <固定target_base> --to-ref <迁移tip>`，
在正确 worktree 中执行，并记录配置版本、命令、覆盖文件和输出。
使用参数列表/支持 NUL 的文件清单处理空格、删除和改名，不用未经引用的 xargs 拼接。
hook 修复文件后，只暂存该提交相关文件并 amend，再运行受影响检查。
没有适用 hook 时记录 N/A 和原因；依赖/网络故障是 NOT_RUN/BLOCKED，不能记 PASS。
不关闭 hooks、不跳过检查，不为本迁移擅自修改仓库 hook 配置。

通用检查能覆盖的部分不另造脚本：

| 检查 | 优先复用 | 限制 |
|---|---|---|
| 未定义名、局部变量、未使用 import | 目标的 Ruff F821/F823/F401 或等价 lint | 核实规则和排除路径；不判断 import 目标符号存在性 |
| Python 语法 | check-ast / 目标语法检查 | 不能证明运行期路径正确 |
| 格式与 import 顺序 | ruff-format、isort、clang-format | 不证明语义 |
| 冲突标记/配置格式 | check-merge-conflict、check-yaml、check-toml | 不发现零冲突语义漂移 |

目标若没有所需规则，优先使用其已有 lint/UT；必要时针对相关文件显式调用可用工具，
或人工 review 并记录限制，不假设被 files/exclude 排除的文件已检查。

## 仅按变更范围使用专项审计

专项工具都在技能 `scripts/`，与 Git hooks 无自动绑定关系，不导入 torch/PPU 扩展。
它们帮助发现候选问题，不能替代语义判断，不要求把所有历史案例脚本化。

### Hook 签名

触及 Hook 或其目标接口时，明确指定相关 Hook 文件和目标仓库：

```bash
python "$SKILL_DIR/scripts/audit_hook_signatures.py" \
  --repo-root <迁移worktree> <相关hook文件绝对路径...>
```

终态需要全量核对时可省略文件列表。源码树外的扩展需要用 `--source-root` 提供实际
版本源码；该参数重复使用时需同时列出仓库 python 源码根。动态 target、外部源码
不可见或纯 *args/**kwargs 的 Hook 都需人工检查真实透传；宽松签名通过不证明底层
调用兼容。`hooks=0` 不能代替预期存在的 Hook 已验证，应解释为何没有扫描到。

### 类成员归属

有新增类/函数成员或目标重构类边界时，用**台账中的明确 SHA 对**，不按 subject 猜源：

```bash
python "$SKILL_DIR/scripts/audit_class_placement.py" \
  --repo-root <迁移worktree> --source <源SHA> --target <目标SHA>
```

merge 须传 `--source-parent <已分析的父SHA>`，目标若非普通一父提交也须明确
`--target-parent`。文件移动用重复的 `--file-map old/path.py=new/path.py` 映射。
工具比较源新增成员与目标落点；名字歧义、找不到新增成员、改名或类归属变化均输出
REVIEW 并返回 1。源类与落点类相同返回 0 也不证明行为正确。合法 mixin 提取应在台账
写明新归属及调用证据；不可为了消除告警把成员搬回已经废弃的类。

### 配置读取契约

目标确实采用声明式 config bag/访问器且迁移触及配置读取时：

```bash
python "$SKILL_DIR/scripts/audit_serverargs_reads.py" \
  --repo-root <迁移worktree> --base <固定target_base> --tip <当前迁移SHA>
```

该工具检查净 diff 新增的 server_args 属性读取，与目标 ServerArgs 的 resolvable=True
注解及同文件 getter 惯例对照；包括新增文件。候选 REVIEW 返回 1，需核实运行期 vs
解析期上下文。NOT_APPLICABLE 表示目标未检测到此声明模式，不能当成全体配置已验证。
显式 base/tip 不存在、祖先关系错误、源码无法解析时返回 2，不能静默零发现。
它只针对该契约模式；其他版本采用何种配置入口仍需阅读实际定义。

## pre-commit 之外的语义检查

以下逐条检查相关符号和调用链，复用目标已有 UT/构建验证；没有可靠通用解析器时
用代码 review 记录证据，不调用未随技能提供的历史脚本：

- **from-import 目标**：定位目标模块中的定义、re-export/子模块、签名与参数语义；
  optional import、动态导出及外部库按真实版本检查，避免把合法动态行为报成错误。
  新建 PPU 文件也要逐个核对其旧基线 import 和调用实参。
- **环境变量属性**：核对新增 envs.XXX 的实际定义和读取语义，含独立 envs 模块与别名。
  遇到删除/重命名先查目标替代实现与 PPU 行为需求，不盲目恢复旧变量或机械取默认值。
- **AOT 构建引用**：逐个核对本次目标的 SAIL setup 源列表、C++ 注册和声明/定义，
  同时检查 Python 调用。删除悬空算子要确认已无调用或已有目标替代；优先实际构建。
- **C/CUDA 与 JIT**：核对命名空间、共享宏/constexpr、struct 字段数和 helper 签名。
  曾有 wrapper 移入 namespace sglang 而旧 PPU kernel 留在外层的漂移；不能据此规定
  所有版本顶格匿名 namespace 一律非法。同源码按宏生成多个模块时，保留必要的
  内部链接，避免跨 .so 合并 function-local static 导致动态 smem 初始化遗漏。
- **能力守卫**：按底层调用搜索全部新旧/融合路径，核对 split-k、smem、dtype 等限制；
  编译成功、符号存在都不证明守卫覆盖正确。
- **PPU-only 路径**：从本次源/目标代码推导实际 JIT loader 和调用参数清单，实机冒烟编译
  并执行。不要沿用固定某版本 DSV4 loader 表；社区默认不走的路径可能从未被 CI 编译。

静态审计输出、合法告警的解释和验证日志写入对应源提交记录。真实 PPU 推理验证仍是
适配完成条件，不能被 pre-commit、AST 检查、导入或冒烟编译替代。

维护这些工具后，可运行 `python -m unittest discover -s "$SKILL_DIR/scripts" -p 'test_*.py'`。
测试在临时 Git 仓库中覆盖台账完整性、冲突证据和专项审计边界，不需要 PPU 或联网；
它验证技能工具本身，不是某次模型迁移的正确性证据。
