---
name: cherry-pick-ppu-patches
description: >-
  Inventory and migrate T-HEAD PPU/SAIL patches between SGLang releases. Compare a
  PPU branch with its community baseline, check upstream PR inclusion and semantic
  coverage against the specified target release tag/branch,
  resolve refactor conflicts, and maintain a per-commit migration ledger.
  Use for PPU patch inventory, migration, or review; not ordinary release-bot
  cherry-picks or unrelated branch rebases.
---

# Cherry-pick PPU Patches

将旧社区基线上的 PPU/SAIL 增量迁入指定的新社区版本。支持只盘点、制定计划、
执行迁移、恢复已有迁移；只盘点或审查不授权执行 cherry-pick。

迁移必须同时做到：A 全量盘点、B 按目标代码去重、C 理解语义后解决冲突、
D 每个候选提交有可追溯记录。不能把“Git 应用成功”当作“适配完成”。

## 工作边界与输入

先利用现有 Git remote/ref 和上下文确认，只有关键输入无法推断时才询问用户：

- `source_base`：源 PPU 版本对应的**社区** tag/commit，例如社区 `v0.5.23`。
- `source_tip`：旧 PPU 分支，例如 `release/v0.5.23`。
- `target_base`：本次要迁入的新社区版本 tag/commit/分支（如 `0.5.24`），
  不能用同名 PPU fork 分支或最新 main 代替。
- 报告目录、迁移工作分支，以及用户明确的排除范围（如有）。

只盘点旧版本增量时不需要 target_base 或迁移工作分支，不为此要求用户选择新版本。
若要求核查“新版本已覆盖哪些功能”或制定去重/迁移计划，则必须先明确 target_base。
没有该输入时只能盘点与记录 PR 进度，不得默认改查 main 后输出 SKIP 建议。

区分社区和 PPU remote，避免同名 tag 指错仓库。需要更新 ref 时仅 fetch 对应 remote，
不自动 pull、移动既有分支或覆盖 tag。将三个 ref 固定为完整 SHA，记录 remote URL、
核实时间和来源。分支之后移动不改变本次基线。

保留用户工作区和已有 cherry-pick/rebase 状态。执行迁移时优先从固定
`target_base` 创建干净的独立 worktree 和 `codex/ppu-migration-...` 分支；
不要为了清理环境丢弃用户修改。技能的绝对目录记为 `SKILL_DIR`，
迁移 worktree 中未必包含技能。报告存放在工作树外的任务产物目录，
持续保存 JSON 台账、逐提交说明和冲突证据，不把报告混入代码提交。

PPU 适配遵循仓库 AGENTS.md：实际设备、SDK 风格、驱动、SAIL、PCCL、框架和算子库
版本都需核实；不擅自改变模型精度/量化。PPU 同名 deep_gemm、FlashAttention、
FlashMLA 库需核对实际接口、dtype 和版本，不能按 NVIDIA 上游实现推断。
平台命名、构建文件和配置读法以**本次目标代码**为准，参考案例不是跨版本硬性布局。

## A. 盘点旧版本增量

先读 [references/ledger.md](references/ledger.md)，初始化可恢复的台账：

```bash
python "$SKILL_DIR/scripts/migration_ledger.py" init \
  --source-base <社区旧基线> --source-tip <PPU源分支> \
  --target-base <社区新基线> --output <报告目录>/ledger.json
```

脚本只读 Git 并新建报告，不迁移代码、不覆盖已有台账。它校验旧基线是源 tip 的祖先，
按 `git rev-list --reverse --topo-order source_base..source_tip` 枚举**全部**候选，
包括 merge；原 SHA、父提交和标题随清单保存。若关系不成立或仓库历史不完整，
先核实重写/浅克隆/选错基线原因，不静默改用 merge-base 或猜测起始 SHA。
只盘点可省略 --target-base；进入迁移前再核实并固定台账中的 refs.target_base。

逐个阅读完整 message 和 diff，以及相关调用者/定义/测试：
普通提交对父提交比较；merge 分别检查父链和 merge 结果中的独有解决逻辑。
标题、作者和 PPU 关键词仅用于定位，不能据此提前丢弃候选。

为每个候选记录目的、影响文件/模块、接口与 dtype/Hook 变化、前置依赖，
分类为 `ppu`、`dependency`、`community_fix` 或 `unrelated`。
社区修复若仍是 PPU 必要依赖应迁移。排除项记录理由和范围依据，不自动排除
所有“非 PPU 标题”提交。原始 source tip 相对旧基线的总 diff 用于交叉检查漏项；
不能仅靠净 diff 取代逐提交分析。

保持祖先先于后代的拓扑顺序；不要按提交日期排序或擅自重排。
merge 不得无条件跳过，也不得未经分析就 `cherry-pick -m 1`：
若父链变更已逐个覆盖，只处理 merge 独有的解决增量；
若采用 mainline 迁移，先记录父节点选择及范围，防止再次重放同一父链内容。

只请求盘点时，在交付完整候选及分类/待确认项后结束，不创建迁移提交。

## B. 按目标版本核查 upstream PR，再逐功能去重

读 [references/target-upstream.md](references/target-upstream.md)。**版本迁移的去重基准
是固定 target_base 的实际代码，不是 main 的当前状态。** 只有用户明确选择 main
为目标时才按它检查；不得为了查 upstream 进度而把 main 替换成版本目标。

从 [PPU upstream roadmap](https://github.com/sgl-project/sglang/issues/37519)
提取阶段、功能范围和关联 PR；必要时沿讨论/关联 PR 补全拆分与替代关系。
查实际 PR diff 和 merged/merge SHA，不把 checkbox、描述、标题或 changed-files 数量当作
等价证明。对每个候选 PR 分开记录：**GitHub 是否合入、固定目标是否包含、功能是否等价**。

对已 merged 的 PR 检查真实 merge commit 是否是固定 target_base 的祖先，并阅读目标中
仍有效的代码。工具只辅助判断提交历史，不会识别语义覆盖：

```bash
python "$SKILL_DIR/scripts/inspect_upstream_prs.py" --repo-root <仓库> \
  --target <固定target_base_SHA> --prs <报告目录>/pr-metadata.json
```

open / closed-but-unmerged 或已合入其他分支、未进入目标的 PR，只记 upstream 进度，
**不据此 SKIP，也不把提案算作目标已覆盖**。但需另查目标是否通过回移、拆分、重构等
方式获得等价实现；merge SHA 不可达并不证明功能不存在。缺对象/历史时补齐所需证据，
无法核实时记 unknown，不静默拿 main 代替。

对目标中确实存在的相关实现，把源提交按功能/代码单元拆开，分别列出
**社区已覆盖**和**仍需保留/适配**的源码文件、符号、调用路径、理由、依赖与证据。
另外注明被新设计替代或已经无效、不应机械恢复的旧代码；保留能力不等于保留旧文件。
一个 PR 可能覆盖多个源提交的一部分，一个源提交也可能对应多个 PR；不能把文件名消失
当成功能缺失，也不能把部分配置/同名 wrapper 出现当作整条提交已覆盖。

先用祖先关系和 `git cherry`/`git patch-id --stable` 寻找候选，再比对可达执行路径的
入口、守卫、参数/返回契约、dtype、默认值、Hook、注册、配置消费者、构建和测试范围。
合入后被 revert 的行为不算仍有效。应用前还要按**当前迁移 HEAD**复核，区分固定社区
目标原有实现与本次前序迁移引入的实现；两者的证据来源分别记录。

逐个逻辑单元标记覆盖情况，并作以下决策：

| 覆盖状态 | 处理 |
|---|---|
| `full` | 全部必要行为已存在且仍有效：`SKIP`，记录目标 SHA、代码位置、PR/commit 及等价理由 |
| `partial` | 保留目标已有逻辑，仅迁入剩余部分；记 `ADAPTED` 或 `RESOLVED`，不得整条 SKIP |
| `absent` | 迁入仍需要的完整逻辑；冲突/语义改写决定最终处理状态 |
| `obsolete` | 目标架构已使旧方案失效：证明需求由何处满足或为何不再适用，再 SKIP |
| `unknown` | 保持 PENDING/BLOCKED 并解释缺少的证据，不能冒充“已 upstream” |

移除冗余 Hook 时同时检查加载注册、目标函数语义和后续依赖。
每次跳过/部分迁入前，复核后续提交是否依赖旧符号、环境变量、构建项或 Hook。
源补丁与其回退提交不能仅因净 diff 为零就自动跳过，须检查中间依赖。
用户更换目标版本/分支或推进固定基线后，原覆盖结论标为需复核；重建目标 PR 包含清单和
逐功能拆分，不沿用原 main/其他 release 的 SKIP 结论。

## C. 逐提交迁移与冲突处理

在应用**之前**保存 `target_before`、决策和预期改动。普通完整补丁使用
`git cherry-pick -x <source_sha>`；经分析需要部分迁入/重定位时，可用
`--no-commit` 在隔离工作树中编辑后提交，或实现目标结构下的等价增量。
保留原始作者/意图，记录 source SHA；说明实际适配与原 patch 的差别。
已初始化但未完成的迁移先读台账、Git 状态和证据再继续，不重复初始化或重放。

冲突发生后，**在编辑/暂存前**保存 Git 状态、`git ls-files -u`、
`git diff --cc`，以及冲突文件的 base/ours/theirs stage blob（存在的 stage）。
文件删除、改名、二进制或无 stage 的手工迁移也要记录对应证据。
证据文件与每个冲突文件/函数的分析关联，详见台账参考。

对每个冲突理解三方：源提交父版本、源提交后版本、当前目标 HEAD。
跟踪目标的重命名、提取 mixin、接口重构和调用者变化，决定保留、移位、改写或
去掉已覆盖部分。**禁止未经逐块语义分析就 Accept Both，或整文件 ours/theirs 覆盖。**
无法确定正确行为时保存现场并标 BLOCKED，不丢弃补丁、不推测等价。

解决后逐文件 review 完整结果，而不只查看冲突行：

- 新增方法是否属于正确的类/函数/分支；零冲突也可能落错位置。
- import 目标、调用签名、配置读取契约、dtype 路由是否符合目标基线；
  新建 PPU 文件可能携带整份旧接口引用，同样需要检查。
- Hook 是否重复注册/覆盖社区已支持逻辑，新增参数是否向底层正确透传。
- C/CUDA 命名空间、宏/constexpr、struct 字段与 helper 签名是否匹配目标。
- 每个 PPU 能力守卫保护的底层算子的全部调用点，含社区新增融合/快速路径，
  是否同样满足限制；不扩大成无关平台的行为修改。

按 [references/audits.md](references/audits.md) 运行适用的 pre-commit 和专项审计。
不要假定 Git hooks 已安装，也不要把本机全局 hook 等同仓库 pre-commit。
禁止为通过检查关闭 hook 或使用 `--no-verify`；工具/环境故障记录为未验证，
修复环境后重试。格式修复只暂存本提交相关文件；完成当前提交的修复和 review，
再迁移下一个。需要 amend 时更新目标 SHA 和证据；不擅自改写已共享历史。

空提交先检查全部逻辑是否已覆盖。只有确认后才能 SKIP：
尚在 cherry-pick 状态时用 `git cherry-pick --skip`；未开始则直接记录跳过，
不能在无活动 cherry-pick 时运行 --skip。零冲突不代表无需适配。

## D. 每个候选提交都要有记录

台账状态见 [references/ledger.md](references/ledger.md)：

- `DIRECT`：无冲突直接应用，未作语义适配。
- `SKIP`：目标/前序迁移已覆盖或有证据证明不再需要，不生成目标提交。
- `RESOLVED`：发生 Git 冲突并完成语义解决，含部分 upstream 的冲突迁移。
- `ADAPTED`：没有 Git 冲突，但需部分迁入、重定位或按新接口改写。
- `EXCLUDED`：有范围依据的不迁移项；不是“社区已包含”的同义词。
- `PENDING` / `BLOCKED`：尚未完成，不能计作成功。

每个候选都保存 source SHA、分类、目的、影响范围、依赖、覆盖证据、处理方式、
target_before/target_sha、冲突原文证据、三方逻辑、解决理由、适配差异与验证结果。
覆盖记录固定 `coverage.target_base_sha` 和实际审查的 `coverage.basis_sha`；后者是
固定目标或其迁移链上的当前 HEAD，不能填其他分支快照。部分覆盖必须保存代码单元拆分，
让“复用社区已有实现”和“迁入剩余增量”的处理逐项可查。
**SKIP/EXCLUDED 也立即落盘**。每次开始操作前、冲突后、完成 review 后更新；
中断恢复时结合 Git 状态核对最后一条记录，不能仅凭提交标题推断来源。
台账检验脚本只验证结构、Git 范围和映射，不能替代语义审查：

```bash
python "$SKILL_DIR/scripts/migration_ledger.py" check <报告目录>/ledger.json
python "$SKILL_DIR/scripts/migration_ledger.py" check <报告目录>/ledger.json \
  --target-tip <迁移终态SHA> --complete
```

## 最终验证与交付

先核对固定 target_base 到迁移终态的实际 diff、源候选覆盖和目标提交映射。
每个目标提交应可追溯到源记录；若确需单独集成修复提交，在台账
`integration_commits` 记录 SHA、原因、涉及源提交和验证，不把它伪装成社区原有内容。
保持源 patch 意图，不以源/目标最终树逐字相同为要求。

优先运行目标仓库已有 pre-commit、相关 UT 与构建，按变更范围选择专项检查，
不用缺失的历史工具阻塞，也不将其标为 PASS。随后在核实过的 PPU 环境，
使用原模型、精度、卡数及启动参数验证真实启动、prefill/decode、
相关 Hook/kernel 路径和准确性；多卡变更覆盖实际 PCCL 路径。
构建/JIT 冒烟与真实推理分别记录。性能相关改动在同配置下比较基线，
报告 batch、输入/输出长度、并发、卡数、拓扑与精度，不用理论峰值代替端到端结果。

最终报告包含固定 refs/SHA、全量逐提交清单与处理数量、upstream 对照、
冲突和适配说明、目标 SHA 映射、测试命令/结果/日志及未完成项。
候选数量必须等于所有状态数量之和，完整完成时 PENDING/BLOCKED 为零。
实际推理未验证时明确写“代码迁移完成，PPU 实机验证待完成”，不能宣称适配完成。
本技能不额外授权 push、创建 MR/PR、部署或改写共享分支。

代码模式与历史案例按需读 [reference.md](reference.md)，不能代替对当前目标的核实。
