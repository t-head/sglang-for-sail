# 迁移台账与冲突证据

## 文件与恢复

报告目录位于迁移工作树外，至少保存：`ledger.json`（全量候选、决策和映射）、
`report.md`（可读最终报告）、`commits/<source_sha>.md`（逐条语义分析），以及
`evidence/<source_sha>/` 下的原始冲突、关键 diff、审计与实机验证日志。
证据引用相对 ledger.json 所在目录；记录中的代码引用包含对应 SHA、路径、符号/行号。
不把可执行审计工具放在 Git 元数据目录。

`migration_ledger.py init` 固定三个输入 SHA，检查完整历史和祖先关系，保存每个候选的
完整 SHA、所有父节点和标题，不使用关键词过滤或省略 merge。输出文件已存在时拒绝覆盖。
按脚本生成的 records 顺序逐条补充，不能删除 SKIP/EXCLUDED 记录。
只盘点时可省略 --target-base，refs.target_base 为 null；之后开始迁移前将其补成
包含 ref 和完整 sha 的对象，记录 remote/核实时间，再运行 check。不要重新初始化覆盖
已经做过的分析；target_base 未固定时不接受已迁移/SKIP 记录或 --complete。

恢复时先核对台账固定 SHA、当前 HEAD、工作区、CHERRY_PICK_HEAD 和最近操作日志。
若提交已创建但台账尚未写入，通过保存的 target_before、原始 SHA trailer 和实际 diff
确认归属；不要再次 pick，也不要仅凭相同 subject 猜配。修改/重写目标提交后更新映射。

## 每条记录

初始化字段只是待填写的结构，空字段不是已完成证据：

```json
{
  "source_sha": "完整源SHA",
  "parents": ["源父SHA"],
  "subject": "原标题",
  "classification": "ppu",
  "intent": "修改目的及触发条件",
  "impact": ["path:符号；模块/接口/dtype/Hook/构建影响"],
  "dependencies": ["范围内的前置源SHA"],
  "external_dependencies": ["范围外依赖及在目标已满足的证据"],
  "coverage": {
    "state": "partial",
    "target_base_sha": "本次固定社区目标SHA",
    "basis_sha": "实际逐功能复核的target_before完整SHA",
    "evidence": ["目标SHA:path:symbol；PR URL/merge SHA；逻辑等价解释及核实时间"],
    "remaining": ["目标仍缺少的BF16路由"]
  },
  "code_split": {
    "basis_sha": "实际逐单元复核的target_before完整SHA",
    "community_covered": [{
      "id": "C1",
      "source_file": "源文件路径",
      "symbols": ["源符号或hunk对应的行为"],
      "target_file": "固定目标中的对应路径",
      "target_symbols": ["目标符号/执行入口"],
      "origin": "community_target",
      "evidence": ["目标SHA和源码位置；真实PR diff/commit；等价理由"],
      "action": "reuse_existing"
    }],
    "retain": [{
      "id": "R1",
      "source_file": "仍需保留的源文件路径",
      "symbols": ["未覆盖的符号/行为"],
      "condition": "仍有需求的条件",
      "requires": ["接口/配置消费者/前置提交和验证依赖"],
      "reason": "该PR或其他目标实现为何未覆盖",
      "action": "retain_and_adapt"
    }],
    "do_not_restore": [],
    "unresolved": []
  },
  "decision": "RESOLVED",
  "reason": "社区已有平台检测，仅迁入剩余BF16支持",
  "target_before": "开始本条操作时的完整HEAD",
  "target_sha": "最终目标SHA；SKIP/EXCLUDED为空",
  "conflicts": [{
    "path": "path/to/file.py",
    "symbol": "Class.method",
    "evidence": ["evidence/源SHA/conflict.diff", "evidence/源SHA/stage2.txt"],
    "cause": "社区将函数提取为mixin并改了参数",
    "source_intent": "源补丁要保留的PPU行为",
    "target_intent": "目标重构的结构/契约及必须保留的行为",
    "resolution": "按目标新参数透传BF16分支，不恢复已覆盖的平台检测",
    "verification": "检查调用者并运行相应UT，日志位置"
  }],
  "adjustments": ["实际迁入范围、旧新符号映射、重定位/改写及理由"],
  "validation": [{
    "command": "实际命令与环境",
    "result": "PASS",
    "evidence": "结果说明或未运行原因",
    "artifacts": ["evidence/源SHA/pre-commit.log"]
  }],
  "review": "complete",
  "operations": ["时间：分析/尝试/冲突/解决/格式修复/amend及SHA变化"],
  "notes": "剩余风险或后续依赖说明"
}
```

`coverage.state` 为 full/partial/absent/obsolete/unknown。
`classification` 为 ppu/dependency/community_fix/unrelated，未分析时为 unclassified。
`decision` 为 PENDING/BLOCKED/DIRECT/SKIP/RESOLVED/ADAPTED/EXCLUDED。
DIRECT 表示无冲突、无语义适配；普通格式修正可记录在 adjustments。
partial 只能迁移剩余逻辑，最终为 ADAPTED 或 RESOLVED。
部分覆盖必须同时填写 code_split.community_covered 和 code_split.retain，逐项注明文件、
符号、目标替代实现、等价理由或剩余需求；do_not_restore 记录旧实现废弃/被替代原因。
有未解决单元时先保持 PENDING/BLOCKED，不把它伪装成“全部覆盖”。PR 一旦合入不代表
其所有源功能都覆盖，PR 仅发布配置也不代表相应 kernel/runner 已存在。
EXCLUDED 记录用户排除范围或已授权任务范围的具体依据；不要用它绕过未知依赖。

SKIP 必须有 full/obsolete 的覆盖证据和后续依赖复查说明。EXCLUDED 需有范围理由。
两者均不创建目标提交，也不要求伪造一次 pre-commit PASS。
SKIP/已迁移记录必须填写 coverage.target_base_sha 与 basis_sha，且 basis_sha 等于本条
实际审查的 target_before，位于固定目标的迁移祖先链。纯已 upstream 的跳过也应记录当时
HEAD。校验器拒绝其他分支的快照或更换目标后残留的基线；不会证明自然语言的等价判断。
提供 code_split 时，它的 basis_sha 必须与 coverage.basis_sha 一致，防止更换页眉后继续
沿用旧版本的逐功能清单；证据内容本身仍需人工复核。
所有已处理记录都需填写 impact 和 operations。验证日志/输出文件引用放入 validation.artifacts，
不能仅把文件名写进自然语言 evidence 后声称已保存；校验器检查 artifacts 中的文件存在。
简单检查没有输出文件时可以只写 evidence 说明；它不证明工件内容或代码等价。
PENDING/BLOCKED 保存当前操作与缺失条件，不能出现在“全部完成”的报告中。

## 冲突证据保存

在编辑/暂存前，以工具或独立命令把以下原始输出写入本条 evidence 目录：

```bash
git status --porcelain=v1
git ls-files -u
git diff --cc
git show :1:path/to/file.py
git show :2:path/to/file.py
git show :3:path/to/file.py
```

stage 1 是三方基线，stage 2 是当前目标，stage 3 是所迁源内容；删除冲突可能没有某个
stage，明确记录缺失，不伪造空文件。二进制 blob 按二进制保存并描述选择依据。
重命名同时记录旧/新路径；证据需能还原原始冲突内容，不能只写“已手动解决”。
每个冲突文件/函数分别说明三方意图、原因、选择和验证。相同原因可以引用共用分析，
但不能用一个“Accept Both”笼统替代逐块解释。

## 检查与最终报告

`check` 可在中途执行；`--complete --target-tip SHA` 还校验无未决记录、终态证据字段、
全部目标提交的映射、target_before 与真实父节点，以及源依赖顺序。校验器不会理解
自然语言证据或证明代码等价；人工 review 和真实推理验证仍必需。
PENDING/BLOCKED 可保留旧目标的历史分析；若已有分析的 coverage.target_base_sha 或内嵌的
target_upstream_prs.target_base_sha 与当前目标不同，CLI 在 warnings 中提示 stale。
结构检查 PASS 不表示这些旧覆盖结论已适用于新目标；必须重查后才能完成迁移决定。

目标版本核查时，顶层保存 target_upstream_prs（或其证据文件路径）：固定 target_base SHA、
roadmap 功能→PR 对照、实际 merged 状态、真实 merge SHA、目标包含判定和核实时间。
可用 inspect_upstream_prs.py 生成历史包含检查，格式见
[target-upstream.md](target-upstream.md)。未合入目标的提案与目标已覆盖实现分开记录。
code_split 中 origin=earlier_migration 的内容需指向实际前序迁入 SHA，不能写成社区原有。
无目标的纯源盘点可保留 coverage 基线为空；目标补充或更换后，重新生成 PR 包含清单和
覆盖拆分，PENDING 旧记录不自动变成可执行的 SKIP。

正常是一条源记录对应一条目标提交。多个源被一个提交语义覆盖时，实际迁入者记录目标
SHA，后续记录 SKIP 并引用前序目标；不随意 squash/reorder。merge 使用经分析的 mainline
或独有解决增量，记录选定源父节点和原始证据。

若需要额外集成修复，顶层 `integration_commits` 每项保存：
`target_sha`、`reason`、`source_shas`、`validation`。它不能替代漏掉的源提交记录。
检查全部源记录的总数与各 decision 数量；检查 target_base..target_tip 内没有未解释提交，
没有映射到范围外的提交。不要用“提交总数相等”证明基线或内容正确。

最终 report.md 列出固定 refs/SHA、remote、核实时间、每条决策和源→目标映射、
SKIP 的 upstream/目标证据、冲突与适配细节链接、适用检查和结果、PPU 环境与推理
命令/精度结果、尚未验证项。区分代码迁移完成和真实适配验证完成。
