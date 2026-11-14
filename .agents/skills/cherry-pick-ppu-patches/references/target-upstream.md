# 固定目标上的 upstream PR 包含关系与功能拆分

## 先固定目标，后读取 PR 状态

目标必须是用户指定的新版本 tag/branch/commit，解析为完整 commit SHA。
区分社区 remote 与 PPU fork；annotated tag 同时记录 tag object 和 peeled commit。
不默认读取 main，不用 PR 的 base 分支或当前最新发布版本替代输入。
只盘点源增量且没有目标时，PR 状态可作为进度信息；“目标包含/已覆盖/应跳过”保持未知。

Roadmap：<https://github.com/sgl-project/sglang/issues/37519>。
读取实际内容及相关讨论，提取功能单元和 PR URL；跟进拆分、堆叠、替代或关闭的 PR。
roadmap 是索引，PR 描述是意图，实际 diff 与固定目标的存活实现才是覆盖证据。
不要将某次核实的 PR 状态/设备布局永久硬编码到技能。
Roadmap/API 无法读取时记录未核实的范围，不能填空清单后宣称已完成 PR 覆盖核查。

从连接器/API 取得每个 PR 的 merged、state、merge_commit_sha、head_sha、实际文件/diff
及核实时间。保存原始响应，并整理为以下本地输入，供历史包含关系检查：

```json
{
  "prs": [{
    "number": 12345,
    "url": "PR的原始URL",
    "merged": true,
    "state": "closed",
    "merge_commit_sha": "GitHub返回的完整真实mergeSHA",
    "head_sha": "核实的PR head SHA",
    "roadmap_scope": "此PR实际负责的功能/数据范围"
  }]
}
```

运行 `scripts/inspect_upstream_prs.py --target <固定SHA> --prs <本地JSON>`。
脚本只读本地 Git 并输出 JSON，不 fetch、不修改分支、不自动作迁移决定。
缺对象需通过正确 remote/PR 获取所需历史再核实，不能把读取失败当作不包含。

| 实际状态 | 历史判定 | 后续处理 |
|---|---|---|
| merged=true，真实 merge SHA 是 target_base 祖先 | included | 仅证明合入历史存在；继续检查 revert、重构、执行路径与语义 |
| merged=true，但不在目标祖先链 | not_in_target | 不以“main 已合入”作 SKIP；另查目标的回移/等价实现 |
| open 或 closed 且 merged=false | not_merged | 只记提案进度，不算目标覆盖；临时 merge_commit_sha 即使非空也不能使用 |
| 合入字段不足、merge SHA/对象缺失或历史不完整 | unknown | 记录原因，补齐证据；不默认 main、不宣称覆盖或缺失 |

目标可能通过 cherry-pick、重写或拆分而包含等价代码，此时原 merge SHA 不可达，仍需
按目标实际 commit/路径证明等价。反之，祖先可达也可能已经 revert 或只保留不可达代码。
`history_presence` 与 `coverage.state` 是独立维度。

## 从整个提交拆到可处理的功能单元

源提交逐文件/函数/hunk 与目标中的实现对应，记录功能意图和实际范围，不只数文件或行数。
源依赖与影响范围包括 wrapper、调用者、metadata、注册、构建、配置及其消费者、测试场景。
检查一个 PR 拆分多个源提交，以及一个源提交组合多个 PR 的多对多关系。

逐源提交至少输出这两份可操作清单：

- **社区已覆盖**：源文件/符号/行为 → 固定目标中的文件/符号/commit；说明参数、dtype、
  默认值、调用路径和守卫为何等价。目标重构成子类、mixin 或新 factory 也可覆盖，
  不要求恢复旧 Hook 或旧文件名。
- **仍需保留/适配**：缺少的源码单元、为什么仍有需求、目标落点、依赖、适用条件及验证。
  部分覆盖只迁移剩余增量；保留能力不要求保留旧实现形式。

另外明确 **不应恢复的旧代码**（已由新设计替代、死代码/冗余或已废弃路线），以及
**未解决的覆盖问题**。有条件保留项必须说明触发条件，不能把未验证项强行归入前两类。

以下差异需要按调用链核实，不能直接套用“整个 PR 已覆盖”：

- wrapper 名称相同，但原全局 ops Hook 还影响其他 backend：检查注册/apply 的时机、
  模块导入别名与缓存 partial/对象字段；局部子类覆盖不自动覆盖其他消费者。
- PR 只发布配置数据：逐文件比较实际内容，还需核对 schema、selector 版本/命名、
  加载目录和 kernel/runner 的消费契约。相同配置文件不证明相同运行能力。
- 更保守的目标守卫满足安全目的，但不等同源版本的全部准入策略；不为复原旧文本
  擅自扩大设备/dtype 支持。
- 普通 sanity tests 不等同原多卡、DP/MTP、padding、精度或性能回归场景。

## 代码单元到迁移动作

| 覆盖 | 单元处理 | 整条源提交处理 |
|---|---|---|
| equivalent | 复用目标已有实现，不再次迁入 | 所有必要行为均覆盖且无剩余依赖问题，才可 SKIP |
| partial | 保留目标已覆盖部分，只迁入缺口并适配新接口 | ADAPTED；实际发生 Git 冲突则 RESOLVED |
| absent | 迁入仍需保留的功能及依赖 | DIRECT / ADAPTED / RESOLVED，依实际结果 |
| superseded/obsolete | 证明新结构满足原需求或需求已不再成立 | 不机械恢复旧代码；记录理由 |
| unknown | 保留问题与缺失证据 | PENDING/BLOCKED，不能整条 SKIP |

将代码单元写入每条记录的 `code_split`，格式见 [ledger.md](ledger.md)。
PR 未进入目标不妨碍继续迁移仍需要的 PPU patch，也不能凭 PR 标题重合先删源代码。
如果目标通过其他实现已满足部分需求，仍按上述单元拆分处理。

固定 target_base 表示“社区原有覆盖”；当前迁移 HEAD 还可能包含前序移植成果。
应用前用 `target_before` 再核对并记录来源，不能把前序迁入内容记成原社区 PR 已合入。
`coverage.target_base_sha` 固定本次目标，`coverage.basis_sha` 标明实际复核快照。
更换 tag/branch/SHA 后保留历史证据，但将旧覆盖结论标为需复核；PR 包含关系与单元
清单全部重新按新目标检查，不能只替换报告页眉中的版本名。
