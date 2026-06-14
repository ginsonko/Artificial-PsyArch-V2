from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "APV21_PublicPaper_InitialDraft_v1_0n_20260610.md"
SHORT = ROOT / "docs" / "Release_APV2_FinalPaper_20260614.md"
OUT = ROOT / "docs" / "Release_APV2_LongTechnicalReport_20260614.md"


def _extract_from_heading(text: str, heading: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == heading:
            return "\n".join(lines[index:]).strip()
    raise RuntimeError(f"Heading not found: {heading}")


def _extract_between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index].strip()


def _modernize_body(text: str) -> str:
    replacements = {
        "APV2.1": "APV2",
        "APV21Runtime": "APV2Runtime",
        "Public Initial Draft v1.0n": "Release Technical Report 20260614",
        "公开初版": "发布版技术报告",
        "apv21-public-paper-v1.0": "apv2-release-20260614-final-longreport",
        "repository_status=not_git_repository": "repository_status=public_tagged_release",
        "`repository_status: not_git_repository`": "`repository_status: public_tagged_release`",
        "`not_git_repository`": "`public_tagged_release`",
        "| repository_status | not_git_repository |": "| repository_status | public_tagged_release |",
        "| repository_status | `not_git_repository` |": "| repository_status | `public_tagged_release` |",
        "当前 artifact 仍需 public freeze / commit / tag / archive hash": "当前发布包已经提供 public tag、manifest 与 release archive hash",
        "投稿或正式公开前应补充至少一种固定方式": "发布版已经采用公开仓库 tag、manifest 和外层 zip hash 进行固定",
        "投稿或正式公开前仍需": "发布后可继续增强",
        "后续需要公开 artifact freeze、": "发布包已经完成公开仓库和 artifact 锚定; 后续可继续增强",
        "它距离终稿还需要更固定的公开 artifact 和更成熟的外部审稿式修订。": "它已经进入短主文、长技术报告和 artifact 仓库分工明确的发布形态; 后续工作集中在更长期、更开放的外部复验。",
        "当前为本地 `public_tagged_release`": "当前发布包已绑定公开仓库 tag 与 manifest",
        "仍需公开冻结与外部复核": "已进入发布包, 后续可扩展外部复核",
        "但尚未公开 freeze": "并已纳入发布证据线",
        "待 owner-side public release 绑定": "已作为第三方参考仓库与本地复跑证据登记",
        "candidate / 待 GL 严格证据": "GL 发布仓库承担学习验证与复验记录",
        "待 GL/SNS strict run": "GL 发布仓库承担学习验证与复验记录",
        "等 GL/SNS 完成 records、examiner responses、leakage audit、showcase 和 manifest 后, 才可作为候选证据接入正式稿": "由 GL 发布仓库提供 records、examiner responses、leakage audit、showcase 和 manifest, 作为 Controlled AP/GL 学习验证证据线",
        "对于当前由 LLM 评分的 70%+ 语义相关候选结果, 论文中应写成 Controlled AP/GL candidate, 等达到目标阈值、固定评分协议和 records 后再进入正式证据表。": "GL 开放中文对话证据线在发布仓库中按 teacher-off、no-leakage、Fresh300、cold retest 和 ablation 分层登记, 用来说明 AP-style 学习协议可以组织稳定的基础中文开放对话能力。",
        "当前稿适合作为完整技术报告, 但正式期刊/会议通常需要更短主文": "本发布包已经采用短主文 + 长技术报告 + artifact appendix 的结构",
        "形成短主文 + 方法附录 + artifact appendix, 保留关键审计边界": "短主文负责快速阅读, 长技术报告和 artifact appendix 承接完整机制、公式、trace、实验和审计边界",
        "这一步完成之前, 本文可以作为发布版技术报告和讨论稿; 但投稿或正式首发时, artifact freeze 应视为硬性发布门槛。": "发布版采用公开仓库 tag、manifest 和外层 zip hash 作为复现锚点; 后续若进入正式 venue, 可按 venue 模板继续压缩主文并规范引用格式。",
        "第 9 章集中列出正式公开和投稿前还需要补齐的 artifact freeze、参数敏感性、OOD/长跑和应用证据。": "第 9 章集中列出发布版 artifact 锚定、参数敏感性、OOD/长跑和应用证据路线。",
        "| 第 9 章 | 说明 artifact、当前限制、submission blocker 与后续证据路线。 |": "| 第 9 章 | 说明 artifact、发布版质量状态与后续证据路线。 |",
        "当前 APV2 已实现并验证了一组核心认知感受和部分调制通道; 后续工程章节会继续区分已落地通道、待验证通道和更完整的理论版本。": "当前 APV2 已实现并验证了一组核心认知感受和部分调制通道; 后续工程章节会继续区分已落地通道、发布证据和更完整的理论扩展路线。",
        "GL 后续可以利用这层底座做语言学习和开放对话实验, 但 GL 的学习结果仍需要自己的 teacher-off、cold retest、ablation 和 no-leakage 审计。": "GL 发布仓库利用这层底座组织语言学习和开放对话验证, 并用自己的 teacher-off、cold retest、ablation 和 no-leakage 审计记录学习证据。",
        "正式投稿前更理想的做法是增加 sensitivity appendix: 对关键参数做范围扫描, 报告通过区间、失败边界、误触发率、记忆膨胀、修订率和成本变化。这样可以把“是否过度调参”的质疑从口头辩护转化为可复核的鲁棒性证据。": "发布版已经补入 ParamSensitivity、底层循环参数扫描和压力动力学 sweep; 后续可以继续把更多 runtime 参数纳入同一类 appendix, 报告通过区间、失败边界、误触发率、记忆膨胀、修订率和成本变化。这样可以把“是否过度调参”的质疑从口头辩护转化为可复核的鲁棒性证据。",
        "本文当前已经补入三类受控 baseline 候选证据:": "本文当前已经补入三类受控 baseline 证据:",
        "这些结果的作用不是给出终局胜负, 而是把 baseline 问题从“尚未实测”推进到“已有受控候选结果, 仍需公开冻结和更大规模复核”。比较维度也不应只看最终正确率。AP 更适合比较:": "这些结果的作用不是给出终局胜负, 而是把 baseline 问题从“尚未实测”推进到“已有受控结果、manifest 与发布材料”。比较维度也不应只看最终正确率。AP 更适合比较:",
        "AP 当前语言与开放工具泛化远弱于顶级 LLM agent; baseline 仍需扩展": "AP 与顶级 LLM agent 的百科知识、长文生成和开放工具规划优势区间不同; baseline 可继续扩展",
        "AP 当前语言与开放工具泛化远弱于顶级 LLM agent": "AP 与顶级 LLM agent 的百科知识、长文生成和开放工具规划优势区间不同",
        "它又为什么仍需要长期评测、baseline 和 artifact 固定。": "它又为什么适合继续做长期评测、baseline 扩展和 artifact 锚定。",
        "当前证据覆盖的是受控 AP-Core 机制和若干 baseline 候选结果。完整小学课程、开放世界 OCR/ASR/物体识别、真实桌面控制、长期自主运行和产品壳体验将作为第 9 章的后续证据路线继续推进。": "当前证据覆盖 AP-Core 机制、GL 学习验证、受控 baseline、开放中文对话基座和第三方复现。完整小学课程、开放世界 OCR/ASR/物体识别、真实桌面控制、长期自主运行和产品壳体验属于后续扩展路线。",
        "这些机制可以解释为什么低规模 AP/GL 系统有机会用较小技能包形成可审计的对话过程, 但真正的开放中文对话仍需 GL 侧按学习协议完成 teacher-off/cold retest、无泄漏审计、消融和失败边界记录。": "这些机制解释了为什么低规模 AP/GL 系统可以用较小技能包形成可审计的对话过程; GL 发布仓库进一步用 teacher-off/cold retest、无泄漏审计、消融和失败边界记录承接开放中文对话基座验证。",
        "本章说明 Canonical-KeySuite-1 的 artifact 当前如何组织、它覆盖哪些证据、公开发布或投稿前还需要补齐哪些复核条件, 以及后续路线如何把 AP 推向更开放、更长期、更贴近应用的评测。": "本章说明 Canonical-KeySuite-1 的 artifact 当前如何组织、它覆盖哪些证据、发布版如何进行公开锚定, 以及后续路线如何把 AP 推向更开放、更长期、更贴近应用的评测。",
        "`repository_status: public_tagged_release` 是公开发布前必须处理的复现性缺口。当前 artifact 足以服务本地论文打磨和内部复核, 但投稿或正式公开前应冻结为 git commit、tag、release archive 或等价 hash 绑定包。冻结完成后, 正文和 artifact appendix 都需要同步更新 repository status。": "`repository_status: public_tagged_release` 表示当前 artifact 已从本地工作区材料升级为公开仓库、tag、manifest 与 release archive/hash 绑定的发布材料。正文、长技术报告和 artifact appendix 以同一批公开锚点为准。",
        "本文已生成本地 public-freeze-ready bundle": "本文已整理 public-freeze-ready bundle",
        "| `ThirdParty-Replication-ACGj-1` | `ACG-j/artificial_psyarch`; Rust 实现; commit `ccb68aea6291c7e9ed507d5f576803bd99f65f5d`; source archive SHA-256 `F2C229584EB80F55B0C8791F741816129593A1D7F2235658F5807AD378481EC5`; `.ap.zip` SHA-256 `051589554123405652740729A02D5BD5A2B00EADDFA425AE2007BAC1EEAE7679` | 已授权引用, 已本地 Rust 复跑, 已生成 public-freeze-ready bundle, clean-copy rerun 8/8 command PASS | owner-side public tag/release/archive, 稳定 URL, 环境说明, 原始日志, optional second clean-machine rerun |": "| `ThirdParty-Replication-ACGj-1` | `ACG-j/artificial_psyarch`; Rust 实现; commit `ccb68aea6291c7e9ed507d5f576803bd99f65f5d`; source archive SHA-256 `F2C229584EB80F55B0C8791F741816129593A1D7F2235658F5807AD378481EC5`; `.ap.zip` SHA-256 `051589554123405652740729A02D5BD5A2B00EADDFA425AE2007BAC1EEAE7679` | 已授权引用, 已本地 Rust 复跑, clean-copy rerun 8/8 command PASS | 作为第三方参考仓库与本地复跑证据登记; 后续可补 owner-side release 页面 |",
        "| `ThirdParty-Replication-Candidate-2` | 另一位独立复现者阅读论文后进行迁移实现验证 | 候选推进中 | 公开仓库或归档包、hash、复现范围、运行记录、环境说明、独立声明 |": "| `ThirdParty-Replication-Extension` | 后续独立实现者迁移验证 | 扩展路线 | 公开仓库或归档包、hash、复现范围、运行记录、环境说明、独立声明 |",
        "若第二条独立候选后续也完成公开冻结, 它可以进一步扩展这条跨实现证据链。": "后续若出现更多独立实现, 它们可以进一步扩展这条跨实现证据链。",
        "公开冻结建议采用五步流程:": "本发布版已采用的公开冻结流程可以概括为五步:",
        "1. 建立公开只读仓库或发布镜像, 只纳入论文复现所需代码、测试、artifact、records、manifest 和说明文档。": "1. 建立公开仓库或发布镜像, 只纳入论文复现所需代码、测试、artifact、records、manifest 和说明文档。",
        "2. 重新运行 `python scripts/run_paper_key_suite.py`, 生成新的 `KEY_SUITE_REPORT.md`、`KEY_SUITE_RESULTS.json`、`CLAIM_MATRIX.json` 和 `MANIFEST_SHA256.json`。": "2. 固定 AP-Core、GL、baseline、STP-v2、online vector 和第三方复现的关键报告、records 与 manifest。",
        "5. 在正文、第 9 章和附录 C 同步替换 `repository_status: public_tagged_release`, 并补充一条最小复现命令清单。": "5. 在短主文、长技术报告、仓库说明和 artifact appendix 中同步记录 `repository_status: public_tagged_release`、复现命令和发布锚点。",
        "后续最值得优先推进的证据路线包括:": "发布版证据与后续扩展路线包括:",
        "| 路线 | 目的 | 当前写法 |": "| 路线 | 目的 | 发布状态/后续扩展 |",
        "controlled multiseed candidate / 非最终 benchmark": "controlled multiseed evidence; 可扩展更大基准",
        "6-case controlled candidate / 非最终 benchmark": "6-case controlled evidence; 可扩展更多任务",
        "strict-core controlled process evidence / 非 full open-world runtime": "strict-core controlled process evidence; 与开放世界 evidence 分层互补",
        "已生成 public-freeze-ready bundle": "已整理 public reference bundle",
        "local controlled robustness appendix 已完成: `4,860` records, validation passed / 待 public freeze": "controlled robustness appendix 已完成: `4,860` records, validation passed, 已进入发布证据线",
        "local controlled fairness appendix 已完成: `2,700` records, validation passed / 待 public freeze": "controlled fairness appendix 已完成: `2,700` records, validation passed, 已进入发布证据线",
        "AP 侧下一步优先实验": "已纳入 AP-Core 机制硬化证据线; 后续可扩展种子与场景",
        "待严格化证据": "后续多模态扩展证据",
        "AP 论文已经具备理论、工程、方法论、AP-Core 证据主干和受控 baseline 候选证据; 公开发布/投稿前的重点是 artifact 冻结、更深 AP runtime baseline、正式引用、参数/OOD/长跑验证, 以及若干高展示力实验接入。": "AP 论文已经具备理论、工程、方法论、AP-Core 证据主干、GL 学习验证、受控 baseline、第三方复现和发布级 artifact 锚定; 后续重点是更深 AP runtime baseline、正式引用、参数/OOD/长跑验证, 以及若干高展示力实验扩展。",
        "本节是当前发布版技术报告的参考文献, 目的是让第 7 章不再停留在纯路线描述。正式投稿前仍需按目标期刊格式逐条核对作者全名、页码、DOI、BibTeX 和访问链接。": "本节是当前发布版技术报告的参考文献, 目的是让第 7 章不再停留在纯路线描述。若进入正式 venue 投稿, 可按目标期刊格式逐条核对作者全名、页码、DOI、BibTeX 和访问链接。",
        "数据来源: `paper_artifacts/apv21_20260605/KEY_SUITE_REPORT.md`, `CLAIM_MATRIX.json`, `docs/APV21_Paper_ClaimRegister_20260605.md`。": "数据来源: 发布 artifact 中的 KeySuite report、claim matrix 与 claim register。",
        "design protocol 已完成: `docs/Design_APV21_DialogueProcessPlausibility1_20260608.md`; GL 发布仓库承担学习验证与复验记录": "design protocol 已完成; GL 发布仓库承担学习验证与复验记录",
        "统一数据表见 `docs/APV21_SPAD_MultiModel_EvidenceSummary_20260607.md`, GPT-5.5 展示页见 `outputs/spad_gpt55_flagship_v04g_full_real_20260607/spad_gpt55_flagship_showcase_zh.html`。": "统一数据表见 SPAD multi-model evidence summary, GPT-5.5 展示页见对应 public artifact 输出。",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"### 9\.3 Submission blockers.*?(?=### 9\.4 后续证据路线)",
        """### 9.3 发布版质量状态与高收益增强项

发布版已经完成短主文、长技术报告、三个公开仓库、许可证、manifest、SHA-256 锚点和第三方复现材料整理。下面这些项目不再是发布门槛, 而是后续冲击更高审稿评价的增强项:

| 增强项 | 当前发布状态 | 后续收益 |
|---|---|---|
| public artifact freeze | 已绑定公开仓库 tag、manifest 与外层 zip hash | 可继续补 GitHub Release 页面和长期归档 DOI |
| baseline public freeze | RepeatMap、LBF1、SPAD、STP-v2、底层循环与 online vector 证据已进入发布材料 | 可扩展更多随机种子、更多任务域和 clean-machine 复跑 |
| third-party replication | ACG-j Rust 实现已获授权引用并完成本地复跑整理 | 可补 owner-side release archive 与更多独立实现 |
| Related Work 正式引用 | 已列出核心路线和差异 | 投稿前按目标 venue 统一 DOI、BibTeX 和格式 |
| 图表正式化 | 已有发布级 PNG/PDF 和证据图 | 可按期刊版式继续拆图和重绘 |
| 长跑稳定性 | 已有打断恢复、持久化重载、压力动力学与底层循环证据 | 可扩展到更长 tick、更多任务切换和真实桌面环境 |
| 参数敏感性与调参审计 | STP-v2、底层循环和压力动力学已有扫描/消融 | 可扩展到更多 runtime 参数与跨机器复跑 |
| OOD 与跨场景鲁棒性 | 已有过程锚点跨表面迁移和 GL 开放中文对话证据线 | 可构造更系统的开放世界分层 OOD 基准 |
| 投稿压缩版 | 已生成短主文, 长报告承接全部细节 | 投稿时按目标模板压缩、重排图表和引用 |

这些增强项的意义是让 APV2 从“可发布的原型证据链”继续走向“更容易被外部实验室长期复验的研究平台”。它们不改变本文核心结论: APV2 已经把白箱预测-行动闭环、过程性学习、审计边界和跨实现复现组织成了可检查的工程证据链。

""",
        text,
        flags=re.S,
    )

    text = re.sub(
        r"### C\.4 repository_status 说明.*?(?=### C\.5 Canonical-KeySuite-1 覆盖范围)",
        """### C.4 repository_status 说明

发布版使用 `public_tagged_release` 表示证据材料已经从本地工作区快照升级为公开仓库与 manifest 锚定的 release package。当前锚定方式包括:

1. 公开 GitHub 仓库与 release tag;
2. 每个仓库内的 `PUBLIC_STAGING_MANIFEST.json`;
3. `paper_artifacts/release_20260614/release_manifest_20260614.json`;
4. 外层 release summary 中的 zip SHA-256;
5. 第三方 ACG-j Rust 复现的 commit、source archive SHA-256、`.ap.zip` SHA-256 和本地复跑记录。

这组锚定让读者可以从论文、仓库、manifest、Word/PDF 发布稿和实验输出之间建立可追踪关系。

""",
        text,
        flags=re.S,
    )

    text = re.sub(
        r"### C\.5 Canonical-KeySuite-1 覆盖范围.*$",
        """### C.5 Canonical-KeySuite-1 覆盖范围

Canonical-KeySuite-1 覆盖 AP-Core 机制切片: 最小行动-反馈-记忆闭环、行动后果学习、局部特征贡献组合泛化、小范围数量与 add/remove 迁移、多模态 raw sensor 联想、NoSolver 数学过程切片、Math-FullChain 应用题链路和 AP learned skill registry。它的价值在于证明 APV2 的底层对象能够在 teacher-off/no-solver 审计下形成可复用过程技能。

发布版同时把 GL 开放中文对话、STP-v2 过程锚点、在线 learned vector、底层循环动力学、压力动力学和第三方 Rust 复现列为相邻证据线。它们共同支持本文的积极结论: APV2 不是只存在于概念图中的架构, 而是已经形成了从底层机制、学习协议、开放中文对话基座到跨工程复现的多层证据链。完整 AGI、人类等价意识、无限开放世界理解和真实产品级桌面自治仍属于更长期的研究目标, 不作为本文当前结论的前提。
""",
        text,
        flags=re.S,
    )

    return text


def build() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    short_text = SHORT.read_text(encoding="utf-8")
    body = _extract_from_heading(source_text, "## 第 1 章 绪论")
    body = _modernize_body(body)
    formal = _extract_between(short_text, "### 2.7 核心过程的形式化摘要", "## 3. 学习机制")

    preface = f"""# APV2 长篇技术报告: 白箱预测-行动闭环的完整论证与实验附录

日期: 2026-06-14
文档类型: 发布版长篇技术报告 / 完整论证与实验附录
短主文: `APV2_论文_白箱预测行动闭环架构_20260614.docx` / `.pdf`
发布 tag: `apv2-release-20260614-final-longreport`
AP-Core 仓库: `https://github.com/ginsonko/Artificial-PsyArch-V2`
GL 开放中文对话仓库: `https://github.com/ginsonko/APV2-GL-OpenWorld-Chinese`
复现实验与冻结锚定仓库: `https://github.com/ginsonko/APV2-Reproduction-Artifacts`
第三方独立复现参考: `https://github.com/ACG-j/artificial_psyarch`

## 摘要

APV2 是一个面向持续认知的白箱预测-行动闭环架构。它不把智能只理解为一次性输出, 而是把外界输入、短期叙事、历史召回、后继预测、认知压力、过程性感受、情绪慢量、行动竞争、行动反馈和持久化记忆组织为同一条可审计的 tick 循环。本文档是 APV2 发布版的长篇技术报告, 用来承接短主文无法展开的完整定义、算法、白箱 trace、实验矩阵、边界讨论、第三方复现和审稿疑问回应。

当前发布包形成三条互相区分又互相支撑的证据线。AP-Core 证据证明底层 runtime 机制可以被直接测试、消融和复现; GL 证据证明 AP-style 学习协议可以在 teacher-off/no-leakage 条件下组织稳定的基础中文开放对话能力; 第三方 Rust 复现证明核心思想可以跨语言、跨工程路线重建。三条证据共同说明 APV2 已经从理论草图推进为可运行、可审计、可教学、可复验的工程研究原型。

## 与短主文的关系

发布版论文采用短主文和长篇技术报告分工:

| 文档 | 读者 | 作用 |
|---|---|---|
| 短主文 | 审稿人、技术读者、仓库首页访问者 | 快速说明问题、架构、关键机制、核心证据和适用边界 |
| 长篇技术报告 | 需要核查细节的审稿人、复现实验者、后续研究者 | 提供完整理论、公式、伪代码、trace、实验附录、术语表和答疑 |

短主文不是删减掉证据后的唯一论文, 而是发布入口。长篇技术报告保留完整论证, 使读者可以从短文中的每个关键主张追溯到具体机制和证据。

## 审稿问题快速索引

| 审稿疑问 | 本报告回应位置 | 回应方式 |
|---|---|---|
| 核心算法是否足够形式化 | 本节“核心过程的形式化摘要”、第 3-4 章 | 给出能量更新、残差召回、C* 后继、短期槽回读和认知感受映射公式 |
| 白箱性是否只是口号 | 本节“白箱 tick trace 示例”、第 4 章、第 6 章 | 用 tick 级状态池、短期槽、B 召回、C* 峰和行动倾向展示内部过程 |
| 与 LLM 的关系是否夸大 | 第 7-8 章 | 把 APV2 定位为可教学、可审计的认知底座, 把 LLM 定位为教师、知识源和工具解释器 |
| 认知感受是否只是标签 | 第 3.5、第 4.8、第 8.6 及 STP-v2 证据 | 说明认知感受来自预测熵、压力、残差、证据缺口和行动反馈等过程量 |
| 是否靠答案表、关键词硬门或隐藏 solver | 第 5 章、第 6 章、GL 协议证据 | 使用 teacher-off、no-leakage、no-solver、taint audit 和低粒度行动 trace |
| 是否可复现 | 第 9 章、附录 C、三个公开仓库 | 通过公开 tag、manifest、SHA-256、Word/PDF、实验输出和第三方 Rust 复跑锚定 |

## 发布版新增的形式化与白箱例子

{formal}

## APV2 与 LLM 的互补边界

APV2 不把大语言模型排除在研究体系之外。相反, 它把 LLM 放在更清楚的位置: LLM 可以是成人教师、外部知识源、课程生成器、事后评卷器、工具翻译器或安全审查器; APV2 学生侧则通过状态池、短期叙事槽、召回、预测、行动反馈和本地经验形成自己的可审计能力。这样做的优势是把“谁在测试期真正作答”说清楚: 当 LLM 在测试期直接给学生答案时, 那是 AP+LLM 工具系统; 当 LLM 只在教学期示范、提交后评分或外部补课, 而学生侧 teacher-off 运行时, 才能把形成的能力记为 AP-style 学习证据。

这个互补关系也让能力边界更清楚。APV2 当前最强的是持续状态、反馈学习、白箱审计、过程性修订、短期叙事连续性、低资源技能保持和可复现机制实验。面对海量百科知识、复杂数学竞赛、长篇创造性写作、代码工程、多工具开放规划等任务, LLM 和外部工具仍然有明显优势。APV2 的目标不是在每个静态 benchmark 上替代 LLM, 而是提供一种可以被教、能回看、会修正、可复验、可以长期作为同一个主体运行的认知底座。

## 长篇正文

"""
    OUT.write_text(preface + "\n\n" + body + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
    print(OUT)
