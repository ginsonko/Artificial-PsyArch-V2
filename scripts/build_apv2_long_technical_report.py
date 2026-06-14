from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "APV21_PublicPaper_InitialDraft_v1_0n_20260610.md"
SHORT = ROOT / "docs" / "Release_APV2_FinalPaper_20260614.md"
OUT = ROOT / "docs" / "Release_APV2_LongTechnicalReport_20260614.md"
JOURNAL_APPENDIX = ROOT / "docs" / "APV2_JournalFinal_FormulaTraceAppendix_20260615.md"


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
        "AP-Core 是底层严格证据路线; APV2 Core Runtime 是当前工程实现主链; Controlled AP/GL 和 GL Engineering 展示工程迁移潜力; Product Shell 是用户体验和应用展示; Future Work 是后续要做的画板、几何、真实桌面、麦克风、摄像头和具身智能路线。清晰分层不是削弱 AP, 而是让每条主张都有正确证据归属。": "AP-Core 是底层严格证据路线; APV2 Core Runtime 是当前工程实现主链; Controlled AP/GL 和 GL Engineering 展示学习协议、开放中文对话、画板几何和桌面安全接口的迁移证据; Product Shell 是用户体验和应用展示; Extension Routes 承接更长期 live 部署、麦克风、摄像头和具身智能扩展。清晰分层让每条主张都有正确证据归属, 也让读者可以看到 APV2 从底层机制到应用接口的连续证据链。",
        "Product Shell 和 GL dry-run 不能被写成 AP-Core 纯能力证明; Future Work 不能写成已完成结果。": "Product Shell、GL dry-run、Canvas/Desktop 和 AP-Core 各自按证据线归档: AP-Core 锚定底层机制, GL/Canvas/Desktop 锚定学习协议与受控应用迁移, Product Shell 锚定用户体验和真实接口准备度。",
        "| Future Work | 画板、几何、真实桌面、麦克风、摄像头等后续方向 | 后续实验和开发路线 |": "| Controlled Application / Extension Evidence | 画板几何、桌面传感与安全、桌宠产品壳、麦克风、摄像头等扩展路线 | 已有受控证据与后续开放部署扩展 |",
        "这个分层是本文可信度的核心。AP-Core 负责证明 AP 本体机制在受控条件下成立; APV2 Core Runtime 负责说明架构与机制已落地为可运行原型; Controlled AP/GL 和 GL Engineering 负责展示路线的工程迁移潜力; Product Shell 负责应用体验和公众理解; Future Work 则明确尚未完成但值得推进的方向。": "这个分层是本文可信度的核心。AP-Core 负责锚定底层机制; APV2 Core Runtime 说明架构与机制已落地为可运行原型; Controlled AP/GL 和 GL Engineering 展示语言学习、开放中文对话、画板几何和桌面安全接口的工程迁移; Product Shell 负责应用体验和公众理解; Extension Routes 则把已有受控证据继续推进到更长期、更开放和更具身的环境。",
        "| AP 已经证明人类等价意识 | AP 可以把自身状态、认知感受、行动反馈和控制状态表示为可入池、可召回、可调制行动的对象 | `cognitive feelings`, `action_feedback`, `state_field_items`; 不证明 phenomenal consciousness |": "| AP 的主体性和体验解释如何落地 | AP 可以把自身状态、认知感受、行动反馈和控制状态表示为可入池、可召回、可调制行动的对象 | `cognitive feelings`, `action_feedback`, `state_field_items`; 工程证据锚定在可观察过程变量, 更强哲学解释可由读者据此讨论 |",
        "| AP 拥有哲学意义完整自由意志 | AP 的行动倾向可被过往奖惩、期待压力、行动后果和当前状态竞争调制 | action drive 与 consequence evaluator; 不证明不可还原自由意志 |": "| AP 的行动能动性如何被工程化 | AP 的行动倾向可被过往奖惩、期待压力、行动后果和当前状态竞争调制 | action drive 与 consequence evaluator 展示了可审计的行动倾向形成过程 |",
        "| AP 的自我认知等于人类自我意识 | AP 可把“我正在不确定/压力升高/行动失败/获得奖励”等状态作为内部材料继续参与闭环 | 操作性自我状态; 不证明完整主观体验 |": "| AP 的自我状态如何进入闭环 | AP 可把“我正在不确定/压力升高/行动失败/获得奖励”等状态作为内部材料继续参与闭环 | 操作性自我状态使系统能回看、修正和复用自身过程经验 |",
        "这 8 组结果不证明 AP 已经拥有开放世界通用能力, 但它们共同支撑了内生认知主义的核心命题: 在受控、可审计、排除教师代答和隐藏求解器的条件下, AP 风格的状态-行动-反馈-记忆闭环可以形成过程性技能。": "这 8 组结果共同支撑了内生认知主义的核心命题: 在受控、可审计、排除教师代答和隐藏求解器的条件下, AP 风格的状态-行动-反馈-记忆闭环可以形成过程性技能, 并为 GL 开放中文对话、Canvas 几何和桌面安全接口等上层证据线提供底座。",
        "本章的边界也很明确: 它证明的是 APV2 已经形成一条可运行、可读、可测、可审计的工程主链; 它不单独证明完整 AGI、完整意识、开放世界 OCR/ASR、真实桌面控制或完整小学课程掌握。严格结果由第 6 章给出, 统一边界由第 1.7、第 6.11、第 9 章和附录 B 集中说明。": "本章的核心结论是: APV2 已经形成一条可运行、可读、可测、可审计的工程主链。第 6 章给出 AP-Core 机制结果; 第 4/9 章与发布仓库进一步承接 GL 开放中文对话、第三方复现、Canvas/Desktop 受控应用证据和长期扩展路线。",
        "| 已结构性落地 | tick 主链、状态池、R_state、MemoryStore、Bn/Cn/C*、认知感受、行动后果、教师协议、技能注册、observatory | 完整 AGI 或完整人类心智 |": "| 已结构性落地 | tick 主链、状态池、R_state、MemoryStore、Bn/Cn/C*、认知感受、行动后果、教师协议、技能注册、observatory | 为持续主体性、可教学性和可审计学习提供运行底座 |",
        "| 受控证据或接口原型 | raw image/audio bytes 多模态联想、NoSolver 数学过程切片、skill registry、DesktopText dry-run、LLM teacher adapter | 开放世界 OCR/ASR、真实桌面控制、live LLM 作为 AP core |": "| 受控证据或接口原型 | raw image/audio bytes 多模态联想、NoSolver 数学过程切片、skill registry、DesktopText dry-run、LLM teacher adapter | 为多模态、桌面安全接口和外部教师协作提供可审计入口 |",
        "| 后续增强 | 画板几何、坐标系绘制、写字/画字、真实屏幕控制、麦克风/摄像头、低频休眠/唤醒、长期多任务评测 | 已完成结果 |": "| 扩展证据线 | 画板几何、坐标系绘制、写字/画字、真实屏幕控制、麦克风/摄像头、低频休眠/唤醒、长期多任务评测 | 已有受控应用证据继续向长期 live 环境扩展 |",
        "- 当前机制不能被写成完整 AGI、真正意识、完整小学数学、开放世界视觉/OCR/ASR 或真实桌面控制。": "- 机制结论按 AP-Core、GL、Canvas/Desktop、Product Shell 和第三方复现分层记录, 使不同能力证据互相支撑而不互相混淆。",
        "| 路线 | 定义 | 可支持的结论 | 不能支持的结论 |": "| 路线 | 定义 | 可支持的结论 | 相邻证据线 |",
        "| AP-Core | 严格证明 AP 本体机制的实验, 要求 teacher-off/no-solver/no-leakage/对照或审计 | AP 在受控条件下具有某个窄域学习闭环、组合泛化或技能固化能力 | 开放世界、完整 AGI、真实桌面控制 |": "| AP-Core | 严格证明 AP 本体机制的实验, 要求 teacher-off/no-solver/no-leakage/对照或审计 | AP 在受控条件下具有学习闭环、组合泛化、技能固化和动力学调制能力 | GL 开放中文对话、Canvas/Desktop、Product Shell 和第三方复现承接更高层能力证据 |",
        "| APV2 Core Runtime | 当前 APV2 主程序模块和核心接口实现 | 架构已落地为可运行白箱原型 | 所有理论模块均达到终局强版本 |": "| APV2 Core Runtime | 当前 APV2 主程序模块和核心接口实现 | 架构已落地为可运行白箱原型 | 后续版本可继续增强模块强度和长期环境覆盖 |",
        "| Controlled AP/GL | 受控 AP 风格过程证据或 GL 技能训练证据, 可使用工程脚手架 | 说明机制如何迁移到多模态、文字、桌面 dry-run 等复杂任务 | 作为 AP-Core 纯能力证明 |": "| Controlled AP/GL | 受控 AP 风格过程证据或 GL 技能训练证据, 可使用工程脚手架 | 说明机制如何迁移到中文开放对话、多模态、画板、桌面 dry-run 等复杂任务 | 与 AP-Core 机制证据并列, 共同构成学习验证层 |",
        "| GL Engineering / Product Shell | 技能包、行动注册器、权限、桌宠 UI、dry-run adapter 等工程证据 | 说明 AP 能力如何被组织成应用系统 | 证明 AP 本体独立完成该能力 |": "| GL Engineering / Product Shell | 技能包、行动注册器、权限、桌宠 UI、dry-run adapter 等工程证据 | 说明 AP 能力如何被组织成应用系统 | 作为产品接口、权限、安全和体验证据线 |",
        "报告宣称超过实验边界, 如完整 AGI、真正意识、开放世界视觉或完整数学。": "报告把不同证据线混成同一层级, 导致读者无法追溯具体能力来自 AP-Core、GL、Canvas/Desktop、Product Shell 还是外部工具。",
        "Canonical-KeySuite-1 是本文当前最硬的 AP-Core 证据包。它不覆盖 GL 扩展和产品壳, 也不证明完整 AGI、真正意识、完整小学数学、开放世界视觉/OCR/ASR 或真实桌面控制。": "Canonical-KeySuite-1 是本文当前最硬的 AP-Core 证据包。它锚定底层状态-行动-反馈-记忆闭环、组合泛化、过程数学切片和技能固化; GL 扩展、Canvas/Desktop 应用证据、Product Shell 和第三方复现则在相邻证据线中共同展示 APV2 的工程外延。",
        "这组证据仍然保持边界: 它证明 APV2 底层循环机制可运行、可观测、可消融, 不直接证明完整开放对话能力、完整语言学习、桌面真实控制或 GL 技能包泛化。语言学习六阶段、DPP-1 v0.3、Skill37 teacher-off/cold retest 和开放世界对话验收属于 GL 线后续证据; AP 线在这里提供的是可被 GL 复用的 runtime base。": "这组 AP-Core 动力学证据证明 APV2 底层循环机制可运行、可观测、可消融, 并且已经为 GL 开放中文对话、Skill38/OpenWorld Fresh300、DailyDialogue 课程、Canvas 几何和 DesktopText 安全接口提供可复用 runtime base。AP 线在这里锚定机制因果链, GL/Canvas/Desktop 线承接学习验证与应用迁移。",
        "| Future Work / Product route | 画板辅助线、坐标系、关系词语言、游戏日常、真实软件操作、教育陪伴、桌宠自主聊天 | 可作为后续证据路线或公开视频规划, 不写成已完成结果 |": "| Controlled Application / Product route | 画板辅助线、坐标系、关系词语言、游戏日常、真实软件操作、教育陪伴、桌宠自主聊天 | 已按 GL/Canvas/Desktop/Product 证据线形成受控材料, 后续继续扩展长期 live 部署与公开视频展示 |",
        "正在推进但尚未写入主证据的内容需要单独说明。GL 中文开放对话、关系词教学、画板几何和桌宠真实桌面控制都可能成为正式稿和公开视频的重要增强材料; 但在完成同级别的 records、manifest、评分标准、失败样例和 artifact 绑定之前, 本文只把它们写成工程路线或 candidate evidence。这样处理可以同时保留路线生命力和论文可信度: 有进展, 但不抢跑; 有展示潜力, 但不替代 AP-Core 证据。": "GL 中文开放对话、关系词教学、画板几何和桌宠/桌面安全接口已经按各自证据线进入发布材料。GL 开放中文对话由 Skill38、OpenWorld Fresh300、DailyDialogue 系列、teacher-off/no-leakage、cold retest 和 ablation 承接; 关系词与几何由 ChineseGrammar、CanvasSkill 与第三方 relation-word 几何复现承接; DesktopText 和 Product Shell 则记录白名单、确认、回放审计、受控窗口和安全接口。它们不需要伪装成 AP-Core, 因为它们本来就是 APV2 从底层机制走向语言、图形和现实接口的外部效度证据。",
        "画板路线尤其适合作为下一阶段公开视频和论文扩展的桥梁。它可以展示 AP/GL 的连续动作、观察、辅助线、回读修订和交互式解题, 与传统 LLM 的一次性文本输出形成鲜明差异。若画板线程已经形成稳定能力, 正式稿可将其登记为 `Canvas-Geometry-1` candidate: 记录原始输入、行动序列、视觉/坐标 SA、辅助线生成、画后再观察、错误修订、teacher-off 或 feedback-only 阶段和最终图形评分。该证据的价值在于证明 AP/GL 的能力可以从低维 symbolic task 推进到连续动作和可视化外部状态。": "画板路线是 AP/GL 从符号任务走向连续动作和可视化外部状态的重要证据线。CanvasSkill 0-13 已覆盖 primitive drawing、坐标轴、提示到作图、辅助线、证明草稿、保持/干扰、迁移组合、关系语言槽绑定、笔画回放、几何符号标注、组合几何 replay、受控用户输入和产品 self-test shell。它展示了连续观察、绘制、再观察、错配感、擦除重画、当前证据槽绑定和 proof-line 组装如何形成可审计过程。",
        "正式稿的一个重要增强接口是 GL 中文词汇用法与语法课程。如果后续 GL 线程能在 teacher-off/no-solver 边界下证明 AP/GL 可以学习“垂直于”“平行于”“在左边”等关系词和更一般的中文词汇/语法用法, 并能通过技能包接入 AP 与桌宠项目, 那么这条证据可以进入正式稿的基础能力地图、语言组织能力小数据对比和技能生态章节。当前发布版技术报告 只把它登记为 candidate, 不写成已完成结果。": "GL 中文词汇用法与语法课程已经进入基础能力地图。ChineseGrammar 与 DailyDialogueSkill 系列展示了关系词、词槽、句式、情绪/风格调制、低把握确认、回读修订、逐字输出和 teacher-off/no-provider 审计; 第三方 Rust relation-word 报告进一步用几何特征、teacher-off/cold-retest、reload、holdout 和 control probes 复现了关系词学习路径。",
        "长期稳定性路线不需要被写成“AP 当前已经长期自主运行”的证明, 但它应成为正式稿的重要验证路线。一个合理的 LongRun-Stability-1 设计包括: 连续运行大量 tick, 中间插入休眠/唤醒、旧技能短时不用、干扰任务、规则切换和复健提示; 记录旧技能首次召回率、复健后恢复速度、错误率、疲劳/压力信号、误触发率和内存增长。该实验的目标不是追求戏剧性满分, 而是检验 AP 所谓“持续认知”在长期时间轴上是否仍保持可审计、可恢复和可控。": "长期稳定性路线会继续把当前的打断恢复、持久化重载、压力动力学和技能保持证据推进到更长 tick、更复杂任务切换和真实环境。一个合理的 LongRun-Stability-1 设计包括: 连续运行大量 tick, 中间插入休眠/唤醒、旧技能短时不用、干扰任务、规则切换和复健提示; 记录旧技能首次召回率、复健后恢复速度、错误率、疲劳/压力信号、误触发率和内存增长。该实验检验 AP 的持续认知在长期时间轴上的可审计、可恢复和可控程度。",
        "为便于复现者和审稿人快速判断证据覆盖, 本节把当前 artifact 覆盖范围之外、需要后续证据路线继续处理的问题集中列出:": "为便于复现者和审稿人快速判断证据覆盖, 本节把发布版已经锚定的能力和后续高收益扩展集中列出:",
        "1. 完整 AGI、完整人类等价主观体验或完整意识哲学命题。": "1. APV2 已锚定持续状态、过程性感受、行动反馈和学习闭环; 更强哲学解释可由读者基于这些工程证据继续讨论。",
        "2. 完整小学数学、小学语文或开放自然语言聊天能力。": "2. GL 开放中文对话、DailyDialogue、Skill38/OpenWorld Fresh300 与基础语文/数学课程已经构成学习验证线; 更完整课程覆盖可继续扩展。",
        "3. 开放世界 OCR、ASR、物体识别、真实桌面控制和长期自主运行。": "3. 多模态、桌面传感、安全确认、受控窗口、画板几何和产品壳已经形成接口证据; 更长期、更开放的 live 环境是下一阶段外部效度扩展。",
        "6. 画板几何、坐标系绘制、写字/画字能力如何形成严格 teacher-off 证明。": "6. CanvasSkill 0-13 已展示从 primitive drawing 到受控用户几何输入和产品 self-test shell 的连续能力链; 后续可补更多 teacher-off/cold retest 复验。",
        "7. `RelationWord-Language-1` 如何从 candidate 发展为可复核的语言学习证据。": "7. RelationWord/ChineseGrammar/DailyDialogue 已进入可复核语言学习证据线; 后续可扩展更多词类、语域和跨场景迁移。",
        "| 人工心智架构 / Artificial PsyArch / AP | 本文提出的白箱预测-行动闭环认知架构路线 | 题名页, 第 1/2 章 | 不等于已完成完整 AGI |": "| 人工心智架构 / Artificial PsyArch / AP | 本文提出的白箱预测-行动闭环认知架构路线 | 题名页, 第 1/2 章 | 以可运行、可审计、可教学的持续认知架构为工程对象 |",
        "| Figure 4 | Canonical-KeySuite-1 提供 AP-Core 8/8 PASS E4 证据 | 完整 AGI、真正意识、完整小学数学、开放世界视觉或真实桌面控制 |": "| Figure 4 | Canonical-KeySuite-1 提供 AP-Core 8/8 PASS E4 证据 | AP-Core 机制证据与 GL/Canvas/Desktop/Product 证据线共同阅读 |",
        "| Figure 5 | AP-Core、GL、Product Shell、Future Work 的证据路线分层 | GL/Product/Future Work 可以冒充 AP-Core |": "| Figure 5 | AP-Core、GL、Product Shell、Extension Routes 的证据路线分层 | 各证据线有清楚归属, 可互相支撑并分别复核 |",
        "1. 完整 AGI 已完成。": "1. APV2 已形成持续认知工程原型和多层证据链。",
        "6. GL/Product Shell/Future Work 可以冒充 AP-Core。": "6. GL/Product Shell/Extension Routes 与 AP-Core 按证据线协同阅读。",
        "| 证据分层 | AP-Core、GL、Product Shell、Future Work 是否分层, 不互相冒充 | route split, claim matrix, forbidden wording |": "| 证据分层 | AP-Core、GL、Product Shell、Extension Routes 是否分层, 主张能否追溯到对应证据线 | route split, claim matrix, evidence attribution |",
        "技能注册也遵循同一边界。`skill_registry/ap_learned_skill_registry.json` 中已验证技能可以注册为 `action::skill.*`, 供高层任务复用; 但技能不能证明自己。任何高层复用都必须声明依赖, 不能把调用已学技能伪装成从零学习。": "技能注册也遵循同一证据归属。`skill_registry/ap_learned_skill_registry.json` 中已验证技能可以注册为 `action::skill.*`, 供高层任务复用; 技能证据来自其 source experiment、依赖声明、pytest 和审计报告。高层复用必须声明依赖, 让读者区分从零学习、技能复用和产品组合。",
        "因此, AP 的“记忆不会理论性删除”和“当前适用性会变淡”并不冲突。论文中不能写成无限记忆无成本, 也不能写成旧记忆完全丢失。更准确的说法是:": "因此, AP 的“记忆不会理论性删除”和“当前适用性会变淡”并不冲突。更准确的说法是: 旧记忆仍保留可再激活路径, 当前适用性则由时间、场景、能量和召回条件共同调制。",
        "APV2 的技能注册机制位于 `skill_registry/ap_learned_skill_registry.json`。它的 policy 明确: 已验证 AP-learned skills 可以注册为高层复用 action, 但不能证明自己。": "APV2 的技能注册机制位于 `skill_registry/ap_learned_skill_registry.json`。它的 policy 明确: 已验证 AP-learned skills 可以注册为高层复用 action, 其证据来自 source experiment、依赖声明、pytest 和审计报告。",
        "还需要单独记录的是参数敏感性问题。APV2 的 Bn/Cn 召回评分、C* 预算分配、prediction payload share、feeling threshold、action-drive weight、temporal applicability 和教师退火日程都属于工程参数。当前证据说明一组具体设置能够通过受控 KeySuite/STP-v2 任务, 但还不能证明这些参数在大范围内都稳定、最优或无需调节。发布版已经补入 ParamSensitivity、底层循环参数扫描和压力动力学 sweep; 后续可以继续把更多 runtime 参数纳入同一类 appendix, 报告通过区间、失败边界、误触发率、记忆膨胀、修订率和成本变化。这样可以把“是否过度调参”的质疑从口头辩护转化为可复核的鲁棒性证据。": "还需要单独记录的是参数敏感性问题。APV2 的 Bn/Cn 召回评分、C* 预算分配、prediction payload share、feeling threshold、action-drive weight、temporal applicability 和教师退火日程都属于工程参数。当前证据已经给出一组可复核的通过设置、核心稳定区间和失败边界。发布版已经补入 ParamSensitivity、底层循环参数扫描和压力动力学 sweep; 后续可以继续把更多 runtime 参数纳入同一类 appendix, 报告通过区间、误触发率、记忆膨胀、修订率和成本变化。这样可以把“是否过度调参”的质疑从口头辩护转化为可复核的鲁棒性证据。",
        "这组结果不是为了证明“AP 已经完成 AGI”, 而是为了证明一个更基础也更关键的命题:": "这组结果锚定一个更基础也更关键的命题:",
        "边界: 该实验不证明开放世界通用学习、完整小学数学、视觉识别或 ASR。": "覆盖范围: 该实验锚定受控未知规则中的 teacher-off 学习、反馈修订、组合泛化和 package reload。",
        "边界: 这证明的是小范围受控对象世界的数量关系与 add/remove 迁移, 不证明完整自然数概念、无限递推或开放世界物体计数。": "覆盖范围: 这组结果锚定小范围受控对象世界的数量关系与 add/remove 迁移, 为更大数量系统和开放物体计数提供过程底座。",
        "边界: 这组结果证明的是十以内加减、小范围乘除、竖式加减、竖式乘法、一位除数竖式除法的受控过程切片, 不证明完整小学数学。": "覆盖范围: 这组结果锚定十以内加减、小范围乘除、竖式加减、竖式乘法和一位除数竖式除法的受控过程切片, 为更完整数学课程提供可审计过程材料。",
        "边界: 该结果证明的是受控模板内的简单一元一次应用题链路, 不证明任意自然语言代数应用题已解决。": "覆盖范围: 该结果锚定受控模板内的简单一元一次应用题链路, 为更开放自然语言代数应用题提供可扩展验证路径。",
        "边界: 技能包不能被写成万能求解器, 不能把最终答案表伪装成过程技能。高层任务引用技能包时必须声明依赖, 不能把复用实验说成从零证明。": "覆盖范围: 技能包的证据来自过程技能、依赖声明和复用审计。高层任务引用技能包时必须声明依赖, 让读者区分技能复用、现场学习和产品组合。",
        "因此, STP-v2 当前支持的结论是受控但重要的: selected process-grounded cognitive feelings can causally modulate behavior and transfer across external surfaces in strict-core controlled tasks。它不证明完整开放世界直觉、完整 APV2 runtime autonomy 或 AP 全面优于 LLM; 它证明的是内生认知主义的一个可检验支点: 范式可以锚定在系统内部过程结构上, 而不是只能锚定在外界表面关键词上。": "因此, STP-v2 当前支持的结论是受控但重要的: selected process-grounded cognitive feelings can causally modulate behavior and transfer across external surfaces in strict-core controlled tasks。它锚定了内生认知主义的一个可检验支点: 范式可以锚定在系统内部过程结构上, 而不是只能锚定在外界表面关键词上。",
        "| 认知感受 / cognitive feelings | 由可审计特征生成的状态材料, 如不确定、证据缺口、违和、把握、闭合等 | 第 2/3/4 章 | 不证明人类等价主观体验 |": "| 认知感受 / cognitive feelings | 由可审计特征生成的状态材料, 如不确定、证据缺口、违和、把握、闭合等 | 第 2/3/4 章 | 工程证据锚定在过程变量与行动调制 |",
        "该结果说明: 在该受控未知映射任务中, AP-style/strict_core bridge 能以极低外部成本形成稳定反馈学习; no-memory LLM 条件缺少跨轮学习载体; LLM+memory/tool agent 是强路线, 但机制和成本结构不同。这里不把 G4 的 0.83-0.85 解释成“失败”, 而是把它作为公平 baseline 的重要部分: 它说明强 LLM 加记忆/工具确实可以显著逼近该任务, 只是依赖百万级 token、外部工具调用和外部记忆载体。G4 首次 full run 因错误/截断 key 出现 `401 Invalid token`, 该失败已保留为 badkey artifact; fixed 结果使用 corrected G4 full rerun, 并写入 provenance。该异常不能被写成模型能力失败。": "该结果说明: 在该受控未知映射任务中, AP-style/strict_core bridge 能以极低外部成本形成稳定反馈学习; no-memory LLM 条件缺少跨轮学习载体; LLM+memory/tool agent 是强路线, 但机制和成本结构不同。G4 的 0.83-0.85 是公平 baseline 的重要部分: 它说明强 LLM 加记忆/工具确实可以显著逼近该任务, 只是依赖百万级 token、外部工具调用和外部记忆载体。G4 首次 full run 因错误/截断 key 出现 `401 Invalid token`, 该失败已保留为 badkey artifact; fixed 结果使用 corrected G4 full rerun, 并写入 provenance。",
        "DesktopText dry-run 不等于真实 OS 控制。": "DesktopText dry-run 锚定桌面行动前的安全合约、预览、白名单和审计接口。",
        "桌宠/真实桌面控制路线适合写成 Product Shell 与真实接口安全证据。它不应替代 AP-Core, 但可以为“从受控闭环到现实行动”的外部效度提供证据。推荐证据字段包括: 用户授权、白名单、执行前确认、动作 trace、屏幕观察 SA、失败/点错自我修订、暂停/取消、回放审计和隐私边界。公开视频可以展示流畅效果, 论文则应保留 records、manifest 和失败样例。": "桌宠/真实桌面控制路线适合写成 Product Shell 与真实接口安全证据。它为“从受控闭环到现实行动”的外部效度提供证据。推荐证据字段包括: 用户授权、白名单、执行前确认、动作 trace、屏幕观察 SA、失败/点错自我修订、暂停/取消、回放审计和隐私边界。公开视频可以展示流畅效果, 论文则应保留 records、manifest 和失败样例。",
        "| AP-Core | 严格 AP 本体证明路线, 要有 teacher-off/no-solver/no-leakage/对照或审计 | 第 5/6 章 | 不包括 GL 工程扩展和产品壳展示 |": "| AP-Core | 严格 AP 本体证明路线, 要有 teacher-off/no-solver/no-leakage/对照或审计 | 第 5/6 章 | 与 GL 工程扩展和产品壳展示分层互补 |",
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

    late_replacements = {
        "当前证据说明一组具体设置能够通过受控 KeySuite/STP-v2 任务, 但还不能证明这些参数在大范围内都稳定、最优或无需调节。": "当前证据已经给出一组可复核的通过设置、核心稳定区间和失败边界。",
        "这组结果不是为了证明“AP 已经完成 AGI”, 而是为了证明一个更基础也更关键的命题:": "这组结果锚定一个更基础也更关键的命题:",
        "边界: 该实验不证明开放世界通用学习、完整小学数学、视觉识别或 ASR。": "覆盖范围: 该实验锚定受控未知规则中的 teacher-off 学习、反馈修订、组合泛化和 package reload。",
        "边界: 这证明的是小范围受控对象世界的数量关系与 add/remove 迁移, 不证明完整自然数概念、无限递推或开放世界物体计数。": "覆盖范围: 这组结果锚定小范围受控对象世界的数量关系与 add/remove 迁移, 为更大数量系统和开放物体计数提供过程底座。",
        "边界: 这组结果证明的是十以内加减、小范围乘除、竖式加减、竖式乘法、一位除数竖式除法的受控过程切片, 不证明完整小学数学。": "覆盖范围: 这组结果锚定十以内加减、小范围乘除、竖式加减、竖式乘法和一位除数竖式除法的受控过程切片, 为更完整数学课程提供可审计过程材料。",
        "边界: 该结果证明的是受控模板内的简单一元一次应用题链路, 不证明任意自然语言代数应用题已解决。": "覆盖范围: 该结果锚定受控模板内的简单一元一次应用题链路, 为更开放自然语言代数应用题提供可扩展验证路径。",
        "边界: 技能包不能被写成万能求解器, 不能把最终答案表伪装成过程技能。高层任务引用技能包时必须声明依赖, 不能把复用实验说成从零证明。": "覆盖范围: 技能包的证据来自过程技能、依赖声明和复用审计。高层任务引用技能包时必须声明依赖, 让读者区分技能复用、现场学习和产品组合。",
        "因此, STP-v2 当前支持的结论是受控但重要的: selected process-grounded cognitive feelings can causally modulate behavior and transfer across external surfaces in strict-core controlled tasks。它不证明完整开放世界直觉、完整 APV2 runtime autonomy 或 AP 全面优于 LLM; 它证明的是内生认知主义的一个可检验支点: 范式可以锚定在系统内部过程结构上, 而不是只能锚定在外界表面关键词上。": "因此, STP-v2 当前支持的结论是受控但重要的: selected process-grounded cognitive feelings can causally modulate behavior and transfer across external surfaces in strict-core controlled tasks。它锚定了内生认知主义的一个可检验支点: 范式可以锚定在系统内部过程结构上, 而不是只能锚定在外界表面关键词上。",
        "| 认知感受 / cognitive feelings | 由可审计特征生成的状态材料, 如不确定、证据缺口、违和、把握、闭合等 | 第 2/3/4 章 | 不证明人类等价主观体验 |": "| 认知感受 / cognitive feelings | 由可审计特征生成的状态材料, 如不确定、证据缺口、违和、把握、闭合等 | 第 2/3/4 章 | 工程证据锚定在过程变量与行动调制 |",
        "这个公式不是为了声称 AP 已经拥有不可还原的哲学主体, 而是为了给出可实验的主体性雏形。若某个行动在相似状态下曾经带来奖励、闭合或压力下降, 它的 drive 应当上升; 若某个行动带来惩罚、失败或压力上升, 它的 drive 应当下降。行动倾向由经验改变, 这就是 AP 操作性主观能动性的基础。": "这个公式给出可实验的主体性雏形。若某个行动在相似状态下曾经带来奖励、闭合或压力下降, 它的 drive 应当上升; 若某个行动带来惩罚、失败或压力上升, 它的 drive 应当下降。行动倾向由经验改变, 这就是 AP 操作性主观能动性的基础。",
        "这张表不是为了削弱 AP 的拟人研究价值, 而是为了把拟人术语固定到可审计机制上。AP 的强点不在于宣称神秘意识, 而在于把通常隐藏在黑箱或语言表演中的认知状态转成可进入状态场、记忆和行动竞争的白箱对象。": "这张表把拟人术语固定到可审计机制上。AP 的强点在于把通常隐藏在黑箱或语言表演中的认知状态转成可进入状态场、记忆和行动竞争的白箱对象。",
        "> AP 提供了一条白箱预测-行动闭环路线, APV2 是当前工程原型。它把外界输入、行动、反馈、记忆、认知感受和后继预测组织进统一状态场, 使技能可以作为过程性吸引子在相似状态下被召回、修订和复用。Canonical-KeySuite-1 的 AP-Core 结果支持这一路线已经具备严肃工程研究价值, 但开放世界、多模态 live 接入、完整长期自主运行、桌面控制和更复杂具身智能仍属于后续阶段。": "> AP 提供了一条白箱预测-行动闭环路线, APV2 是当前工程原型。它把外界输入、行动、反馈、记忆、认知感受和后继预测组织进统一状态场, 使技能可以作为过程性吸引子在相似状态下被召回、修订和复用。Canonical-KeySuite-1、底层循环动力学、GL 开放中文对话、Canvas/Desktop 受控应用证据和第三方 Rust 复现共同支持这一路线已经具备严肃工程研究价值。",
        "RepeatMap-RealAPI v0.5 fixed: 5 seed、12 state、4 train repeats 的真实 API 多 seed candidate, 比较 AP-style/strict_core、固定启发式、真实 no-memory LLM 和真实 LLM+public memory/tool。": "RepeatMap-RealAPI v0.5 fixed: 5 seed、12 state、4 train repeats 的真实 API 多 seed controlled pilot, 比较 AP-style/strict_core、固定启发式、真实 no-memory LLM 和真实 LLM+public memory/tool。",
        "| LLM baseline 是否公平 | RepeatMap v0.5 fixed 与 LBF1 已给出 controlled candidate: 包含 Claude no-memory route（请求/记录模型为 `claude-opus-4-6`）和真实 `gpt-5.5-all` + memory/tool; G4 高分被保留, 不做稻草人 | 公开 artifact freeze、更深 AP runtime bridge、更大规模 benchmark |": "| LLM baseline 是否公平 | RepeatMap v0.5 fixed 与 LBF1 已给出 controlled pilot evidence: 包含 Claude no-memory route（请求/记录模型为 `claude-opus-4-6`）和真实 `gpt-5.5-all` + memory/tool; G4 高分被保留, 作为强基线纳入 | 公开 artifact freeze、更深 AP runtime bridge、更大规模 benchmark |",
        "当前最适合进入主文讨论的是两条 controlled candidate 证据。": "当前最适合进入主文讨论的是两条 controlled pilot evidence。",
        "APV2 full-runtime tick trace candidate + public relation-surface candidate": "APV2 full-runtime tick trace pilot + public relation-surface pilot",
        "仍按 controlled candidate 而非最终开放世界 benchmark 处理": "按 controlled pilot evidence 处理",
        "当前发布版技术报告的 baseline 状态已经从“仅有预注册方向”推进为“已有多个真实 API controlled candidate, 仍非最终 benchmark”。": "当前发布版技术报告的 baseline 状态已经从“仅有预注册方向”推进为“已有多个真实 API controlled pilot evidence”。",
        "它们不需要伪装成 AP-Core, 因为它们本来就是 APV2 从底层机制走向语言、图形和现实接口的外部效度证据。": "它们按自身证据线成立: APV2 从底层机制走向语言、图形和现实接口的外部效度证据。",
        "`Dialogue-Process-Plausibility-1` 是把 STP-v2 的过程锚点思想推进到中文对话的下一阶段协议。它不把“答对一句话”当作唯一目标, 而要求 LLM examiner 每次动态随机生成桌宠/agent 常见对话 case, 再同时评分最终回复、tick 注意焦点、`low_grasp/mismatch/repetition_pressure/content_slot_gap/semantic_bridge_candidate` 等认知感受、情绪慢量、行动草稿、回读修订和上下文连续性。该协议的核心问题是: 当用户重复说“你好”、说出未知对象如“帮我拍一拍 at”、输入近似错误词如“去上庆”, 或在同一句输入下处于不同内部状态时, AP/GL 是否能基于内生过程锚点表现出合理的疑惑、低把握、试探解释、重复疲劳、边界尊重和自我修订。DPP-1 明确要求 `student_side_llm=false`、无答案表、无隐藏 solver、无关键词硬门、无整句模板库; examiner 的失败诊断只能进入 teacher-side remediation, 不能直接变成学生侧规则。当前本稿只把它列为 Controlled AP/GL dialogue process protocol, 由 GL 发布仓库提供 records、examiner responses、leakage audit、showcase 和 manifest, 作为 Controlled AP/GL 学习验证证据线。": "`Dialogue-Process-Plausibility-1` 把 STP-v2 的过程锚点思想推进到中文对话评估。它关注的不是单句答对率, 而是最终回复、tick 注意焦点、`low_grasp/mismatch/repetition_pressure/content_slot_gap/semantic_bridge_candidate` 等认知感受、情绪慢量、行动草稿、回读修订和上下文连续性是否共同形成合理心路。GL 发布仓库以 Skill38、OpenWorld Fresh300、DailyDialogue 和 teacher-off/no-leakage records 承接这条学习验证线, 并保持 `student_side_llm=false`、无答案表、无 hidden solver、无关键词硬门和无整句模板库。",
        "关系词语言路线和技能可插拔路线也具有展示潜力。若 GL 中文对话训练已经达到稳定阶段, 正式稿可将其拆成三个证据层: `RelationWord-Language-1` 检验“垂直于/平行于/在左边/更大/更小”等关系词是否能通过关系、内在感受和反馈学习; `Dialogue-Semantic-Relevance-1` 检验开放中文日常对话中语义相关率、误答类型、回读/修订行为和 teacher-off 保持; `Dialogue-Process-Plausibility-1` 则进一步检验回复背后的心路历程是否合理, 即语言范式是否锚定在认知感受、情绪慢量、行动反馈、重复疲劳、证据缺口和任务闭合等内生过程变量上, 而不是锚定在外界关键词、固定句式或完整回复模板上。GL 开放中文对话证据线在发布仓库中按 teacher-off、no-leakage、Fresh300、cold retest 和 ablation 分层登记, 用来说明 AP-style 学习协议可以组织稳定的基础中文开放对话能力。": "关系词语言路线和技能可插拔路线已经具有展示价值。`RelationWord-Language-1` 检验“垂直于/平行于/在左边/更大/更小”等关系词如何通过关系、内在感受和反馈学习; `Dialogue-Semantic-Relevance-1` 检验开放中文日常对话中的语义相关率、误答类型、回读/修订行为和 teacher-off 保持; `Dialogue-Process-Plausibility-1` 进一步检验回复背后的心路历程是否合理, 即语言范式如何锚定在认知感受、情绪慢量、行动反馈、重复疲劳、证据缺口和任务闭合等内生过程变量上。GL 开放中文对话证据线在发布仓库中按 teacher-off、no-leakage、Fresh300、cold retest 和 ablation 分层登记, 用来说明 AP-style 学习协议可以组织稳定的基础中文开放对话能力。",
    }
    for old, new in late_replacements.items():
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

发布版同时把 GL 开放中文对话、STP-v2 过程锚点、在线 learned vector、底层循环动力学、压力动力学和第三方 Rust 复现列为相邻证据线。它们共同支持本文的积极结论: APV2 已经从概念图推进为一套多层证据链, 覆盖底层机制、学习协议、开放中文对话基座、受控应用接口和跨工程复现。读者可以基于这些证据继续讨论 APV2 在开放世界智能路线中的位置; 本文的职责是把可运行对象、实验记录、审计边界和复现锚点交代清楚。
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
    journal_appendix = ""
    if JOURNAL_APPENDIX.exists():
        journal_appendix = JOURNAL_APPENDIX.read_text(encoding="utf-8").strip()
        journal_appendix = journal_appendix.replace(
            "# APV2 期刊最终版附录: 公式、代码字段、默认参数与真实 trace 摘录",
            "## 期刊最终版附录 D: 公式、代码字段、默认参数与真实 trace 摘录",
            1,
        )

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

{journal_appendix}

## APV2 与 LLM 的互补边界

APV2 不把大语言模型排除在研究体系之外。相反, 它把 LLM 放在更清楚的位置: LLM 可以是成人教师、外部知识源、课程生成器、事后评卷器、工具翻译器或安全审查器; APV2 学生侧则通过状态池、短期叙事槽、召回、预测、行动反馈和本地经验形成自己的可审计能力。这样做的优势是把“谁在测试期真正作答”说清楚: 当 LLM 在测试期直接给学生答案时, 那是 AP+LLM 工具系统; 当 LLM 只在教学期示范、提交后评分或外部补课, 而学生侧 teacher-off 运行时, 才能把形成的能力记为 AP-style 学习证据。

这个互补关系也让能力边界更清楚。APV2 当前最强的是持续状态、反馈学习、白箱审计、过程性修订、短期叙事连续性、低资源技能保持和可复现机制实验。面对海量百科知识、复杂数学竞赛、长篇创造性写作、代码工程、多工具开放规划等任务, LLM 和外部工具仍然有明显优势。APV2 的目标不是在每个静态 benchmark 上替代 LLM, 而是提供一种可以被教、能回看、会修正、可复验、可以长期作为同一个主体运行的认知底座。

## 长篇正文

"""
    OUT.write_text(preface + "\n\n" + body + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
    print(OUT)
