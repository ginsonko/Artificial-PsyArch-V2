# TransitionIsolation-1 报告

结论: 通过

## 关键检查

```json
{
  "direct_isolation_probe_passed": true,
  "recall_successor_probe_passed": true
}
```

## 直接隔离 (A -> B transition 训练前后)

| 量 | 训练前 | 训练后 |
|---|---:|---:|
| 后继 learned_transition(A->B) | 0.0 | 0.9412 |
| 概念向量 learned_vector_similarity(A,B) | 0.184 | 0.184 |
| 共现 learned_similarity(A,B) | 0.0 | 0.0 |
| 反向后继 learned_transition(B->A) | - | 0.0 |

pair_evidence(A,B): transition_raw=32.0, positive_raw=0.0

## 召回侧后继 (successors API)

source A = mem-1, top successor = mem-2, score = 1.0436, learned_transition_score = 0.2422

## 结论口径

- 大量训练 A->B 后继, learned_transition(A->B) 从 0.0 升到 0.9412。
- 同时 A、B 的对称概念相似度完全不变 (vector 0.184 -> 0.184, similarity 0.0 -> 0.0): transition 增强后继, 不污染概念向量。
- 后继是有向的: A->B (0.9412) 远大于 B->A (0.0)。
- pair_evidence 为纯 transition (transition_raw>0, positive_raw=0)。
- 召回侧 successors API 暴露正的 learned_transition_score, 后继增益可审计。
- 这是 AP-Core bottom-loop 机制证据, 不修改 runtime, 不宣称开放世界对话基座。
