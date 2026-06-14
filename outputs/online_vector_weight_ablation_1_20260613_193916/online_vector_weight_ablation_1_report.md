# OnlineVector-WeightAblation-1 报告

结论: 通过

## 关键检查

```json
{
  "h1_learned_vector_improves_neighborhood_recall": true,
  "h1b_learned_similarity_improves_main_recall": true,
  "h2_exact_match_not_dominated_by_learned_vector": true,
  "h3_monotonic_and_bounded": true
}
```

## audit 召回路径: 三档 learned_weight

| 档位 (weight) | neighbor score | neighbor lvs | neighbor rank | unrelated score | unrelated rank | exact score | exact rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| off (0.0) | 0.6848 | 0.4413 | 1 | 0.6522 | 2 | 14.6954 | 0 |
| default (0.28) | 1.046 | 0.4413 | 1 | 0.6768 | 2 | 14.966 | 0 |
| high (0.9) | 1.6308 | 0.4413 | 1 | 0.6768 | 2 | 15.1495 | 0 |

说明:
- neighbor = A_neighbor，与 query 表层 token 不重叠，但其主体 `gamma` 是 query 主体 `alpha` 的 learned 邻居。
- unrelated = A_unrelated，无表层重叠也无在线经验。
- exact = B_exact，与 query 精确 label/energy 匹配，无在线经验。
- lvs = `learned_vector_score`（online learned vector 在 audit 路径的贡献）。

## 结论口径

- online learned vector 在 audit 召回路径改善了经验邻域召回: neighbor 在 default 档分数高于 off 档，并排在 unrelated 之上。
- 它没有压过主证据: 即使 high 档，精确 label/energy 匹配的 B_exact 仍排在仅靠 learned 相似的 A_neighbor 之前。
- 行为单调有界: off 档 learned 分支对总分贡献为 0（lvs 字段虽被计算暴露，但乘以 `min(0.22, 0)=0`），neighbor 分数随权重单调不减，且 learned vector 分支被 `min(0.22, learned_weight)` cap。
- 这是 AP-Core bottom-loop 机制证据。它不修改 runtime，不宣称开放世界对话基座。
