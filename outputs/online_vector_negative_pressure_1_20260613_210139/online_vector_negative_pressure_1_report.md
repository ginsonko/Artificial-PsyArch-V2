# OnlineVector-NegativePressure-1 报告

结论: 通过

## 关键检查

```json
{
  "direct_pressure_probe_passed": true,
  "recall_competition_probe_passed": true
}
```

## 直接负压学习 (learned vector 相似度)

| 量 | 值 |
|---|---:|
| 正确关联 C_real-S_good 负压前 | 0.879 |
| 正确关联 C_real-S_good 负压后 | 0.8504 (下降 0.0286) |
| 错误残留 S_wrong-C_real 负压前 | 0.3598 |
| 错误残留 S_wrong-C_real 负压后 | -0.5937 (下降 0.9535) |

pair_evidence(S_wrong, C_real): negative_raw=20.0, source_negative_support=20.0

## 召回竞争 (audit 路径 learned_vector_score)

| 快照 | learned_vector_score |
|---|---:|
| 正确 subject 快照 (S_good) | 0.8136 |
| 错误残留快照 (S_wrong) | 0.0 |

## 结论口径

- 反复过预测错误 subject 产生的负认知压, 把错误残留 S_wrong 从正相似 (0.3598) 推到非正 (-0.5937)。
- 同时正确关联 C_real-S_good 基本保持 (0.879 -> 0.8504), 负压靶向错误残留而不误伤有用经验。
- 在 audit 召回竞争中, 以 C_real 为 query, 正确 subject 快照的 learned-vector 贡献高于错误残留快照。
- 这是 AP-Core bottom-loop 机制证据, 不修改 runtime, 不宣称开放世界对话基座。
