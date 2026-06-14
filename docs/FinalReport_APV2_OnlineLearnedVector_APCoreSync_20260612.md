# APV2 Online Learned Vector AP-Core Sync Final Report

Date: 2026-06-12

## 1. Design

This round evaluated the Stage 6A.4 online learned vector mechanism for AP-Core sync.

The design target was not a new solver. It was a bottom-loop learning substrate: AP can locally reshape token neighborhoods from pressure, co-occurrence, co-focus, correction, transition, and feedback evidence while keeping the main AP definitions intact.

Design doc:

- `docs/Design_APV2_OnlineLearnedVector_APCoreSync_20260612.md`

## 2. Review And Refinement

The review accepted the mechanism as an auxiliary learned-neighborhood branch.

Core AP remains unchanged:

- Bn/B recall is still state-field recognition over SA payloads, energy, sequence, relation, numeric, posting, and vector evidence.
- Cn/C* successor prediction is still directed transition and successor-payload prediction under a fixed B-derived energy budget.
- learned vectors help candidate ranking, learned family formation, and audit; they do not replace Bn/Cn.

Review doc:

- `docs/Review_APV2_OnlineLearnedVector_APCoreSync_20260612.md`

## 3. Implementation Status

The shared AP base already contains the mechanism:

- `memory/embedding/online_store.py`
  - positive pairs pull token vectors closer;
  - negative pairs push token vectors apart;
  - directed positive/negative anchors move only the subject token;
  - transition pairs write directed successor evidence and do not move symmetric concept vectors.

- `memory/store/memory_store.py`
  - each snapshot exposes `vector_spaces.hash_vector`;
  - each snapshot exposes `vector_spaces.online_learned_vector`;
  - recall scoring/audit exposes `learned_vector_score`;
  - successor lookup exposes `learned_transition_score`;
  - learned transition is bounded by `transition_learned_weight`.

- `memory/persistence/recording.py`, `memory/persistence/jsonl_store.py`, and `memory/persistence/postgres_store.py`
  - persistence accepts and records learned vectors alongside hash vectors.

- `GL_TaskBuilder/EDUCATION_PROTOCOL.md`
  - section 12.1 now defines learned families as experience-shaped neighborhoods, not teacher/LLM hard labels.

AP-side test added:

- `tests/test_online_learned_vector_boundary.py`

## 4. Validation

Commands run:

```text
python GL_TaskBuilder\experiments\gl_openworld_stage6a4_online_learned_vector_reality_check.py
python -m pytest tests\test_online_learned_vector_boundary.py -q
python -m pytest GL_TaskBuilder\tests\test_dailydialogueskill36_successor_packet_online_learning.py tests\test_math1_single_digit_add_sub_persistence.py tests\test_p1h_postgres_persistence_and_audit.py tests\test_p1h2_postgres_warm_load.py -q
```

Results:

```text
Stage 6A.4 experiment: passed=true
AP online learned vector boundary tests: 2 passed
Related regression set: 27 passed, 1 skipped, 1 warning
```

The warning is the existing `legacy_apv2/sensors/hearing_sensor_v1.py` Python `audioop` deprecation warning. It does not affect this mechanism.

Stage 6A.4 evidence:

- positive vector score moved from `0.0` to `0.209`;
- negative learning pushed the same pair to `-0.0294`;
- directed negative anchor produced `subject_anchor_score = -0.106`;
- transition vector similarity stayed unchanged while directed transition score reached `1.0`;
- snapshots carried both `hash_vector` and `online_learned_vector`;
- persisted records carried both vector spaces;
- recall audit exposed learned-vector scores, including `0.5365` and `0.4285` in the current probe.

## 5. What This Proves

This validates a concrete AP bottom-loop capability:

AP can form local learned neighborhoods from its own online experience, keep those neighborhoods separated from deterministic hash vectors, and expose them through recall/persistence audit without relying on an external embedding model or a hidden solver.

It also validates a key design principle for open-world learning:

`family` and `same kind` do not have to be pre-registered as hard labels. They can be approximated by bounded local vector neighborhoods shaped by co-occurrence, co-focus, pressure, correction, transition, and feedback.

The transition result is especially important: successor learning can become strong while preserving directionality. This supports imitation, sequence learning, and process continuation without contaminating symmetric concept similarity.

## 6. Next AP-Core Experiments

The next AP-side experiments should focus on mechanism pressure, not GL learning-score claims:

1. `OnlineVector-WeightAblation-1`
   - Compare `learned_weight=0`, default learned weight, and high learned weight.
   - Acceptance: default improves recall where labels differ but experience links objects; high weight must not dominate exact SA/energy evidence.

2. `OnlineVector-NegativePressure-1`
   - Repeatedly over-predict a wrong subject, then test whether it is pushed away from the actual context.
   - Acceptance: wrong residue loses learned-vector support while useful context remains recoverable.

3. `TransitionIsolation-1`
   - Train A -> B transitions and test A/B concept similarity separately from directed successor score.
   - Acceptance: successor score rises; symmetric learned-vector similarity does not rise just because of transition.

4. `AttentionBand-6A.5`
   - Connect focus-anchor/continue/release/diverge to learned-vector neighborhood bands.
   - Acceptance: attention spreads to learned neighbors under budget, and ablation weakens this effect.

5. `PostgresVectorRoundtrip-1`
   - Use a real PostgreSQL/pgvector instance to write and reload both vector spaces.
   - Acceptance: `memory_vectors.vector_space` roundtrips `hash_vector` and `online_learned_vector`.

## 7. Paper/FAQ Use

Recommended paper phrasing:

APV2 includes a local online learned-vector layer that is updated by AP-native pressure and transition evidence. In current validation, positive co-occurrence pulls token vectors together, negative pressure pushes over-predicted subjects away from real anchors, and transition evidence remains directed. Memory snapshots persist both deterministic hash vectors and learned vectors, and recall audit exposes their separate contributions. This supports learned family formation without keyword gates, answer tables, student-side LLM calls, or hidden solver routes.

Recommended boundary phrasing:

This is bottom-loop mechanism evidence. GL remains responsible for learning curriculum and open-world teacher-off/cold-retest validation. The product shell remains an integration and observability layer.
