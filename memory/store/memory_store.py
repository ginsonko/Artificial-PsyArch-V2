from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections import Counter
from itertools import combinations
from math import exp
import copy
from time import perf_counter

from learning_events import LearningEventBuilder
from memory.embedding.online_store import OnlineEmbeddingStore
from memory.persistence import MemoryPersistenceAdapter, NullMemoryPersistence
from memory.relations import RelativeRelationStore
from memory.retrieval.faiss_index import FaissHnswConfig, FaissHnswIndex
from memory.retrieval.hash_vector_index import HashVectorIndex
from memory.retrieval.numeric_feature_index import NumericFeatureIndex, NumericFeatureIndexConfig
from memory.retrieval.posting_index import PostingIndex
from memory.spacetime.transition_store import TransitionStore

"""
PHASE1_MINIMAL_REPLACED:
This store upgrades the original phase-1 overlap-only memory into an explicit
white-box retrieval pipeline with:

1. posting-based candidate recall
2. bigram support
3. hash-vector soft similarity
4. explicit successor transition tracking

It is still a compact APV2.1 implementation, but it is no longer the earlier
"next snapshot only" placeholder.
"""


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _bigrams(tokens: list[str]) -> list[str]:
    clean = [str(token or "").strip() for token in tokens if str(token or "").strip()]
    return [f"{clean[idx]}__{clean[idx + 1]}" for idx in range(0, len(clean) - 1)]


class MemoryStore:
    EPISODE_SUCCESSOR_KIND = "__episode__"

    def __init__(
        self,
        *,
        recall_top_k: int,
        predict_top_k: int,
        prediction_energy_scale: float,
        max_snapshots_per_kind: int,
        vector_dim: int = 64,
        candidate_limit: int = 64,
        core_item_limit: int = 1024,
        query_feature_limit: int = 256,
        posting_label_token_limit: int = 256,
        posting_display_token_limit: int = 128,
        posting_bigram_token_limit: int = 192,
        posting_sequence_token_limit: int = 192,
        vector_token_limit: int = 512,
        scoring_candidate_limit: int = 96,
        learned_rerank_limit: int = 16,
        state_query_signature_token_limit: int = 256,
        numeric_enabled: bool = True,
        numeric_dim: int = 64,
        numeric_candidate_limit: int = 64,
        numeric_top_k_per_channel: int = 24,
        numeric_weight: float = 1.15,
        relation_enabled: bool = True,
        relation_token_limit: int = 256,
        relation_event_limit: int = 128,
        relation_context_limit: int = 8192,
        relation_score_weight: float = 0.68,
        relation_focus_score_weight: float = 0.92,
        index_jobs_per_tick: int = 1,
        ann_enabled: bool = True,
        ann_m: int = 24,
        ann_ef_search: int = 64,
        ann_ef_construction: int = 80,
        online_enabled: bool = True,
        online_dim: int = 32,
        online_token_limit: int = 2048,
        online_min_support_to_promote: int = 2,
        online_per_tick_update_limit: int = 8,
        online_scoring_token_limit: int = 256,
        learned_weight: float = 0.28,
        learned_vector_candidate_weight: float = 4.5,
        transition_learned_weight: float = 0.18,
        temporal_applicability_enabled: bool = True,
        temporal_tick_seconds: float = 0.1,
        temporal_fatigue_window_ticks: int = 80,
        temporal_fatigue_strength: float = 0.92,
        temporal_fatigue_recovery_exponent: float = 1.0,
        temporal_recent_gain_window_ticks: int = 864_000,
        temporal_recent_gain: float = 0.14,
        temporal_long_half_life_ticks: int = 25_920_000,
        temporal_floor: float = 0.18,
        persistence: MemoryPersistenceAdapter | None = None,
        persistence_required: bool = False,
        long_term_recall_enabled: bool = True,
        long_term_recall_kinds: tuple[str, ...] | list[str] | None = None,
        long_term_posting_limit: int = 96,
        long_term_rehydrate_limit: int = 48,
        long_term_rehydrated_resident_limit: int = 512,
        long_term_hot_confidence_threshold: float = 2.25,
        long_term_hot_confident_count: int = 1,
    ) -> None:
        self.recall_top_k = max(1, int(recall_top_k))
        self.predict_top_k = max(1, int(predict_top_k))
        # Memory prediction may distribute less than the current Bn budget, but
        # it must not create energy by configuration. Attention/drive can still
        # amplify state-pool items elsewhere; Cn readout itself stays budgeted.
        self.prediction_energy_scale = _clamp(float(prediction_energy_scale), 0.0, 1.0)
        self.max_snapshots_per_kind = max(8, int(max_snapshots_per_kind))
        # Rank-decay for saturating label-overlap accumulation. After sorting
        # matched-label contributions descending, the rank-th match is weighted
        # by decay**rank, so a long tail of low-specificity generic matches
        # saturates (bounded by max_contribution/(1-decay)) instead of piling up
        # by sheer count. Head (high-specificity skill anchors) keeps full
        # weight. 1.0 would restore the old plain sum; lower = stronger tail
        # suppression. See _weighted_label_overlap.
        self._label_overlap_rank_decay = 0.72
        self.candidate_limit = max(8, int(candidate_limit))
        self.core_item_limit = max(8, int(core_item_limit))
        self.query_feature_limit = max(8, int(query_feature_limit))
        self.posting_label_token_limit = max(16, int(posting_label_token_limit))
        self.posting_display_token_limit = max(16, int(posting_display_token_limit))
        self.posting_bigram_token_limit = max(16, int(posting_bigram_token_limit))
        self.posting_sequence_token_limit = max(16, int(posting_sequence_token_limit))
        self.vector_token_limit = max(32, int(vector_token_limit))
        self.scoring_candidate_limit = max(4, int(scoring_candidate_limit))
        self.learned_rerank_limit = max(0, int(learned_rerank_limit))
        self.state_query_signature_token_limit = max(32, int(state_query_signature_token_limit))
        self.numeric_enabled = bool(numeric_enabled)
        self.numeric_candidate_limit = max(0, int(numeric_candidate_limit))
        self.numeric_top_k_per_channel = max(1, int(numeric_top_k_per_channel))
        self.numeric_weight = max(0.0, float(numeric_weight))
        self.relation_enabled = bool(relation_enabled)
        self.relation_token_limit = max(8, int(relation_token_limit))
        self.relation_event_limit = max(8, int(relation_event_limit))
        self.index_jobs_per_tick = max(0, int(index_jobs_per_tick))
        self.learned_weight = max(0.0, float(learned_weight))
        self._learned_vector_candidate_weight = max(0.0, float(learned_vector_candidate_weight))
        self.transition_learned_weight = max(0.0, float(transition_learned_weight))
        self.temporal_applicability_enabled = bool(temporal_applicability_enabled)
        self.temporal_tick_seconds = max(0.001, float(temporal_tick_seconds))
        self.temporal_fatigue_window_ticks = max(0, int(temporal_fatigue_window_ticks))
        self.temporal_fatigue_strength = _clamp(float(temporal_fatigue_strength), 0.0, 0.98)
        self.temporal_fatigue_recovery_exponent = _clamp(float(temporal_fatigue_recovery_exponent), 0.25, 4.0)
        self.temporal_recent_gain_window_ticks = max(1, int(temporal_recent_gain_window_ticks))
        self.temporal_recent_gain = _clamp(float(temporal_recent_gain), 0.0, 0.65)
        self.temporal_long_half_life_ticks = max(1, int(temporal_long_half_life_ticks))
        self.temporal_floor = _clamp(float(temporal_floor), 0.01, 1.0)
        self._runtime_tick_offset = 0
        self._persistence: MemoryPersistenceAdapter = persistence if persistence is not None else NullMemoryPersistence()
        self._persistence_required = bool(persistence_required)
        self._persistence_write_count = 0
        self._persistence_error_count = 0
        self._last_persistence_error = ""
        self.long_term_recall_enabled = bool(long_term_recall_enabled)
        self.long_term_recall_kinds = {
            str(kind or "")
            for kind in list(long_term_recall_kinds or ("state", "focus", "short_term_slot"))
            if str(kind or "")
        }
        self.long_term_posting_limit = max(8, int(long_term_posting_limit))
        self.long_term_rehydrate_limit = max(1, int(long_term_rehydrate_limit))
        self.long_term_rehydrated_resident_limit = max(32, int(long_term_rehydrated_resident_limit))
        self.long_term_hot_confidence_threshold = max(0.0, float(long_term_hot_confidence_threshold))
        self.long_term_hot_confident_count = max(1, int(long_term_hot_confident_count))
        self.online_scoring_token_limit = max(8, int(online_scoring_token_limit))
        self.online_enabled = bool(online_enabled)
        # B energy transfer is intentionally nonlinear and conservative:
        # good matches should retain most current cognitive energy, while weak
        # matches must not receive a hard zero. These are future tuner hooks.
        self.match_efficiency_soft_k = 2.4
        self.match_efficiency_gamma = 0.45
        self.match_efficiency_floor = 0.08
        self.match_efficiency_ceiling = 0.985
        self.match_efficiency_absolute_weight = 0.72
        self.match_efficiency_relative_weight = 0.28
        self.b_virtual_carry_factor = 0.65
        self.b_attention_real_carry_factor = 0.35
        self.b_drive_fallback_real_factor = 0.35
        self.b_real_transfer_softcap = 8.0
        self.b_virtual_transfer_softcap = 6.0
        self.b_prediction_energy_softcap = 6.0
        # Repeated successor support is an energy-allocation calibration layer,
        # not a new rule that writes answers and not a source of extra energy.
        # Stable "B -> C" experience may make one successor win more of the
        # current B budget, but Cn must still inherit its mass from Bn. This
        # preserves AP's energy semantics: virtual energy means prediction
        # strength/grasp under a fixed cognitive budget, not occurrence count.
        self.successor_payload_support_enabled = True
        self.successor_payload_support_limit = 8192
        self.successor_payload_source_limit = 8
        self.successor_payload_target_limit = 8
        self.successor_payload_support_gain = 0.62
        self.successor_payload_support_soft_k = 5.0
        self.successor_payload_max_gain = 1.65
        self.successor_payload_text_reserve_limit = 4
        self.successor_payload_text_action_reserve_limit = 3
        self.successor_lag_shaping_enabled = True
        self._successor_payload_support: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._successor_payload_outgoing_support: dict[str, float] = defaultdict(float)
        self._snapshots: list[dict] = []
        self._recent_by_kind: dict[str, list[dict]] = defaultdict(list)
        self._previous_by_kind: dict[str, dict | None] = defaultdict(lambda: None)
        self._previous_episode_snapshot: dict | None = None
        self._snapshot_by_id: dict[str, dict] = {}
        self._snapshot_features_by_id: dict[str, dict] = {}
        self._snapshot_energy_by_id: dict[str, dict[str, float]] = {}
        self._snapshot_energy_mass_by_id: dict[str, float] = {}
        self._snapshot_numeric_by_id: dict[str, dict[str, list[float]]] = {}
        self._snapshot_relations_by_id: dict[str, dict] = {}
        self._snapshot_learned_vector_by_id: dict[str, list[float]] = {}
        self._long_term_rehydrated_ids: OrderedDict[str, str] = OrderedDict()
        self._label_document_frequency_by_kind: dict[str, Counter[str]] = defaultdict(Counter)
        # Number of documents (snapshots) actually counted into the document-
        # frequency tables per kind. Used as the IDF `total` so specificity is
        # self-consistent with `frequency` (same sample space). Without this the
        # IDF total came from the bounded _recent_by_kind window while frequency
        # came from the full DF counter, distorting specificity.
        self._document_count_by_kind: dict[str, int] = defaultdict(int)
        self._token_document_frequency_by_kind_field: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self._next_id = 1
        self._posting = PostingIndex()
        self._numeric = NumericFeatureIndex(
            config=NumericFeatureIndexConfig(
                dim=max(4, int(numeric_dim)),
                hnsw_m=max(8, int(ann_m)),
                ef_search=max(8, int(ann_ef_search)),
                ef_construction=max(8, int(ann_ef_construction)),
            )
        )
        self._relations = RelativeRelationStore(
            enabled=self.relation_enabled,
            max_relation_tokens_per_snapshot=self.relation_token_limit,
            max_events_per_snapshot=self.relation_event_limit,
            context_limit=max(64, int(relation_context_limit)),
            score_weight=max(0.0, float(relation_score_weight)),
            focus_score_weight=max(0.0, float(relation_focus_score_weight)),
        )
        # NOTE: HashVectorIndex is retained as a deterministic embedder (cheap + stable).
        # The actual ANN search uses FAISS/HNSW (no full scans), and falls back to a
        # posting-candidate rerank when FAISS is unavailable.
        self._embedder = HashVectorIndex(dim=vector_dim)
        self._ann_enabled = bool(ann_enabled)
        self._ann_config = FaissHnswConfig(
            dim=max(16, int(vector_dim)),
            m=max(8, int(ann_m)),
            ef_search=max(8, int(ann_ef_search)),
            ef_construction=max(8, int(ann_ef_construction)),
        )
        self._ann_by_kind: dict[str, FaissHnswIndex] = {}
        # Parallel ANN over online learned vectors (same dim as hash vectors =
        # vector_dim). This gives the learned semantic bridge its own candidate
        # recall channel, so a math question can surface math-skill snapshots by
        # learned-vector neighborhood even when surface tokens (posting) miss.
        self._ann_learned_by_kind: dict[str, FaissHnswIndex] = {}
        self._vector_cache: dict[str, list[float]] = {}
        self._transitions = TransitionStore()
        self._online = OnlineEmbeddingStore(
            dim=max(16, int(vector_dim)),
            token_limit=online_token_limit,
            min_support_to_promote=online_min_support_to_promote,
            per_tick_update_limit=online_per_tick_update_limit,
        )
        # Incremental-caching anchors (migrated from legacy MemoryStoreV2 design).
        # The key rule is: do not scan; reuse bounded caches keyed by query signature
        # and invalidated by a monotonic memory revision counter.
        self._memory_revision = 0
        self._runtime_state_dirty = False
        self._runtime_state_persist_suspended = 0
        self._runtime_state_last_persist_revision = -1
        self._runtime_relation_restore_high_watermark_tick = -1
        self._query_vector_cache_limit = 192
        self._query_feature_cache_limit = 64
        self._candidate_cache_limit = 96
        self._query_energy_cache_limit = 64
        self._recall_result_cache_limit = 96
        self._query_vector_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_feature_cache: OrderedDict[str, dict] = OrderedDict()
        self._query_energy_cache: OrderedDict[str, tuple[dict[str, float], float]] = OrderedDict()
        self._recall_result_cache: OrderedDict[tuple[int, str, str, str], list[dict]] = OrderedDict()
        # key: (coarse_memory_epoch, memory_kind, query_signature) -> payload.
        # Recent snapshots are appended directly between epoch refreshes, so a
        # cached candidate set never hides the newest B/C evidence.
        self._candidate_cache: OrderedDict[tuple[int, str, str], dict] = OrderedDict()
        self._candidate_cache_revision_stride = 1024
        self._recent_direct_candidate_limit = 48
        self._cache_stats: dict[str, int] = {
            "query_vector_hit": 0,
            "query_vector_miss": 0,
            "candidate_hit": 0,
            "candidate_miss": 0,
            "query_feature_hit": 0,
            "query_feature_miss": 0,
            "query_energy_hit": 0,
            "query_energy_miss": 0,
            "recall_result_hit": 0,
            "recall_result_miss": 0,
            "snapshot_signature_hit": 0,
            "snapshot_signature_miss": 0,
            "index_job_processed": 0,
            "index_job_pending": 0,
            "index_job_pending_heavy": 0,
            "index_job_skip_missing": 0,
            "long_term_pruned_hot_confident": 0,
            "long_term_posting_query": 0,
            "long_term_posting_no_candidate": 0,
            "long_term_posting_candidate": 0,
            "long_term_rehydrated": 0,
            "long_term_rehydrate_skip": 0,
            "long_term_successor_edge_loaded": 0,
            "long_term_successor_rehydrated": 0,
        }
        # Signature-keyed cache for repeated 1024-level state snapshots. This keeps
        # repeated equal cognitive fields cheap without changing the B/C definition.
        self._snapshot_payload_cache_limit = 64
        self._snapshot_payload_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._successor_cache_limit = 512
        self._successor_cache: OrderedDict[tuple[int, str, str, int], list[dict]] = OrderedDict()
        self._successor_cache_revision_stride = 1024
        # ANN tombstone tracking for bounded eviction consistency.
        self._ann_tombstones_by_kind: dict[str, set[str]] = defaultdict(set)
        self._ann_removed_since_rebuild_by_kind: dict[str, int] = defaultdict(int)
        self._ann_learned_tombstones_by_kind: dict[str, set[str]] = defaultdict(set)
        self._ann_learned_removed_since_rebuild_by_kind: dict[str, int] = defaultdict(int)
        self._ann_rebuild_threshold_ratio = 0.18
        self._ann_min_removed_before_rebuild = 64
        self._pending_index_jobs: OrderedDict[str, dict] = OrderedDict()
        self._indexed_snapshot_ids: set[str] = set()
        self._indexed_count_by_kind: dict[str, int] = defaultdict(int)
        self._multimodal_learning_events_total = 0
        self._last_multimodal_learning_events: list[dict] = []
        self._energy_learning_events_total = 0
        self._last_energy_learning_events: list[dict] = []
        self._relation_learning_events_total = 0
        self._last_relation_learning_events: list[dict] = []
        self._structured_learning_event_builder = LearningEventBuilder()
        self._structured_learning_events_total = 0
        self._structured_learning_total_by_type: Counter[str] = Counter()
        self._structured_learning_total_by_layer: Counter[str] = Counter()
        self._structured_learning_total_by_writer: Counter[str] = Counter()
        self._structured_learning_total_by_rule: Counter[str] = Counter()
        self._last_structured_learning_events: list[dict] = []
        self._multimodal_event_preview_limit = 16
        self._energy_event_preview_limit = 16
        self._energy_learning_real_threshold = 0.08
        self._energy_learning_pressure_threshold = 0.08
        self._energy_learning_subject_limit = 8
        self._energy_learning_context_limit = 16
        self._energy_learning_real_softcap = 1.2
        self._energy_learning_pressure_softcap = 1.4
        self._non_core_label_prefixes = (
            "feeling::",
            "timefelt::",
            "rhythmfelt::",
            "expectation_pressure::",
            "action::",
            "action_feedback::",
            "text_action::",
            "control::",
        )
        self._non_core_families = {
            "cognitive_feeling",
            "time_feeling",
            "rhythm_feeling",
            "expectation_pressure",
            "action",
            "action_feedback",
            "text_action",
            "action_control",
        }
        self._non_core_source_types = {
            "cognitive_feeling",
            "time_feeling",
            "rhythm_feeling",
            "expectation_pressure",
            "action_selection",
            "action_feedback",
            "text_action",
            "action_control",
        }

    def write_snapshot(
        self,
        *,
        tick_index: int,
        memory_kind: str,
        items: list[dict],
        focus_labels: list[str],
        source_text: str,
        asset_refs: list[dict] | None = None,
        process_indexes: bool = True,
        successor_boundary: bool = False,
        episode_successor_boundary: bool | None = None,
    ) -> dict:
        local_tick_index = int(tick_index)
        effective_tick_index = self._effective_runtime_tick(local_tick_index)
        self._online.begin_tick(effective_tick_index)
        # Persist tick_index into each item for downstream ordering (e.g. successor payload selection).
        # This keeps TransitionStore free from needing to know tick semantics beyond snapshot metadata.
        stamped_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            stamped = self._apply_runtime_tick_offset_to_item(
                item,
                local_tick_index=local_tick_index,
                effective_tick_index=effective_tick_index,
            )
            stamped_items.append(stamped)
        snapshot = {
            "memory_id": f"mem-{self._next_id}",
            "tick_index": int(effective_tick_index),
            "memory_kind": str(memory_kind),
            "items": stamped_items,
            "focus_labels": [str(label or "") for label in focus_labels if str(label or "")],
            "source_text": str(source_text or ""),
            "asset_refs": self._clean_asset_refs(asset_refs or []),
            "successor_boundary": bool(successor_boundary),
        }
        payload_signature = self._items_signature(
            stamped_items,
            focus_labels=snapshot["focus_labels"] if snapshot["memory_kind"] == "focus" else [],
            tick_index=int(effective_tick_index),
            memory_kind=snapshot["memory_kind"],
        )
        cached_payload = self._snapshot_payload_cache.get((snapshot["memory_kind"], payload_signature))
        if cached_payload is not None:
            self._cache_stats["snapshot_signature_hit"] += 1
            self._snapshot_payload_cache.move_to_end((snapshot["memory_kind"], payload_signature))
            snapshot["sequence_features"] = dict(cached_payload.get("sequence_features", {}) or {})
            snapshot["relation_features"] = dict(cached_payload.get("relation_features", {}) or {})
            state_field_labels = list(cached_payload.get("state_field_labels", []) or [])
            anchor_labels = list(cached_payload.get("anchor_labels", cached_payload.get("core_labels", [])) or [])
            snapshot["state_field_items"] = self._items_for_cached_labels(
                stamped_items,
                state_field_labels,
                fallback=lambda: self._select_state_field_items(stamped_items, limit=self.core_item_limit),
            )
            snapshot["anchor_items"] = self._items_for_cached_labels(
                stamped_items,
                anchor_labels,
                fallback=lambda: self._select_anchor_items(stamped_items, limit=self.core_item_limit),
            )
            # Compatibility view only. The philosophical/main Bn recognition field is
            # `state_field_items`; `core_items` remains the old external-anchor view
            # because persistence, observatory previews, and older tests still read it.
            snapshot["core_items"] = list(snapshot["anchor_items"])
            snapshot_features = dict(cached_payload.get("features", {}) or {})
            vector = list(cached_payload.get("vector", []) or [])
        else:
            self._cache_stats["snapshot_signature_miss"] += 1
            snapshot["sequence_features"] = self._build_sequence_features(snapshot["items"], snapshot["focus_labels"])
            snapshot["state_field_items"] = self._select_state_field_items(snapshot["items"], limit=self.core_item_limit)
            snapshot["anchor_items"] = self._select_anchor_items(snapshot["items"], limit=self.core_item_limit)
            snapshot["core_items"] = list(snapshot["anchor_items"])
            snapshot["relation_features"] = self._relations.build_features(
                memory_kind=snapshot["memory_kind"],
                items=snapshot["state_field_items"],
                focus_labels=snapshot["focus_labels"],
            )
            snapshot_features = self._build_snapshot_features(snapshot)
            vector = self._embedder.embed(self._vector_tokens_for_index(snapshot_features))
            self._snapshot_payload_cache[(snapshot["memory_kind"], payload_signature)] = {
                "sequence_features": dict(snapshot["sequence_features"]),
                "relation_features": dict(snapshot["relation_features"]),
                "state_field_labels": [str(item.get("sa_label", "") or "") for item in snapshot["state_field_items"] if str(item.get("sa_label", "") or "")],
                "anchor_labels": [str(item.get("sa_label", "") or "") for item in snapshot["anchor_items"] if str(item.get("sa_label", "") or "")],
                "core_labels": [str(item.get("sa_label", "") or "") for item in snapshot["core_items"] if str(item.get("sa_label", "") or "")],
                "features": dict(snapshot_features),
                "vector": list(vector),
            }
            self._bounded_ordered_dict(self._snapshot_payload_cache, self._snapshot_payload_cache_limit)
        learned_vector = self._online.learned_vector(
            self._vector_tokens_for_index(snapshot_features),
            limit=self.online_scoring_token_limit,
        ) if self.online_enabled else [0.0] * len(vector)
        snapshot["vector_spaces"] = {
            "hash_vector": list(vector),
            "online_learned_vector": list(learned_vector),
        }
        snapshot["prediction_payload_items"] = self._build_prediction_payload_items(snapshot)
        snapshot["action_feedback_items"] = self._extract_action_feedback_items(snapshot.get("items", []), limit=24)
        snapshot_energy = self._energy_profile(self._snapshot_state_field_items(snapshot), limit=self.core_item_limit)
        snapshot_energy_mass = self._energy_mass(snapshot_energy)
        snapshot_numeric = self._numeric_feature_profile(self._snapshot_state_field_items(snapshot), limit=self.core_item_limit)
        snapshot["numeric_features"] = {channel: list(vector) for channel, vector in snapshot_numeric.items()}
        snapshot_relations = dict(snapshot.get("relation_features", {}) or {})
        self._next_id += 1
        self._snapshots.append(snapshot)
        bucket = self._recent_by_kind[snapshot["memory_kind"]]
        previous = None if bool(successor_boundary) else (bucket[-1] if bucket else None)
        episode_boundary = bool(successor_boundary) if episode_successor_boundary is None else bool(episode_successor_boundary)
        previous_episode = None if episode_boundary else self._previous_episode_snapshot
        previous_memory_id = str((previous or {}).get("memory_id", "") or "")
        previous_episode_memory_id = str((previous_episode or {}).get("memory_id", "") or "")
        transition_edges = []
        if previous_episode_memory_id:
            transition_edges.append(
                {
                    "memory_kind": self.EPISODE_SUCCESSOR_KIND,
                    "source_memory_id": previous_episode_memory_id,
                    "successor_memory_id": snapshot["memory_id"],
                    "transition_meta": {
                        "schema_id": "apv21_episode_successor_edge/v1",
                        "source_memory_kind": str((previous_episode or {}).get("memory_kind", "") or ""),
                        "successor_memory_kind": str(snapshot.get("memory_kind", "") or ""),
                        "policy": "global_episode_time_successor_not_kind_local_shortcut",
                    },
                }
            )
        self._persist_snapshot_authoritative(
            snapshot=snapshot,
            features=snapshot_features,
            vector=vector,
            learned_vector=learned_vector,
            energy_profile=snapshot_energy,
            energy_mass=snapshot_energy_mass,
            numeric_features=snapshot_numeric,
            relation_features=snapshot_relations,
            previous_memory_id=previous_memory_id,
            transition_edges=transition_edges,
        )
        bucket.append(snapshot)
        if len(bucket) > self.max_snapshots_per_kind:
            removed = bucket.pop(0)
            self._evict_snapshot(removed)
        # `_snapshots` is a global append-only list used only for `latest_snapshot(None)`
        # and debugging. Keep it bounded too, otherwise long runs will leak memory.
        # This is not on the tick hot path once steady-state is reached.
        max_global = max(self.max_snapshots_per_kind * 4, 512)
        if len(self._snapshots) > max_global:
            del self._snapshots[0 : len(self._snapshots) - max_global]
        self._snapshot_by_id[snapshot["memory_id"]] = snapshot
        self._transitions.register_snapshot(snapshot)
        if previous is not None:
            self._transitions.link_successor(snapshot["memory_kind"], previous["memory_id"], snapshot["memory_id"])
            self._observe_successor_payload_support(previous, snapshot)
            self._invalidate_successor_cache(snapshot["memory_kind"], previous["memory_id"])
        if previous_episode is not None:
            self._transitions.link_successor(self.EPISODE_SUCCESSOR_KIND, previous_episode["memory_id"], snapshot["memory_id"])
            if previous_episode is not previous:
                self._observe_successor_payload_support(previous_episode, snapshot)
            self._invalidate_successor_cache(self.EPISODE_SUCCESSOR_KIND, previous_episode["memory_id"])
        self._previous_by_kind[snapshot["memory_kind"]] = snapshot
        self._previous_episode_snapshot = snapshot
        self._snapshot_features_by_id[snapshot["memory_id"]] = snapshot_features
        self._snapshot_energy_by_id[snapshot["memory_id"]] = snapshot_energy
        self._snapshot_energy_mass_by_id[snapshot["memory_id"]] = snapshot_energy_mass
        self._snapshot_numeric_by_id[snapshot["memory_id"]] = snapshot_numeric
        self._snapshot_relations_by_id[snapshot["memory_id"]] = snapshot_relations
        self._snapshot_learned_vector_by_id[snapshot["memory_id"]] = list(learned_vector)
        if self.online_enabled:
            self._learn_from_snapshot(snapshot, previous)
        self._register_label_document_frequencies(snapshot)
        if snapshot_relations and not bool(
            snapshot["memory_kind"] == "state" and len(snapshot.get("items", []) or []) > 256
        ):
            self._relations.add_snapshot(
                memory_kind=snapshot["memory_kind"],
                memory_id=snapshot["memory_id"],
                relation_features=snapshot_relations,
                tick_index=int(tick_index),
            )
        self._queue_index_job(snapshot=snapshot, features=snapshot_features, vector=vector, learned_vector=learned_vector, previous=previous)
        if process_indexes:
            job = self._pending_index_jobs.get(snapshot["memory_id"])
            is_heavy = str(snapshot["memory_kind"] or "") == "state" and len(snapshot.get("items", []) or []) > 256
            if job is not None and not is_heavy:
                self._index_snapshot_job_without_learning(snapshot=snapshot, job=job)
                self._pending_index_jobs.pop(snapshot["memory_id"], None)
                self._update_pending_index_stats()
        self._touch_memory_revision()
        return snapshot

    def process_pending_index_jobs(self, budget: int | None = None, *, max_ms: float | None = None, include_heavy: bool = False) -> dict:
        """
        Maintain heavy recall indexes under a fixed budget.

        The hot write path registers snapshots, core payloads and transition links
        immediately. Posting / ANN / online-learning are derived indexes, so they
        can lag by a bounded number of ticks; newest snapshots still participate
        through recent_direct candidates.
        """

        cap = self.index_jobs_per_tick if budget is None else max(0, int(budget))
        started_at = perf_counter()
        max_seconds = None if max_ms is None else max(0.0, float(max_ms)) / 1000.0
        processed = 0
        skipped = 0
        deferred_heavy: list[tuple[str, dict]] = []
        while cap > 0 and self._pending_index_jobs:
            if max_seconds is not None and (perf_counter() - started_at) >= max_seconds:
                break
            memory_id, job = self._pending_index_jobs.popitem(last=False)
            if bool(job.get("heavy", False)) and not include_heavy:
                deferred_heavy.append((memory_id, job))
                if len(deferred_heavy) >= len(self._pending_index_jobs) + 1:
                    break
                continue
            if max_seconds is not None and bool(job.get("heavy", False)):
                estimated_ms = self._estimate_index_job_ms(job)
                elapsed = perf_counter() - started_at
                if processed > 0 and elapsed + (estimated_ms / 1000.0) > max_seconds:
                    self._pending_index_jobs[memory_id] = job
                    self._pending_index_jobs.move_to_end(memory_id, last=False)
                    break
            snapshot = self._snapshot_by_id.get(memory_id)
            if snapshot is None:
                skipped += 1
                continue
            self._index_snapshot_job(snapshot=snapshot, job=job)
            processed += 1
            cap -= 1
        for memory_id, job in reversed(deferred_heavy):
            self._pending_index_jobs[memory_id] = job
            self._pending_index_jobs.move_to_end(memory_id, last=False)
        self._cache_stats["index_job_processed"] += processed
        self._cache_stats["index_job_skip_missing"] += skipped
        self._update_pending_index_stats()
        return {
            "processed": processed,
            "skipped": skipped,
            "deferred_heavy": len(deferred_heavy),
            "pending": len(self._pending_index_jobs),
            "pending_heavy": self.pending_index_job_summary()["heavy"],
            "ms": round((perf_counter() - started_at) * 1000.0, 4),
        }

    def _estimate_index_job_ms(self, job: dict) -> float:
        vector_len = len(job.get("vector", []) or [])
        feature_len = 0
        features = job.get("features", {}) or {}
        for key in ("labels", "displays", "bigrams", "sequence_bigrams", "focus_labels"):
            feature_len += len(features.get(key, []) or [])
        # Conservative local estimate: enough to avoid starting a large HNSW add
        # when the idle budget is already nearly gone.
        return max(1.0, min(80.0, 0.012 * feature_len + 0.02 * vector_len))

    def process_idle_index_maintenance(self, *, budget: int | None = None, max_ms: float | None = None) -> dict:
        """
        Explicit idle/background maintenance entry.

        Realtime ticks do not consume heavy 1024-level state index jobs by
        default. Idle maintenance can include them under a hard budget, keeping
        state memory high-fidelity without charging that work to every tick.
        """

        trace = self.process_pending_index_jobs(budget, max_ms=max_ms, include_heavy=True)
        trace["policy"] = "idle_heavy_index_maintenance"
        return trace

    def pending_index_job_summary(self) -> dict:
        light = 0
        heavy = 0
        by_kind: dict[str, int] = defaultdict(int)
        heavy_by_kind: dict[str, int] = defaultdict(int)
        for job in self._pending_index_jobs.values():
            kind = str(job.get("memory_kind", "") or "")
            by_kind[kind] += 1
            if bool(job.get("heavy", False)):
                heavy += 1
                heavy_by_kind[kind] += 1
            else:
                light += 1
        return {
            "total": len(self._pending_index_jobs),
            "light": light,
            "heavy": heavy,
            "by_kind": dict(sorted(by_kind.items())),
            "heavy_by_kind": dict(sorted(heavy_by_kind.items())),
        }

    def latest_snapshot(self, memory_kind: str | None = None) -> dict | None:
        if memory_kind is None:
            return self._snapshots[-1] if self._snapshots else None
        bucket = self._recent_by_kind.get(str(memory_kind), [])
        return bucket[-1] if bucket else None

    def ann_summary(self) -> dict:
        return {
            kind: ann.summary()
            for kind, ann in sorted(self._ann_by_kind.items())
        }

    def persistence_summary(self) -> dict:
        adapter_summary = self._persistence.summary()
        return {
            **dict(adapter_summary or {}),
            "write_count_seen_by_memory_store": int(self._persistence_write_count),
            "error_count_seen_by_memory_store": int(self._persistence_error_count),
            "last_error_seen_by_memory_store": self._last_persistence_error,
            "required": bool(self._persistence_required),
            "policy": {
                "authoritative_layer": "attached persistence adapter (SQLite local by default for desktop, PostgreSQL/pgvector for advanced deployments)",
                "runtime_layer": "bounded in-memory posting/ANN/numeric/relation/transition indexes",
                "load_all_history_into_memory": False,
            },
        }

    def clone_for_replay(self, *, persistence: MemoryPersistenceAdapter | None = None) -> "MemoryStore":
        clone = copy.deepcopy(self)
        if persistence is not None:
            clone._persistence = persistence
        return clone

    def numeric_summary(self) -> dict:
        return self._numeric.summary()

    def snapshot_by_id(self, memory_id: str) -> dict | None:
        clean = str(memory_id or "")
        if not clean:
            return None
        snapshot = self._snapshot_by_id.get(clean)
        if snapshot is None and self.long_term_recall_enabled:
            self._rehydrate_persistent_snapshot_by_id(clean)
            snapshot = self._snapshot_by_id.get(clean)
        return dict(snapshot) if snapshot is not None else None

    def warm_load_from_persistence(
        self,
        *,
        memory_kind: str | None = None,
        limit_per_kind: int | None = None,
        process_indexes: bool = True,
        replay_learning: bool = False,
        learn_relations_from_loaded: bool = False,
    ) -> dict:
        """
        Rebuild bounded working memory from the authoritative persistence layer.

        This is the restart path. It intentionally asks the adapter for a hot
        window instead of loading the whole long-term database into RAM.
        """

        runtime_state_trace = self._restore_runtime_state_from_persistence()
        loader = getattr(self._persistence, "load_recent_snapshots", None)
        if loader is None:
            return {
                "schema_id": "apv21_warm_load/v1",
                "loaded": 0,
                "skipped": 0,
                "indexed": 0,
                "policy": "persistence_adapter_has_no_loader",
                "runtime_state": runtime_state_trace,
            }
        adapter_config = getattr(self._persistence, "config", None)
        default_limit = getattr(adapter_config, "resident_hot_snapshots_per_kind", 128)
        rows = loader(
            memory_kind=memory_kind,
            limit_per_kind=limit_per_kind or default_limit,
        )
        return self.ingest_persisted_snapshots(
            list(rows or []),
            process_indexes=process_indexes,
            replay_learning=replay_learning,
            learn_relations_from_loaded=learn_relations_from_loaded,
            runtime_state_trace=runtime_state_trace,
        )

    def ingest_persisted_snapshots(
        self,
        snapshots: list[dict],
        *,
        process_indexes: bool = True,
        replay_learning: bool = False,
        learn_relations_from_loaded: bool = False,
        runtime_state_trace: dict | None = None,
    ) -> dict:
        """
        Ingest persisted snapshots as a bounded hot working-memory set.

        `replay_learning` defaults to False because replaying online learning
        from raw snapshots can double-count old experience (red-line 2: C*
        energy is prediction strength, not an occurrence count).

        `learn_relations_from_loaded` is a narrower, red-line-2-safe restore:
        after loading, it runs one learning-only pass that rebuilds the online
        embedder's token co-occurrence/transition table via `observe_*`. It does
        NOT reinject energy, requeue index jobs, or recount C* -- it only
        restores the *relations* (semantic distances) the seed bank already
        encodes, so the learned-similarity channel (and the learned attention
        bands that consume it) stop scoring against an empty table. This is the
        "restore learning from the seed bank" step the warm-load path otherwise
        skips.
        """

        loaded = 0
        skipped = 0
        loaded_snapshots: list[dict] = []
        indexed_before = int(self._cache_stats.get("index_job_processed", 0) or 0)
        restored_tick = int((runtime_state_trace or {}).get("current_tick", -1) or -1)
        relation_restore_high_watermark_present = "relation_restore_high_watermark_tick" in (runtime_state_trace or {})
        relation_restore_high_watermark_tick = int(
            (runtime_state_trace or {}).get("relation_restore_high_watermark_tick", -1) or -1
        )
        self._runtime_relation_restore_high_watermark_tick = max(
            int(self._runtime_relation_restore_high_watermark_tick),
            relation_restore_high_watermark_tick,
        )
        rows = [dict(row) for row in (snapshots or []) if isinstance(row, dict)]
        rows.sort(key=lambda row: (int(row.get("tick_index", -1) or -1), str(row.get("memory_kind", "") or ""), str(row.get("memory_id", "") or "")))
        previous_by_kind: dict[str, dict] = dict(self._previous_by_kind)
        previous_episode_snapshot: dict | None = self._previous_episode_snapshot
        loaded_by_kind: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            memory_id = str(row.get("memory_id", "") or "")
            memory_kind = str(row.get("memory_kind", "") or "")
            if not memory_id or not memory_kind or memory_id in self._snapshot_by_id:
                skipped += 1
                continue
            snapshot = self._normalize_persisted_snapshot(row)
            features = self._build_snapshot_features(snapshot)
            vector = self._embedder.embed(self._vector_tokens_for_index(features))
            state_field_items = self._snapshot_state_field_items(snapshot)
            energy = self._energy_profile(state_field_items, limit=self.core_item_limit)
            energy_mass = self._energy_mass(energy)
            numeric = self._numeric_feature_profile(state_field_items, limit=self.core_item_limit)
            relations = dict(snapshot.get("relation_features", {}) or self._relations.build_features(
                memory_kind=memory_kind,
                items=state_field_items,
                focus_labels=snapshot.get("focus_labels", []),
            ))
            snapshot["numeric_features"] = {channel: list(values) for channel, values in numeric.items()}
            snapshot["relation_features"] = relations
            snapshot["prediction_payload_items"] = self._build_prediction_payload_items(snapshot)
            snapshot["action_feedback_items"] = self._extract_action_feedback_items(snapshot.get("items", []), limit=24)
            previous_snapshot = previous_by_kind.get(memory_kind)
            previous_episode_for_learning = None if bool(snapshot.get("successor_boundary", False)) else previous_episode_snapshot
            self._append_loaded_snapshot(
                snapshot=snapshot,
                features=features,
                vector=vector,
                energy_profile=energy,
                energy_mass=energy_mass,
                numeric_features=numeric,
                relation_features=relations,
                previous=previous_snapshot,
                previous_episode=previous_episode_for_learning,
                process_indexes=process_indexes,
                replay_learning=replay_learning,
            )
            loaded_snapshots.append(snapshot)
            previous_by_kind[memory_kind] = snapshot
            previous_episode_snapshot = snapshot
            loaded_by_kind[memory_kind].append(memory_id)
            self._advance_next_id_from_memory_id(memory_id)
            loaded += 1
        for memory_kind, memory_ids in loaded_by_kind.items():
            for start in range(0, len(memory_ids), 400):
                self._rehydrate_persistent_successors(memory_kind=str(memory_kind or ""), source_memory_ids=memory_ids[start : start + 400])
        if loaded_snapshots:
            max_loaded_tick = max(int(row.get("tick_index", -1) or -1) for row in loaded_snapshots)
            continuation_offset = max_loaded_tick + max(1, int(self.temporal_fatigue_window_ticks))
            self._runtime_tick_offset = max(int(getattr(self, "_runtime_tick_offset", 0) or 0), int(continuation_offset))
        learned_relations = 0
        replay_rows = loaded_snapshots
        if bool((runtime_state_trace or {}).get("restored", False)) and relation_restore_high_watermark_tick >= 0:
            replay_rows = [
                row
                for row in loaded_snapshots
                if int(row.get("tick_index", -1) or -1) > relation_restore_high_watermark_tick
            ]
        elif (
            bool((runtime_state_trace or {}).get("restored", False))
            and not relation_restore_high_watermark_present
            and restored_tick >= 0
        ):
            replay_rows = [row for row in loaded_snapshots if int(row.get("tick_index", -1) or -1) > restored_tick]
        if learn_relations_from_loaded and self.online_enabled and not replay_learning and replay_rows:
            self._runtime_state_persist_suspended += 1
            try:
                learned_relations = self._restore_relations_from_loaded(replay_rows)
            finally:
                self._runtime_state_persist_suspended = max(0, self._runtime_state_persist_suspended - 1)
            replay_max_tick = max(int(row.get("tick_index", -1) or -1) for row in replay_rows)
            self._runtime_relation_restore_high_watermark_tick = max(
                int(self._runtime_relation_restore_high_watermark_tick),
                replay_max_tick,
            )
            self._persist_runtime_state(reason="warm_load_relation_restore")
        self._update_pending_index_stats()
        indexed_after = int(self._cache_stats.get("index_job_processed", 0) or 0)
        return {
            "schema_id": "apv21_warm_load/v1",
            "loaded": int(loaded),
            "skipped": int(skipped),
            "indexed": max(0, indexed_after - indexed_before),
            "pending_index_jobs": len(self._pending_index_jobs),
            "process_indexes": bool(process_indexes),
            "replay_learning": bool(replay_learning),
            "learned_relations": int(learned_relations),
            "runtime_state": runtime_state_trace or self._runtime_state_restore_trace(),
            "policy": "bounded_hot_window_loaded_from_authoritative_persistence;no_full_history_load",
        }

    def strip_runtime_snapshots(self, rows: list[dict], *, preview_limit: int = 8) -> list[dict]:
        """
        Return Bn rows suitable for default traces.

        The recall scorer needs full snapshots internally, but the observatory
        hot path should return references and small previews. Full details remain
        available through `snapshot_by_id(memory_id)`.
        """

        return [self._strip_runtime_snapshot(row, preview_limit=preview_limit) for row in rows or []]

    def recall(self, query_items: list[dict], *, memory_kind: str, top_k: int | None = None, time_context: dict | None = None, _single_round: bool = False) -> list[dict]:
        top_limit = self.recall_top_k if top_k is None else max(1, int(top_k))
        bucket = self._recent_by_kind.get(str(memory_kind), [])
        if not bucket and not self._long_term_recall_available():
            return []
        query_features = self._build_query_features(query_items, memory_kind=str(memory_kind))
        if not query_features["labels"] and not query_features["displays"] and not query_features["focus_labels"]:
            return []

        candidate_signature = str(query_features.get("candidate_signature", "") or self._build_candidate_signature(query_features))
        energy_signature = self._query_energy_signature(query_items, memory_kind=str(memory_kind))
        time_signature = self._time_context_signature(time_context)
        current_tick = self._current_tick_for_temporal(query_items, time_context=time_context)
        temporal_signature = str(current_tick) if current_tick is not None else "no_current_tick"
        # Candidate lookup is allowed to use a coarse epoch, but final Bn rows
        # are the current tick's cognitive judgement. They include learned
        # scores, normalized B energy and grasp confidence, so they must not be
        # reused across later online-learning updates.
        cache_epoch = int(self._memory_revision)
        result_cache_key = (cache_epoch, str(memory_kind), candidate_signature, energy_signature, time_signature, temporal_signature)
        cached_rows = self._recall_result_cache.get(result_cache_key)
        if cached_rows is not None:
            self._cache_stats["recall_result_hit"] += 1
            self._recall_result_cache.move_to_end(result_cache_key)
            return self._refresh_cached_recall_rows(cached_rows, memory_kind=str(memory_kind), top_limit=top_limit)
        self._cache_stats["recall_result_miss"] += 1
        posting_rows, vector_rows, numeric_rows = self._get_or_build_candidates(
            memory_kind=str(memory_kind),
            query_signature=candidate_signature,
            query_features=query_features,
        )
        merged = self._merge_candidates(posting_rows, vector_rows, numeric_rows)
        rows = []
        query_energy, query_mass, query_real_mass, query_virtual_mass = self._get_or_build_query_energy(energy_signature, query_items)
        query_label_set = query_features.get("label_set", set())
        query_display_set = query_features.get("display_set", set())
        query_bigram_set = query_features.get("bigram_set", set())
        query_focus_set = query_features.get("focus_set", set())
        query_sequence_set = query_features.get("sequence_set", set())
        energy_specificity = self._specificity_map_for(memory_kind=str(memory_kind), labels=query_energy.keys())
        # Query-side online learned vector (computed once; candidate-independent).
        # This is the cross-namespace semantic bridge: a math question's tokens
        # pool into a learned coordinate that is close to math-skill snapshots'
        # learned coordinates even when surface labels differ. Restored into the
        # main recall() path (previously only the audit/exact path consumed it).
        query_learned_vector = (
            self._online.learned_vector(query_features["vector_tokens"], limit=self.online_scoring_token_limit)
            if self.online_enabled
            else []
        )
        for candidate_index, candidate in enumerate(merged[: self.scoring_candidate_limit]):
            snapshot = self._snapshot_by_id.get(str(candidate.get("memory_id", "") or ""))
            if not snapshot:
                continue
            snapshot_features = self._snapshot_features_by_id.get(snapshot["memory_id"]) or self._build_snapshot_features(snapshot)
            snapshot_label_set = snapshot_features.get("label_set", set(snapshot_features["labels"]))
            snapshot_display_set = snapshot_features.get("display_set", set(snapshot_features["displays"]))
            snapshot_bigram_set = snapshot_features.get("bigram_set", set(snapshot_features["bigrams"]))
            snapshot_focus_set = snapshot_features.get("focus_set", set(snapshot_features["focus_labels"]))
            snapshot_sequence_set = snapshot_features.get("sequence_set", set(snapshot_features["sequence_bigrams"]))
            label_overlap = len(query_label_set & snapshot_label_set)
            weighted_label = self._weighted_label_overlap(
                memory_kind=str(memory_kind),
                query_items=query_items,
                query_label_set=query_label_set,
                snapshot_label_set=snapshot_label_set,
            )
            display_overlap = len(query_display_set & snapshot_display_set)
            weighted_display_overlap = self._weighted_token_overlap(
                memory_kind=str(memory_kind),
                field_name="display",
                query_tokens=query_display_set,
                snapshot_tokens=snapshot_display_set,
            )
            bigram_overlap = len(query_bigram_set & snapshot_bigram_set)
            weighted_bigram_overlap = self._weighted_token_overlap(
                memory_kind=str(memory_kind),
                field_name="bigram",
                query_tokens=query_bigram_set,
                snapshot_tokens=snapshot_bigram_set,
            )
            focus_overlap = len(query_focus_set & snapshot_focus_set)
            snapshot_energy = self._snapshot_energy_by_id.get(snapshot["memory_id"]) or self._energy_profile(snapshot.get("items", []), limit=self.core_item_limit)
            snapshot_mass = self._snapshot_energy_mass_by_id.get(snapshot["memory_id"])
            if snapshot_mass is None:
                snapshot_mass = self._energy_mass(snapshot_energy)
            state_match = min(query_mass, snapshot_mass) / max(1.0, max(query_mass, snapshot_mass))
            energy_overlap = self._energy_overlap(query_energy, snapshot_energy, query_mass=query_mass, snapshot_mass=snapshot_mass, specificity_by_label=energy_specificity)
            snapshot_learned_vector = self._snapshot_learned_vector_by_id.get(snapshot["memory_id"])
            if snapshot_learned_vector is None:
                snapshot_learned_vector = list((snapshot.get("vector_spaces", {}) or {}).get("online_learned_vector", []) or [])
            learned_vector_score = (
                max(0.0, float(sum(a * b for a, b in zip(query_learned_vector, snapshot_learned_vector))))
                if query_learned_vector and snapshot_learned_vector
                else 0.0
            )
            sequence_overlap = len(query_sequence_set & snapshot_sequence_set)
            weighted_sequence_overlap = self._weighted_token_overlap(
                memory_kind=str(memory_kind),
                field_name="sequence",
                query_tokens=query_sequence_set,
                snapshot_tokens=snapshot_sequence_set,
            )
            posting_score = float(candidate.get("posting_score", 0.0) or 0.0)
            vector_score = float(candidate.get("vector_score", 0.0) or 0.0)
            numeric_score = float(candidate.get("numeric_score", 0.0) or 0.0)
            numeric_score_breakdown = dict(candidate.get("numeric_score_breakdown", {}) or {})
            relation = self._relations.score(
                memory_kind=str(memory_kind),
                query_features=dict(query_features.get("relation_features", {}) or {}),
                candidate_memory_id=snapshot["memory_id"],
            )
            relation_score = float(relation.get("score", 0.0) or 0.0)
            learned = (
                self._online.learned_similarity(
                    query_features["vector_tokens"],
                    snapshot_features["vector_tokens"],
                    limit=self.online_scoring_token_limit,
                )
                if self.online_enabled and candidate_index < self.learned_rerank_limit
                else {"score": 0.0, "contributions": []}
            )
            learned_score = float(learned.get("score", 0.0) or 0.0)
            time_match = self._time_match(snapshot=snapshot, time_context=time_context)
            score_before_temporal = (
                weighted_label["score"] * 1.15
                + weighted_display_overlap["score"] * 0.45
                + weighted_bigram_overlap["score"] * 0.9
                + focus_overlap * 0.7
                + state_match * 0.55
                + energy_overlap * 1.35
                + weighted_sequence_overlap["score"] * 0.8
                + posting_score * 0.35
                + vector_score * 0.4
                + numeric_score * self.numeric_weight
                + relation_score
                + learned_score * self.learned_weight
                + time_match
            )
            temporal = self._temporal_applicability(snapshot, current_tick=current_tick)
            score = score_before_temporal * float(temporal.get("weight", 1.0) or 1.0)
            if score <= 0.0:
                continue
            rows.append(
                {
                    "memory_id": snapshot["memory_id"],
                    "tick_index": snapshot["tick_index"],
                    "query_tick": current_tick,
                    "memory_kind": snapshot["memory_kind"],
                    "score": _round4(score),
                    "raw_score": _round4(score),
                    "score_before_temporal": _round4(score_before_temporal),
                    "temporal_age_ticks": temporal.get("age_ticks"),
                    "temporal_applicability": _round4(float(temporal.get("weight", 1.0) or 1.0)),
                    "temporal_applicability_phase": str(temporal.get("phase", "") or ""),
                    "temporal_applicability_policy": str(temporal.get("policy", "") or ""),
                    "label_overlap": label_overlap,
                    "weighted_label_overlap": _round4(float(weighted_label["score"])),
                    "weighted_label_matches": weighted_label["matches"],
                    "display_overlap": display_overlap,
                    "weighted_display_overlap": _round4(float(weighted_display_overlap["score"])),
                    "bigram_overlap": bigram_overlap,
                    "weighted_bigram_overlap": _round4(float(weighted_bigram_overlap["score"])),
                    "focus_overlap": focus_overlap,
                    "state_match": _round4(state_match),
                    "energy_overlap": _round4(energy_overlap),
                    "sequence_overlap": sequence_overlap,
                    "weighted_sequence_overlap": _round4(float(weighted_sequence_overlap["score"])),
                    "posting_score": _round4(posting_score),
                    "vector_score": _round4(vector_score),
                    "numeric_score": _round4(numeric_score),
                    "numeric_score_breakdown": {
                        str(key): _round4(value)
                        for key, value in sorted(numeric_score_breakdown.items())
                    },
                    "relative_relation_score": _round4(relation_score),
                    "relative_relation_raw_score": _round4(float(relation.get("raw_score", 0.0) or 0.0)),
                    "relation_channels": {
                        str(key): _round4(value)
                        for key, value in sorted(dict(relation.get("relation_channels", {}) or {}).items())
                    },
                    "relation_matches": list(relation.get("relation_matches", []) or []),
                    "learned_score": _round4(learned_score),
                    "time_match": _round4(time_match),
                    "learned_contributions": list(learned.get("contributions", []) or []),
                    "candidate_sources": list(candidate.get("candidate_sources", []) or []),
                    "matched_tokens": dict(candidate.get("matched_tokens", {}) or {}),
                    "matched_token_weights": dict(candidate.get("matched_token_weights", {}) or {}),
                    "score_breakdown": {
                        "label_overlap": label_overlap,
                        "weighted_label_overlap": _round4(float(weighted_label["score"])),
                        "weighted_label_matches": weighted_label["matches"],
                        "display_overlap": display_overlap,
                        "weighted_display_overlap": _round4(float(weighted_display_overlap["score"])),
                        "weighted_display_matches": weighted_display_overlap["matches"],
                        "bigram_overlap": bigram_overlap,
                        "weighted_bigram_overlap": _round4(float(weighted_bigram_overlap["score"])),
                        "weighted_bigram_matches": weighted_bigram_overlap["matches"],
                        "focus_overlap": focus_overlap,
                        "state_match": _round4(state_match),
                        "energy_overlap": _round4(energy_overlap),
                        "sequence_overlap": sequence_overlap,
                        "weighted_sequence_overlap": _round4(float(weighted_sequence_overlap["score"])),
                        "weighted_sequence_matches": weighted_sequence_overlap["matches"],
                        "posting_score": _round4(posting_score),
                        "posting_specificity_score": _round4(float(candidate.get("posting_specificity_score", posting_score) or 0.0)),
                        "vector_score": _round4(vector_score),
                        "numeric_score": _round4(numeric_score),
                        "numeric_channels": {
                            str(key): _round4(value)
                            for key, value in sorted(numeric_score_breakdown.items())
                        },
                        "relative_relation_score": _round4(relation_score),
                        "relative_relation_raw_score": _round4(float(relation.get("raw_score", 0.0) or 0.0)),
                        "relation_channels": {
                            str(key): _round4(value)
                            for key, value in sorted(dict(relation.get("relation_channels", {}) or {}).items())
                        },
                        "learned_score": _round4(learned_score),
                        "time_match": _round4(time_match),
                        "score_before_temporal": _round4(score_before_temporal),
                        "temporal_applicability": _round4(float(temporal.get("weight", 1.0) or 1.0)),
                        "temporal_age_ticks": temporal.get("age_ticks"),
                        "temporal_phase": str(temporal.get("phase", "") or ""),
                    },
                    "source_text": str(snapshot.get("source_text", "") or ""),
                    "snapshot_ref": {
                        "memory_id": snapshot["memory_id"],
                        "tick_index": int(snapshot.get("tick_index", -1) or -1),
                        "memory_kind": str(snapshot.get("memory_kind", "") or ""),
                        "source_text": str(snapshot.get("source_text", "") or ""),
                        "item_count": len(snapshot.get("items", []) or []),
                        "core_item_count": len(snapshot.get("core_items", []) or []),
                        "asset_refs": self._clean_asset_refs(snapshot.get("asset_refs", []) or [])[:8],
                    },
                    "snapshot_preview": self._snapshot_preview(snapshot),
                    "snapshot": snapshot,
                }
            )
        rows.sort(key=lambda item: (-float(item["score"]), -int(item["tick_index"]), str(item["memory_id"])))
        result = self._annotate_recall_energy(
            rows[:top_limit],
            query_mass=query_mass,
            query_real_mass=query_real_mass,
            query_virtual_mass=query_virtual_mass,
        )
        self._recall_result_cache[result_cache_key] = [dict(row) for row in result]
        self._bounded_ordered_dict(self._recall_result_cache, self._recall_result_cache_limit)
        return result

    def recall_residual(self, query_items: list[dict], *, memory_kind: str, top_k: int | None = None, time_context: dict | None = None) -> list[dict]:
        top_limit = self.recall_top_k if top_k is None else max(1, int(top_k))
        working_query_items = [dict(item) for item in query_items or [] if isinstance(item, dict)]
        if not working_query_items:
            return []
        residual_rows: list[dict] = []
        seen_memory_ids: set[str] = set()
        current_query = [dict(item) for item in working_query_items]
        residual_trace: list[dict] = []
        max_rounds = max(1, min(top_limit * 2, 12))
        for round_index in range(1, max_rounds + 1):
            round_rows = self.recall(current_query, memory_kind=memory_kind, top_k=max(1, top_limit * 2), time_context=time_context, _single_round=True)
            round_rows = [row for row in round_rows if str(row.get("memory_id", "") or "") not in seen_memory_ids]
            if not round_rows:
                break
            winner = dict(round_rows[0])
            memory_id = str(winner.get("memory_id", "") or "")
            if not memory_id or memory_id in seen_memory_ids:
                break
            seen_memory_ids.add(memory_id)
            winner["recall_round_index"] = round_index
            residual_rows.append(winner)
            if len(residual_rows) >= top_limit:
                break
            match_efficiency = max(0.0, float(winner.get("match_efficiency", winner.get("grasp_confidence", 0.0)) or 0.0))
            matched_labels = self._residual_matched_labels(winner, current_query)
            next_query: list[dict] = []
            drained_labels: list[str] = []
            residual_before = self._query_residual_mass(current_query)
            for item in current_query:
                next_item = dict(item)
                label = str(next_item.get("sa_label", "") or "")
                if label and label in matched_labels:
                    scale = max(0.08, 1.0 - match_efficiency * 0.82)
                    drained_labels.append(label)
                else:
                    scale = max(0.48, 1.0 - match_efficiency * 0.20)
                next_item["real_energy"] = _round4(max(0.0, float(next_item.get("real_energy", 0.0) or 0.0) * scale))
                next_item["virtual_energy"] = _round4(max(0.0, float(next_item.get("virtual_energy", 0.0) or 0.0) * scale))
                next_item["attention_gain"] = _round4(max(0.0, float(next_item.get("attention_gain", 0.0) or 0.0) * scale))
                next_item["cognitive_pressure"] = _round4(float(next_item.get("real_energy", 0.0) or 0.0) - float(next_item.get("virtual_energy", 0.0) or 0.0))
                if label not in matched_labels or float(next_item.get("real_energy", 0.0) or 0.0) + float(next_item.get("virtual_energy", 0.0) or 0.0) > 0.02:
                    next_query.append(next_item)
            residual_after = self._query_residual_mass(next_query)
            trace_row = {
                "schema_id": "residual_b_recall_round/v1",
                "round_index": int(round_index),
                "winner_memory_id": memory_id,
                "winner_score": _round4(float(winner.get("score", 0.0) or 0.0)),
                "match_efficiency": _round4(match_efficiency),
                "matched_labels": sorted(matched_labels),
                "drained_labels": sorted(set(drained_labels)),
                "residual_mass_before": _round4(residual_before),
                "residual_mass_after": _round4(residual_after),
                "policy": "one_b_per_round_matched_sa_absorption",
            }
            residual_trace.append(trace_row)
            winner["residual_absorption"] = dict(trace_row)
            current_query = next_query
            if not current_query:
                break
        for row in residual_rows:
            row["residual_recall_trace"] = [dict(trace) for trace in residual_trace]
        return residual_rows

    def _residual_matched_labels(self, winner: dict, current_query: list[dict]) -> set[str]:
        snapshot = winner.get("snapshot", {})
        snapshot_items = self._snapshot_state_field_items(snapshot) if isinstance(snapshot, dict) else []
        snapshot_labels = {
            str(item.get("sa_label", "") or "")
            for item in snapshot_items
            if isinstance(item, dict) and str(item.get("sa_label", "") or "")
        }
        query_labels = {
            str(item.get("sa_label", "") or "")
            for item in current_query or []
            if isinstance(item, dict) and str(item.get("sa_label", "") or "")
        }
        matched = {label for label in query_labels if label in snapshot_labels}
        if matched:
            return matched
        matched_tokens = dict(winner.get("matched_tokens", {}) or {})
        token_matches: set[str] = set()
        for label in query_labels:
            if label in matched_tokens:
                token_matches.add(label)
        return token_matches

    def _query_residual_mass(self, rows: list[dict]) -> float:
        total = 0.0
        for item in rows or []:
            if not isinstance(item, dict):
                continue
            total += max(0.0, float(item.get("real_energy", 0.0) or 0.0))
            total += max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
            total += max(0.0, float(item.get("attention_gain", 0.0) or 0.0)) * 0.35
        return total

    def _snapshot_preview(self, snapshot: dict, *, limit: int = 8) -> dict:
        items = list(snapshot.get("core_items", []) or snapshot.get("items", []) or [])[: max(1, int(limit))]
        return {
            "memory_id": str(snapshot.get("memory_id", "") or ""),
            "tick_index": int(snapshot.get("tick_index", -1) or -1),
            "memory_kind": str(snapshot.get("memory_kind", "") or ""),
            "source_text": str(snapshot.get("source_text", "") or ""),
            "labels": [str(item.get("sa_label", "") or "") for item in items if str(item.get("sa_label", "") or "")],
            "focus_labels": [str(label or "") for label in (snapshot.get("focus_labels", []) or [])[:8] if str(label or "")],
            "asset_refs": self._clean_asset_refs(snapshot.get("asset_refs", []) or [])[:8],
        }

    def _clean_asset_refs(self, refs: list[dict]) -> list[dict]:
        rows = []
        seen = set()
        for ref in refs or []:
            if not isinstance(ref, dict):
                continue
            asset_id = str(ref.get("asset_id", "") or "")
            if not asset_id or asset_id in seen:
                continue
            seen.add(asset_id)
            rows.append(dict(ref))
        return rows

    def _annotate_recall_energy(
        self,
        rows: list[dict],
        *,
        query_mass: float,
        query_real_mass: float,
        query_virtual_mass: float,
    ) -> list[dict]:
        clean_rows = [dict(row) for row in rows or [] if isinstance(row, dict)]
        if not clean_rows:
            return []
        positive_scores = [max(0.0, float(row.get("score", row.get("raw_score", 0.0)) or 0.0)) for row in clean_rows]
        score_total = sum(positive_scores)
        top_score = max(positive_scores) if positive_scores else 0.0
        if score_total <= 1e-9:
            fallback = 1.0 / max(1, len(clean_rows))
            weights = [fallback for _ in clean_rows]
        else:
            weights = [score / score_total for score in positive_scores]
        for row, positive_score, normalized_weight in zip(clean_rows, positive_scores, weights):
            match_efficiency = self._match_efficiency(positive_score, top_score=top_score)
            b_real = max(0.0, float(query_real_mass)) * normalized_weight * match_efficiency
            b_virtual = max(0.0, float(query_virtual_mass)) * normalized_weight * match_efficiency
            effective_real = self._softcap_energy(b_real, self.b_real_transfer_softcap)
            effective_virtual = self._softcap_energy(b_virtual, self.b_virtual_transfer_softcap)
            row["normalized_weight"] = _round4(normalized_weight)
            row["match_efficiency"] = _round4(match_efficiency)
            row["grasp_confidence"] = _round4(match_efficiency)
            row["query_energy_mass"] = _round4(query_mass)
            row["query_real_mass"] = _round4(query_real_mass)
            row["query_virtual_mass"] = _round4(query_virtual_mass)
            row["b_real_energy"] = _round4(b_real)
            row["b_virtual_energy"] = _round4(b_virtual)
            row["b_energy_mass"] = _round4(b_real + b_virtual)
            row["b_effective_real_energy"] = _round4(effective_real)
            row["b_effective_virtual_energy"] = _round4(effective_virtual)
            row["energy_transfer"] = {
                "schema_id": "b_energy_transfer/v1",
                "normalized_weight": _round4(normalized_weight),
                "match_efficiency": _round4(match_efficiency),
                "grasp_confidence": _round4(match_efficiency),
                "query_real_mass": _round4(query_real_mass),
                "query_virtual_mass": _round4(query_virtual_mass),
                "b_real_energy": _round4(b_real),
                "b_virtual_energy": _round4(b_virtual),
                "b_effective_real_energy": _round4(effective_real),
                "b_effective_virtual_energy": _round4(effective_virtual),
                "policy": "query_mass_times_normalized_score_times_nonlinear_grasp",
            }
            breakdown = dict(row.get("score_breakdown", {}) or {})
            breakdown["normalized_weight"] = _round4(normalized_weight)
            breakdown["match_efficiency"] = _round4(match_efficiency)
            breakdown["grasp_confidence"] = _round4(match_efficiency)
            breakdown["b_real_energy"] = _round4(b_real)
            breakdown["b_virtual_energy"] = _round4(b_virtual)
            breakdown["b_effective_real_energy"] = _round4(effective_real)
            breakdown["b_effective_virtual_energy"] = _round4(effective_virtual)
            row["score_breakdown"] = breakdown
        return clean_rows

    def _match_efficiency(self, score: float, *, top_score: float) -> float:
        positive_score = max(0.0, float(score or 0.0))
        if positive_score <= 0.0:
            return 0.0
        absolute = 1.0 - exp(-positive_score / max(1e-6, float(self.match_efficiency_soft_k)))
        absolute = pow(_clamp(absolute, 0.0, 1.0), max(0.05, float(self.match_efficiency_gamma)))
        relative = positive_score / max(positive_score, float(top_score or 0.0), 1e-6)
        blended = (
            absolute * float(self.match_efficiency_absolute_weight)
            + relative * float(self.match_efficiency_relative_weight)
        )
        return _clamp(blended, self.match_efficiency_floor, self.match_efficiency_ceiling)

    def _softcap_energy(self, value: float, softcap: float) -> float:
        energy = max(0.0, float(value or 0.0))
        cap = max(0.1, float(softcap or 0.0))
        return cap * (1.0 - exp(-energy / cap))

    def _strip_runtime_snapshot(self, row: dict, *, preview_limit: int) -> dict:
        cleaned = dict(row or {})
        snapshot = dict(cleaned.pop("snapshot", {}) or {})
        if snapshot:
            cleaned.setdefault(
                "snapshot_ref",
                {
                    "memory_id": str(snapshot.get("memory_id", cleaned.get("memory_id", "")) or ""),
                    "tick_index": int(snapshot.get("tick_index", cleaned.get("tick_index", -1)) or -1),
                    "memory_kind": str(snapshot.get("memory_kind", cleaned.get("memory_kind", "")) or ""),
                    "source_text": str(snapshot.get("source_text", cleaned.get("source_text", "")) or ""),
                    "item_count": len(snapshot.get("items", []) or []),
                    "core_item_count": len(snapshot.get("core_items", []) or []),
                    "asset_refs": self._clean_asset_refs(snapshot.get("asset_refs", []) or [])[:8],
                },
            )
            cleaned.setdefault("source_text", str(snapshot.get("source_text", "") or ""))
            cleaned.setdefault("snapshot_preview", self._snapshot_preview(snapshot, limit=preview_limit))
        return cleaned

    def _refresh_cached_recall_rows(self, rows: list[dict], *, memory_kind: str, top_limit: int) -> list[dict]:
        refreshed = []
        seen = set()
        for row in rows or []:
            memory_id = str((row or {}).get("memory_id", "") or "")
            snapshot = self._snapshot_by_id.get(memory_id)
            if not snapshot:
                continue
            clean = dict(row)
            clean["snapshot"] = snapshot
            clean["snapshot_ref"] = {
                "memory_id": snapshot["memory_id"],
                "tick_index": int(snapshot.get("tick_index", -1) or -1),
                "memory_kind": str(snapshot.get("memory_kind", "") or ""),
                "source_text": str(snapshot.get("source_text", "") or ""),
                "item_count": len(snapshot.get("items", []) or []),
                "core_item_count": len(snapshot.get("core_items", []) or []),
                "asset_refs": self._clean_asset_refs(snapshot.get("asset_refs", []) or [])[:8],
            }
            clean["snapshot_preview"] = self._snapshot_preview(snapshot)
            refreshed.append(clean)
            seen.add(memory_id)
        bucket = self._recent_by_kind.get(str(memory_kind or ""), [])
        for snapshot in reversed(bucket[-min(self._recent_direct_candidate_limit, 8) :]):
            memory_id = str((snapshot or {}).get("memory_id", "") or "")
            if not memory_id or memory_id in seen:
                continue
            clean = {
                "memory_id": memory_id,
                "tick_index": int(snapshot.get("tick_index", -1) or -1),
                "memory_kind": str(snapshot.get("memory_kind", "") or ""),
                "score": 0.0001,
                "raw_score": 0.0001,
                "label_overlap": 0,
                "display_overlap": 0,
                "bigram_overlap": 0,
                "focus_overlap": 0,
                "state_match": 0.0,
                "energy_overlap": 0.0,
                "sequence_overlap": 0,
                "posting_score": 0.0,
                "vector_score": 0.0,
                "numeric_score": 0.0,
                "numeric_score_breakdown": {},
                "relative_relation_score": 0.0,
                "relative_relation_raw_score": 0.0,
                "relation_channels": {},
                "relation_matches": [],
                "learned_score": 0.0,
                "time_match": 0.0,
                "normalized_weight": 0.0,
                "match_efficiency": 0.0,
                "grasp_confidence": 0.0,
                "query_energy_mass": 0.0,
                "query_real_mass": 0.0,
                "query_virtual_mass": 0.0,
                "b_real_energy": 0.0,
                "b_virtual_energy": 0.0,
                "b_energy_mass": 0.0,
                "b_effective_real_energy": 0.0,
                "b_effective_virtual_energy": 0.0,
                "energy_transfer": {
                    "schema_id": "b_energy_transfer/v1",
                    "policy": "cached_recent_direct_fallback_unweighted",
                },
                "learned_contributions": [],
                "candidate_sources": ["recent_direct_cached_result"],
                "matched_tokens": {},
                "score_breakdown": {
                    "label_overlap": 0,
                    "display_overlap": 0,
                    "bigram_overlap": 0,
                    "focus_overlap": 0,
                    "state_match": 0.0,
                    "energy_overlap": 0.0,
                    "sequence_overlap": 0,
                    "posting_score": 0.0,
                    "vector_score": 0.0,
                    "numeric_score": 0.0,
                    "numeric_channels": {},
                    "relative_relation_score": 0.0,
                    "relative_relation_raw_score": 0.0,
                    "relation_channels": {},
                    "learned_score": 0.0,
                    "time_match": 0.0,
                    "normalized_weight": 0.0,
                    "match_efficiency": 0.0,
                    "grasp_confidence": 0.0,
                    "b_real_energy": 0.0,
                    "b_virtual_energy": 0.0,
                    "b_effective_real_energy": 0.0,
                    "b_effective_virtual_energy": 0.0,
                },
                "source_text": str(snapshot.get("source_text", "") or ""),
                "snapshot_ref": {
                    "memory_id": memory_id,
                    "tick_index": int(snapshot.get("tick_index", -1) or -1),
                    "memory_kind": str(snapshot.get("memory_kind", "") or ""),
                    "source_text": str(snapshot.get("source_text", "") or ""),
                    "item_count": len(snapshot.get("items", []) or []),
                    "core_item_count": len(snapshot.get("core_items", []) or []),
                },
                "snapshot_preview": self._snapshot_preview(snapshot),
                "snapshot": snapshot,
            }
            refreshed.append(clean)
            seen.add(memory_id)
            if len(refreshed) >= top_limit:
                break
        refreshed.sort(key=lambda item: (-float(item.get("score", 0.0) or 0.0), -int(item.get("tick_index", -1) or -1), str(item.get("memory_id", "") or "")))
        return refreshed[:top_limit]

    def _time_context_signature(self, time_context: dict | None) -> str:
        if not time_context:
            return "no_time"
        return "|".join(
            [
                str(round(float(time_context.get("target_delta_t", 0.0) or 0.0), 3)),
                str(round(float(time_context.get("time_sigma", 0.0) or 0.0), 3)),
                str(round(float(time_context.get("confidence", 0.0) or 0.0), 3)),
                str(round(float(time_context.get("gain", 0.0) or 0.0), 3)),
            ]
        )

    def successors(
        self,
        memory_id: str,
        *,
        memory_kind: str,
        top_k: int | None = None,
        source_b_row: dict | None = None,
        current_tick: int | None = None,
    ) -> list[dict]:
        top_limit = self.predict_top_k if top_k is None else max(1, int(top_k))
        # Cn rows also carry learned transition scores and prediction-energy
        # calibration. Reusing them across memory revisions would make repeated
        # experience invisible until a very coarse cache epoch flips.
        cache_epoch = int(self._memory_revision)
        lag_shape_key = 1 if bool(getattr(self, "successor_lag_shaping_enabled", True)) else 0
        cache_key = (cache_epoch, str(memory_kind or ""), str(memory_id or ""), int(top_limit), lag_shape_key)
        cached = self._successor_cache.get(cache_key)
        if cached is not None:
            self._successor_cache.move_to_end(cache_key)
            rows = [dict(row) for row in cached]
            rows = self._apply_successor_temporal_applicability(rows, current_tick=current_tick, source_b_row=source_b_row)
            return self._scale_successor_rows_by_b(rows, source_b_row=source_b_row) if source_b_row else rows
        rows = self._successor_rows_with_episode_edges(
            memory_kind=str(memory_kind),
            memory_id=str(memory_id),
            top_limit=top_limit,
        )
        if self.online_enabled:
            source_snapshot = self._snapshot_by_id.get(str(memory_id or ""))
            source_features = (self._snapshot_features_by_id.get(str(memory_id or "")) or self._build_snapshot_features(source_snapshot)) if source_snapshot else {"vector_tokens": []}
            for row in rows:
                successor_snapshot = self._snapshot_by_id.get(str(row.get("successor_memory_id", "") or ""))
                successor_id = str(row.get("successor_memory_id", "") or "")
                successor_features = (self._snapshot_features_by_id.get(successor_id) or self._build_snapshot_features(successor_snapshot)) if successor_snapshot else {"vector_tokens": []}
                learned = self._online.learned_transition(
                    source_features["vector_tokens"],
                    successor_features["vector_tokens"],
                    limit=self.online_scoring_token_limit,
                )
                learned_score = float(learned.get("score", 0.0) or 0.0)
                row["learned_transition_score"] = _round4(learned_score)
                row["learned_transition_contributions"] = list(learned.get("contributions", []) or [])
                row["score"] = _round4(float(row.get("score", 1.0) or 1.0) + learned_score * self.transition_learned_weight)
        else:
            for row in rows:
                row["learned_transition_score"] = 0.0
                row["learned_transition_contributions"] = []
        rows.sort(key=lambda item: (-float(item.get("score", 0.0) or 0.0), str(item.get("successor_memory_id", "") or "")))
        base_result = rows[:top_limit]
        self._successor_cache[cache_key] = [dict(row) for row in base_result]
        self._bounded_ordered_dict(self._successor_cache, self._successor_cache_limit)
        result = self._apply_successor_temporal_applicability(base_result, current_tick=current_tick, source_b_row=source_b_row)
        result.sort(key=lambda item: (-float(item.get("score", 0.0) or 0.0), str(item.get("successor_memory_id", "") or "")))
        return self._scale_successor_rows_by_b(result, source_b_row=source_b_row) if source_b_row else result

    def _successor_rows_with_episode_edges(self, *, memory_kind: str, memory_id: str, top_limit: int) -> list[dict]:
        edge_kinds = [str(memory_kind or "")]
        if str(memory_kind or "") != self.EPISODE_SUCCESSOR_KIND:
            edge_kinds.append(self.EPISODE_SUCCESSOR_KIND)
        merged: dict[str, dict] = {}
        for edge_kind in edge_kinds:
            rows = self._transitions.successors(
                edge_kind,
                str(memory_id or ""),
                top_k=max(1, int(top_limit)),
                prediction_energy_scale=self.prediction_energy_scale,
                lag_shaping_enabled=bool(getattr(self, "successor_lag_shaping_enabled", True)),
            )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                successor_id = str(row.get("successor_memory_id", "") or "")
                if not successor_id:
                    continue
                clean = dict(row)
                clean["successor_edge_kind"] = edge_kind
                existing = merged.get(successor_id)
                if existing is None or float(clean.get("score", 0.0) or 0.0) > float(existing.get("score", 0.0) or 0.0):
                    merged[successor_id] = clean
                else:
                    edge_sources = list(existing.get("successor_edge_kinds", []) or [])
                    if edge_kind not in edge_sources:
                        edge_sources.append(edge_kind)
                    existing["successor_edge_kinds"] = edge_sources
        out = list(merged.values())
        out.sort(key=lambda item: (-float(item.get("score", 0.0) or 0.0), str(item.get("successor_memory_id", "") or "")))
        return out[: max(1, int(top_limit))]

    def _scale_successor_rows_by_b(self, rows: list[dict], *, source_b_row: dict | None) -> list[dict]:
        b_row = dict(source_b_row or {})
        if not b_row:
            return [dict(row) for row in rows or []]
        clean_rows = [dict(row) for row in rows or [] if isinstance(row, dict)]
        if not clean_rows:
            return []
        positive_scores = [max(0.0, float(row.get("score", 0.0) or 0.0)) for row in clean_rows]
        total = sum(positive_scores)
        if total <= 1e-9:
            preliminary_weights = [1.0 / max(1, len(clean_rows)) for _ in clean_rows]
        else:
            preliminary_weights = [score / total for score in positive_scores]
        calibrations = [
            self._successor_energy_calibration(row, b_row=b_row, successor_weight=preliminary_weight)
            for row, preliminary_weight in zip(clean_rows, preliminary_weights)
        ]
        # Stable successor support may bias which C branch wins, but only before
        # normalization. The normalized weights below still sum to one, so the
        # following energy transfer distributes a fixed B-derived budget instead
        # of treating repeated support as extra occurrence count.
        calibrated_scores = [
            positive_score * max(0.0, float(calibration.get("gain", 1.0) or 1.0))
            for positive_score, calibration in zip(positive_scores, calibrations)
        ]
        calibrated_total = sum(calibrated_scores)
        if calibrated_total <= 1e-9:
            weights = list(preliminary_weights)
        else:
            weights = [score / calibrated_total for score in calibrated_scores]
        b_real = max(0.0, float(b_row.get("b_real_energy", 0.0) or 0.0))
        b_virtual = max(0.0, float(b_row.get("b_virtual_energy", 0.0) or 0.0))
        b_effective_real = max(0.0, float(b_row.get("b_effective_real_energy", b_real) or 0.0))
        b_effective_virtual = max(0.0, float(b_row.get("b_effective_virtual_energy", b_virtual) or 0.0))
        b_weight = max(0.0, float(b_row.get("normalized_weight", 0.0) or 0.0))
        b_efficiency = max(0.0, float(b_row.get("match_efficiency", b_row.get("grasp_confidence", 0.0)) or 0.0))
        source_mass = b_effective_real + b_effective_virtual * float(self.b_virtual_carry_factor)
        if source_mass <= 0.0:
            source_mass = max(0.0, b_weight * b_efficiency)
        for row, preliminary_weight, successor_weight, calibration in zip(clean_rows, preliminary_weights, weights, calibrations):
            temporal_weight = _clamp(float(row.get("temporal_applicability", 1.0) or 1.0), 0.0, 1.0 + self.temporal_recent_gain)
            transfer_mass = self._prediction_energy_transfer_mass(source_mass * successor_weight * temporal_weight)
            # Calibration is intentionally audit-only at this post-normalization
            # step. Its gain describes why this successor is trusted, but once
            # successor_weight has allocated a B-derived share, Cn must not
            # multiply that share again. Otherwise stable memories would create
            # energy instead of expressing stronger expectation within budget.
            calibrated_transfer_mass = transfer_mass
            base_items = [dict(item) for item in list(row.get("predicted_items", []) or []) if isinstance(item, dict)]
            base_total = sum(max(0.0, float(item.get("virtual_energy", 0.0) or 0.0)) for item in base_items)
            row["source_b_weight"] = _round4(b_weight)
            row["source_b_match_efficiency"] = _round4(b_efficiency)
            row["source_b_real_energy"] = _round4(b_real)
            row["source_b_virtual_energy"] = _round4(b_virtual)
            row["source_b_effective_real_energy"] = _round4(b_effective_real)
            row["source_b_effective_virtual_energy"] = _round4(b_effective_virtual)
            row["source_b_energy_mass"] = _round4(source_mass)
            row["successor_precalibration_weight"] = _round4(preliminary_weight)
            row["successor_normalized_weight"] = _round4(successor_weight)
            row["successor_temporal_weight"] = _round4(temporal_weight)
            row["energy_transfer_multiplier"] = _round4(transfer_mass)
            row["calibrated_energy_transfer_multiplier"] = _round4(calibrated_transfer_mass)
            row["prediction_energy_calibration"] = calibration
            row["energy_transfer"] = {
                "schema_id": "c_energy_transfer/v1",
                "source_b_weight": _round4(b_weight),
                "source_b_match_efficiency": _round4(b_efficiency),
                "source_b_real_energy": _round4(b_real),
                "source_b_virtual_energy": _round4(b_virtual),
                "source_b_effective_real_energy": _round4(b_effective_real),
                "source_b_effective_virtual_energy": _round4(b_effective_virtual),
                "source_b_energy_mass": _round4(source_mass),
                "successor_precalibration_weight": _round4(preliminary_weight),
                "successor_normalized_weight": _round4(successor_weight),
                "successor_temporal_weight": _round4(temporal_weight),
                "transfer_multiplier": _round4(transfer_mass),
                "calibrated_transfer_multiplier": _round4(calibrated_transfer_mass),
                "calibration": calibration,
                "payload_base_virtual_total": _round4(base_total),
                "energy_budget_semantics": "support_biases_successor_competition_not_post_normalization_energy_gain",
                "policy": "source_b_effective_energy_distributed_over_successor_payload_under_fixed_budget",
            }
            predicted_items = []
            for item in base_items:
                new_item = dict(item)
                base_virtual = max(0.0, float(new_item.get("virtual_energy", 0.0) or 0.0))
                item_share = (base_virtual / base_total) if base_total > 1e-9 else (1.0 / max(1, len(base_items)))
                scaled_virtual = calibrated_transfer_mass * item_share
                new_item["base_virtual_energy"] = _round4(base_virtual)
                new_item["virtual_energy"] = _round4(scaled_virtual)
                meta = dict(new_item.get("anchor_meta", {}) or {}) if isinstance(new_item.get("anchor_meta", {}), dict) else {}
                meta["prediction_energy_transfer"] = {
                    "source_memory_id": str(row.get("source_memory_id", "") or ""),
                    "successor_memory_id": str(row.get("successor_memory_id", "") or ""),
                    "source_b_weight": _round4(b_weight),
                    "source_b_match_efficiency": _round4(b_efficiency),
                    "successor_weight": _round4(successor_weight),
                    "successor_temporal_weight": _round4(temporal_weight),
                    "base_virtual_energy": _round4(base_virtual),
                    "payload_share": _round4(item_share),
                    "transfer_multiplier": _round4(transfer_mass),
                    "calibrated_transfer_multiplier": _round4(calibrated_transfer_mass),
                    "calibration_gain": _round4(float(calibration.get("gain", 1.0) or 1.0)),
                    "energy_budget_semantics": "virtual_energy_is_prediction_strength_not_occurrence_count",
                }
                new_item["anchor_meta"] = meta
                predicted_items.append(new_item)
            row["predicted_items"] = predicted_items
        return clean_rows

    def _prediction_energy_transfer_mass(self, value: float) -> float:
        mass = max(0.0, float(value or 0.0))
        softcap = max(1.0, float(self.b_prediction_energy_softcap))
        # Saturating transfer avoids runaway C* energy in long familiar loops
        # while preserving near-linear behavior for ordinary small query masses.
        return softcap * (1.0 - exp(-mass / softcap)) * float(self.prediction_energy_scale)

    def _successor_energy_calibration(self, row: dict, *, b_row: dict, successor_weight: float) -> dict:
        """
        Return a conservative Cn successor-competition gain.

        Philosophy:
        repeated successor support should make AP more willing to "expect" a
        familiar next state, but only by changing how a fixed B-derived
        prediction budget is distributed among successor candidates. The gain is
        recorded for audit and should be applied before successor normalization,
        not after normalized energy transfer.

        This protects AP's core energy picture:
        - high virtual energy means strong prediction/grasp;
        - repeated B support is evidence for stronger expectation;
        - support does not mean the same event happens multiple times;
        - Cn/C* cannot create energy independently from Bn.
        """

        if not bool(self.successor_payload_support_enabled):
            return {
                "schema_id": "successor_energy_calibration/v1",
                "gain": 1.0,
                "enabled": False,
                "policy": "disabled",
            }
        source_snapshot = self._snapshot_by_id.get(str(row.get("source_memory_id", "") or ""))
        successor_snapshot = self._snapshot_by_id.get(str(row.get("successor_memory_id", "") or ""))
        source_labels = self._successor_support_labels(source_snapshot, limit=self.successor_payload_source_limit)
        target_labels = self._successor_support_labels(successor_snapshot, limit=self.successor_payload_target_limit)
        total_support = 0.0
        for source in source_labels:
            for target in target_labels:
                total_support += float(self._successor_payload_support.get((source, target), 0.0) or 0.0)
        source_outgoing = sum(float(self._successor_payload_outgoing_support.get(source, 0.0) or 0.0) for source in source_labels)
        support_ratio = total_support / max(1.0, source_outgoing)
        support_saturation = 1.0 - exp(-total_support / max(1e-6, float(self.successor_payload_support_soft_k)))
        transition_confidence = max(0.0, float(row.get("learned_transition_score", 0.0) or 0.0))
        grasp_confidence = max(0.0, float(b_row.get("match_efficiency", b_row.get("grasp_confidence", 0.0)) or 0.0))
        decisive_successor = max(0.0, float(successor_weight or 0.0))
        ambiguity_damping = 0.65 + 0.35 * decisive_successor
        evidence_gain = (
            support_saturation * 0.46
            + support_ratio * 0.24
            + transition_confidence * 0.18
            + grasp_confidence * 0.12
        )
        raw_gain = 1.0 + float(self.successor_payload_support_gain) * evidence_gain * ambiguity_damping
        gain = _clamp(raw_gain, 1.0, float(self.successor_payload_max_gain))
        return {
            "schema_id": "successor_energy_calibration/v1",
            "enabled": True,
            "gain": _round4(gain),
            "raw_gain": _round4(raw_gain),
            "support_total": _round4(total_support),
            "support_ratio": _round4(support_ratio),
            "support_saturation": _round4(support_saturation),
            "transition_confidence": _round4(transition_confidence),
            "grasp_confidence": _round4(grasp_confidence),
            "successor_weight": _round4(decisive_successor),
            "ambiguity_damping": _round4(ambiguity_damping),
            "source_label_count": len(source_labels),
            "target_label_count": len(target_labels),
            "policy": "stable_successor_support_biases_successor_competition_under_fixed_b_budget",
        }

    def _observe_successor_payload_support(self, source_snapshot: dict | None, successor_snapshot: dict | None) -> None:
        """
        Accumulate bounded evidence that source labels were followed by target labels.

        TransitionStore records snapshot-to-snapshot edges. This auxiliary table
        records a tiny white-box summary of payload-level repeated succession so
        Cn energy can be calibrated without scanning history or contaminating the
        symmetric concept-similarity embedding.
        """

        if not bool(self.successor_payload_support_enabled):
            return
        source_labels = self._successor_support_labels(source_snapshot, limit=self.successor_payload_source_limit)
        target_labels = self._successor_support_labels(successor_snapshot, limit=self.successor_payload_target_limit)
        if not source_labels or not target_labels:
            return
        source_weight = 1.0 / max(1, len(source_labels))
        target_weight = 1.0 / max(1, len(target_labels))
        amount = source_weight * target_weight
        for source in source_labels:
            for target in target_labels:
                key = (source, target)
                if key not in self._successor_payload_support and len(self._successor_payload_support) >= int(self.successor_payload_support_limit):
                    old_key, old_value = self._successor_payload_support.popitem(last=False)
                    old_source = old_key[0]
                    self._successor_payload_outgoing_support[old_source] = max(
                        0.0,
                        float(self._successor_payload_outgoing_support.get(old_source, 0.0) or 0.0) - float(old_value or 0.0),
                    )
                self._successor_payload_support[key] = float(self._successor_payload_support.get(key, 0.0) or 0.0) + amount
                self._successor_payload_outgoing_support[source] = float(self._successor_payload_outgoing_support.get(source, 0.0) or 0.0) + amount
                self._successor_payload_support.move_to_end(key)

    def _successor_support_labels(self, snapshot: dict | None, *, limit: int) -> list[str]:
        if not isinstance(snapshot, dict):
            return []
        rows = []
        source_items = snapshot.get("prediction_payload_items", None)
        if not isinstance(source_items, list) or not source_items:
            source_items = snapshot.get("state_field_items", None)
        if not isinstance(source_items, list) or not source_items:
            source_items = snapshot.get("items", []) or []
        for item in source_items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or label in rows:
                continue
            rows.append(label)
            if len(rows) >= max(1, int(limit)):
                break
        return rows

    def successor_links(
        self,
        memory_id: str,
        *,
        memory_kind: str,
        top_k: int | None = None,
        current_tick: int | None = None,
        source_b_row: dict | None = None,
    ) -> list[dict]:
        """
        Bounded successor lookup for white-box consequence paths.

        This is intentionally a thin wrapper over the existing transition store:
        no recall, no graph scan, no new definition of Cn.
        """

        return self.successors(
            str(memory_id or ""),
            memory_kind=str(memory_kind or ""),
            top_k=top_k,
            current_tick=current_tick,
            source_b_row=source_b_row,
        )

    def online_embedding_summary(self) -> dict:
        summary = self._online.summary()
        summary["memory_cache_stats"] = dict(self._cache_stats)
        summary["numeric_feature_index"] = self.numeric_summary()
        summary["relative_relations"] = self._relations.summary()
        summary["multimodal_learning"] = {
            "schema_id": "multimodal_handle_learning/v1",
            "total_events": int(self._multimodal_learning_events_total),
            "last_event_count": len(self._last_multimodal_learning_events),
            "last_events": [dict(row) for row in self._last_multimodal_learning_events],
        }
        summary["energy_learning"] = {
            "schema_id": "energy_pressure_online_learning/v1",
            "total_events": int(self._energy_learning_events_total),
            "last_event_count": len(self._last_energy_learning_events),
            "last_events": [dict(row) for row in self._last_energy_learning_events],
            "policy": {
                "real_threshold": _round4(self._energy_learning_real_threshold),
                "pressure_threshold": _round4(self._energy_learning_pressure_threshold),
                "subject_limit": int(self._energy_learning_subject_limit),
                "context_limit": int(self._energy_learning_context_limit),
                "real_softcap": _round4(self._energy_learning_real_softcap),
                "pressure_softcap": _round4(self._energy_learning_pressure_softcap),
                "weight_model": "directed_pressure_subject_to_real_anchor/v3",
                "update_semantics": "only_pressure_subject_moves;real_context_is_anchor",
            },
        }
        structured_summary = self._structured_learning_event_builder.summarize(self._last_structured_learning_events)
        structured_summary["total_events"] = int(self._structured_learning_events_total)
        structured_summary["last_event_count"] = len(self._last_structured_learning_events)
        summary["structured_learning_events"] = {
            **structured_summary,
            "total_by_type": dict(sorted((key, int(value)) for key, value in self._structured_learning_total_by_type.items())),
            "total_by_layer": dict(sorted((key, int(value)) for key, value in self._structured_learning_total_by_layer.items())),
            "total_by_writer": dict(sorted((key, int(value)) for key, value in self._structured_learning_total_by_writer.items())),
            "total_by_rule": dict(sorted((key, int(value)) for key, value in self._structured_learning_total_by_rule.items())),
            "last_events": [dict(row) for row in self._last_structured_learning_events],
            "policy": {
                "schema": "BC events are explanatory contracts for learning evidence",
                "direct_writers": ["MemoryStore._learn_from_snapshot", "ActionOutcomeMemory.record", "BAnchorExpectationVerifier.update"],
                "boundary": "innate rules route learning evidence; specialized stores perform bounded writes",
            },
        }
        summary["layered_online_learning"] = {
            "schema_id": "apv21_layered_online_embedding_summary/v1",
            "layers": [
                {
                    "name": "content_recognition_embedding",
                    "source": "cognitive_pressure_positive_negative_events",
                    "total_events": int(self._energy_learning_events_total),
                },
                {
                    "name": "relation_order_embedding",
                    "source": "relative_relation_tokens_and_events",
                    "total_events": int(self._relation_learning_events_total),
                    "last_event_count": len(self._last_relation_learning_events),
                    "last_events": [dict(row) for row in self._last_relation_learning_events],
                },
                {
                    "name": "multimodal_binding_embedding",
                    "source": "same_tick_cofocus_visual_audio_text_handles",
                    "total_events": int(self._multimodal_learning_events_total),
                },
                {
                    "name": "successor_transition_embedding",
                    "source": "snapshot_successors_focus_successor_and_handle_transition",
                    "total_events": int(self._total_transition_learning_events()),
                },
                {
                    "name": "reconstruction_completion_embedding",
                    "source": "reconstruction_payload_coverage_and_recalled_B_completion",
                    "total_events": int(self._relation_learning_events_total),
                },
                {
                    "name": "action_outcome_embedding",
                    "source": "action_feedback_reward_punishment_successors",
                    "total_events": 0,
                },
                {
                    "name": "focus_sampling_policy_embedding",
                    "source": "focus_precision_pressure_and_payload_fidelity",
                    "total_events": int(self._relation_learning_events_total),
                },
                {
                    "name": "self_state_embedding",
                    "source": "global_energy_pressure_attention_fatigue_metrics",
                    "total_events": int(self._energy_learning_events_total),
                },
            ],
            "policy": {
                "local_only": True,
                "online_update": True,
                "white_box_events": True,
                "black_box_replacement": False,
            },
        }
        return summary

    def learned_similarity(self, query_tokens: list[str], candidate_tokens: list[str], *, limit: int | None = None) -> dict:
        """
        Public read-only bridge for AP's local online association layer.

        Short-term memory recall can use this to bias a recent-window search by
        what AP has already learned, while preserving the learning boundary:
        MemoryStore's snapshot/event pipeline remains the only writer.
        """

        if not self.online_enabled:
            return {"score": 0.0, "contributions": [], "negative_contributions": []}
        return self._online.learned_similarity(
            [str(token or "") for token in list(query_tokens or []) if str(token or "")],
            [str(token or "") for token in list(candidate_tokens or []) if str(token or "")],
            limit=limit if limit is not None else self.online_scoring_token_limit,
        )

    def learned_vector_similarity(self, query_tokens: list[str], candidate_tokens: list[str], *, limit: int | None = None) -> dict:
        """
        Public read-only bridge for the learned vector coordinate space.

        This is intentionally separate from `learned_similarity`: the latter is
        association evidence, while this method reads the actual online learned
        token-vector coordinates introduced for Stage 6A.4.
        """

        if not self.online_enabled:
            return {"score": 0.0, "query_norm": 0.0, "candidate_norm": 0.0}
        return self._online.learned_vector_similarity(
            [str(token or "") for token in list(query_tokens or []) if str(token or "")],
            [str(token or "") for token in list(candidate_tokens or []) if str(token or "")],
            limit=limit if limit is not None else self.online_scoring_token_limit,
        )

    def learned_vector(self, tokens: list[str], *, limit: int | None = None) -> list[float]:
        """
        Public read-only bridge for one learned vector coordinate.

        Runtime attention uses this to memoize repeated vector reads within a
        tick. The only writer remains the MemoryStore snapshot/event learning
        pipeline; callers receive a copy and cannot mutate the online store.
        """

        if not self.online_enabled:
            return [0.0] * int(getattr(self._online, "dim", 64) or 64)
        return list(
            self._online.learned_vector(
                [str(token or "") for token in list(tokens or []) if str(token or "")],
                limit=limit if limit is not None else self.online_scoring_token_limit,
            )
        )

    def _build_query_features(self, query_items: list[dict], *, memory_kind: str = "") -> dict:
        selected_items = self._select_state_field_items(query_items, limit=self.query_feature_limit)
        anchor_items = self._select_anchor_items(query_items, limit=self.query_feature_limit)
        if not selected_items:
            selected_items = [dict(item) for item in list(query_items or [])[: self.query_feature_limit] if isinstance(item, dict)]
        # The state-field view is intentionally energy-sorted so whole-field Bn
        # recall can behave like intuition. Sequence posting is a different
        # channel: it should describe the perceived/focused order itself. Build
        # order tokens from the raw query field, not from the energy-sorted Bn
        # view, otherwise feelings/actions can shuffle "alpha beta gamma" into
        # a false sequence and hide valid posting:sequence evidence.
        sequence_items = [
            dict(item)
            for item in list(query_items or [])[: self.query_feature_limit]
            if isinstance(item, dict) and str(item.get("sa_label", "") or "")
        ]
        if not sequence_items:
            sequence_items = list(selected_items)
        feature_signature = self._query_items_signature(selected_items, memory_kind=str(memory_kind or ""))
        energy_signature = self._query_energy_signature(selected_items, memory_kind=str(memory_kind or ""))
        cached_features = self._query_feature_cache.get(feature_signature)
        if cached_features is not None:
            self._cache_stats["query_feature_hit"] += 1
            self._query_feature_cache.move_to_end(feature_signature)
            payload = self._copy_feature_payload(cached_features)
            payload["energy_signature"] = energy_signature
            payload["selected_items"] = [dict(item) for item in selected_items]
            return payload
        self._cache_stats["query_feature_miss"] += 1
        labels = [str(item.get("sa_label", "") or "") for item in selected_items if str(item.get("sa_label", "") or "")]
        displays = [str(item.get("display_text", "") or "") for item in selected_items if str(item.get("display_text", "") or "")]
        focus = self._ordered_focus_labels(selected_items, fallback_labels=labels)
        sequence_focus = self._ordered_focus_labels(sequence_items, fallback_labels=focus or labels)
        sequence = self._build_sequence_features(sequence_items, sequence_focus)
        relation_features = self._relations.build_features(
            memory_kind=str(memory_kind or ""),
            items=selected_items,
            focus_labels=focus,
        )
        bigrams = _bigrams(displays)
        sequence_bigrams = list(sequence.get("sequence_bigrams", []) or [])
        candidate_items = []
        seen_candidate_labels = set()
        # Candidate lookup keeps external anchors first so large-memory search
        # still has concrete perceptual/textual handles, then appends the same
        # all-SA field used by Bn scoring. This preserves AP's intuition path:
        # action/feeling/control labels are allowed to locate experience too.
        for item in list(anchor_items) + list(selected_items):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen_candidate_labels:
                continue
            seen_candidate_labels.add(label)
            candidate_items.append(item)
            if len(candidate_items) >= self.query_feature_limit:
                break
        candidate_labels = [str(item.get("sa_label", "") or "") for item in candidate_items if str(item.get("sa_label", "") or "")]
        candidate_displays = [str(item.get("display_text", "") or "") for item in candidate_items if str(item.get("display_text", "") or "")]
        candidate_bigrams = _bigrams(candidate_displays)
        candidate_sequence_bigrams = [
            token
            for token in sequence_bigrams
            if not str(token or "").startswith("focus_seq::")
        ]
        if str(memory_kind or "") == "state":
            # State Bn is "what current field resembles", not an exact replay of
            # this tick's internal ordering. Keep content/energy features as the
            # main definition and reserve explicit focus/order sequence features
            # for focus memory where order is the object being developed.
            candidate_sequence_bigrams = []
        payload = {
            "feature_signature": feature_signature,
            "selected_items": [dict(item) for item in selected_items],
            "labels": labels,
            "displays": displays,
            "bigrams": bigrams,
            "sequence_bigrams": sequence_bigrams,
            "relation_features": relation_features,
            "relation_tokens": list(relation_features.get("relation_tokens", []) or []),
            "relation_token_weights": dict(relation_features.get("relation_token_weights", {}) or {}),
            "relation_channels": dict(relation_features.get("relation_channels", {}) or {}),
            "focus_labels": focus,
            "vector_tokens": labels + displays + bigrams + sequence_bigrams + list(relation_features.get("relation_tokens", []) or []),
            "numeric_features": self._numeric_feature_profile(selected_items, limit=self.query_feature_limit),
            "candidate_labels": candidate_labels,
            "candidate_displays": candidate_displays,
            "candidate_bigrams": candidate_bigrams,
            "candidate_sequence_bigrams": candidate_sequence_bigrams,
            "candidate_focus_labels": [],
            "candidate_vector_tokens": candidate_labels + candidate_displays + candidate_bigrams + candidate_sequence_bigrams,
            "label_set": set(labels),
            "display_set": set(displays),
            "bigram_set": set(bigrams),
            "sequence_set": set(sequence_bigrams),
            "focus_set": set(focus),
            "energy_signature": energy_signature,
        }
        payload["candidate_signature"] = self._build_candidate_signature(payload)
        self._query_feature_cache[feature_signature] = payload
        self._bounded_ordered_dict(self._query_feature_cache, self._query_feature_cache_limit)
        return payload

    def _get_or_build_query_energy(self, energy_signature: str, query_items: list[dict]) -> tuple[dict[str, float], float, float, float]:
        clean = str(energy_signature or "")
        cached = self._query_energy_cache.get(clean)
        if cached is not None:
            self._cache_stats["query_energy_hit"] += 1
            self._query_energy_cache.move_to_end(clean)
            return cached
        self._cache_stats["query_energy_miss"] += 1
        energy = self._energy_profile(query_items, limit=self.query_feature_limit)
        mass = self._energy_mass(energy)
        real_mass, virtual_mass = self._query_real_virtual_mass(query_items, limit=self.query_feature_limit)
        payload = (energy, mass, real_mass, virtual_mass)
        self._query_energy_cache[clean] = payload
        self._bounded_ordered_dict(self._query_energy_cache, self._query_energy_cache_limit)
        return payload

    def _copy_feature_payload(self, payload: dict) -> dict:
        copied = {}
        for key, value in (payload or {}).items():
            if isinstance(value, set):
                copied[key] = set(value)
            elif isinstance(value, list):
                copied[key] = list(value)
            else:
                copied[key] = value
        return copied

    def _effective_runtime_tick(self, tick_index: int | float | None) -> int:
        try:
            tick = int(float(tick_index if tick_index is not None else 0))
        except (TypeError, ValueError):
            tick = 0
        offset = int(getattr(self, "_runtime_tick_offset", 0) or 0)
        if offset <= 0:
            return tick
        if tick >= offset:
            return tick
        return tick + offset

    def _apply_runtime_tick_offset_to_item(self, item: dict, *, local_tick_index: int, effective_tick_index: int) -> dict:
        stamped = dict(item)

        def _maybe_offset(value):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return value

        for key in ("tick_index", "last_seen_tick", "last_updated_tick"):
            if key not in stamped:
                if key == "tick_index":
                    stamped[key] = int(effective_tick_index)
                continue
            parsed = _maybe_offset(stamped.get(key))
            if isinstance(parsed, int) and parsed == int(local_tick_index):
                stamped[key] = int(effective_tick_index)
        meta = stamped.get("anchor_meta")
        if isinstance(meta, dict):
            clean_meta = dict(meta)
            for key in ("tick_index", "source_tick_index"):
                parsed = _maybe_offset(clean_meta.get(key))
                if isinstance(parsed, int) and parsed == int(local_tick_index):
                    clean_meta[key] = int(effective_tick_index)
            stamped["anchor_meta"] = clean_meta
        return stamped

    def _normalize_persisted_snapshot(self, row: dict) -> dict:
        snapshot = dict(row or {})
        snapshot["memory_id"] = str(snapshot.get("memory_id", "") or "")
        snapshot["memory_kind"] = str(snapshot.get("memory_kind", "") or "")
        snapshot["tick_index"] = int(snapshot.get("tick_index", 0) or 0)
        snapshot["items"] = [dict(item) for item in list(snapshot.get("items", []) or []) if isinstance(item, dict)]
        snapshot["focus_labels"] = [str(label or "") for label in list(snapshot.get("focus_labels", []) or []) if str(label or "")]
        snapshot["source_text"] = str(snapshot.get("source_text", "") or "")
        snapshot["successor_boundary"] = bool(snapshot.get("successor_boundary", False))
        snapshot["asset_refs"] = self._clean_asset_refs(snapshot.get("asset_refs", []) or [])
        if not isinstance(snapshot.get("sequence_features", None), dict):
            snapshot["sequence_features"] = self._build_sequence_features(snapshot["items"], snapshot["focus_labels"])
        if not isinstance(snapshot.get("state_field_items", None), list):
            snapshot["state_field_items"] = self._select_state_field_items(snapshot["items"], limit=self.core_item_limit)
        else:
            snapshot["state_field_items"] = [dict(item) for item in snapshot.get("state_field_items", []) if isinstance(item, dict)]
        if not isinstance(snapshot.get("anchor_items", None), list):
            snapshot["anchor_items"] = self._select_anchor_items(snapshot["items"], limit=self.core_item_limit)
        else:
            snapshot["anchor_items"] = [dict(item) for item in snapshot.get("anchor_items", []) if isinstance(item, dict)]
        if not isinstance(snapshot.get("core_items", None), list):
            snapshot["core_items"] = list(snapshot["anchor_items"])
        else:
            snapshot["core_items"] = [dict(item) for item in snapshot.get("core_items", []) if isinstance(item, dict)]
            if not snapshot["core_items"]:
                snapshot["core_items"] = list(snapshot["anchor_items"])
        if not isinstance(snapshot.get("relation_features", None), dict):
            snapshot["relation_features"] = self._relations.build_features(
                memory_kind=snapshot["memory_kind"],
                items=snapshot["state_field_items"],
                focus_labels=snapshot["focus_labels"],
            )
        return snapshot

    def _append_loaded_snapshot(
        self,
        *,
        snapshot: dict,
        features: dict,
        vector: list[float],
        energy_profile: dict[str, float],
        energy_mass: float,
        numeric_features: dict[str, list[float]],
        relation_features: dict,
        previous: dict | None,
        previous_episode: dict | None,
        process_indexes: bool,
        replay_learning: bool,
    ) -> None:
        memory_id = str(snapshot.get("memory_id", "") or "")
        memory_kind = str(snapshot.get("memory_kind", "") or "")
        self._snapshots.append(snapshot)
        self._snapshot_by_id[memory_id] = snapshot
        self._snapshot_features_by_id[memory_id] = features
        self._snapshot_energy_by_id[memory_id] = energy_profile
        self._snapshot_energy_mass_by_id[memory_id] = energy_mass
        self._snapshot_numeric_by_id[memory_id] = numeric_features
        self._snapshot_relations_by_id[memory_id] = relation_features
        self._register_label_document_frequencies(snapshot)
        bucket = self._recent_by_kind[memory_kind]
        bucket.append(snapshot)
        if len(bucket) > self.max_snapshots_per_kind:
            removed = bucket.pop(0)
            self._evict_snapshot(removed)
        max_global = max(self.max_snapshots_per_kind * 4, 512)
        if len(self._snapshots) > max_global:
            del self._snapshots[0 : len(self._snapshots) - max_global]
        self._transitions.register_snapshot(snapshot)
        previous_for_learning = None if bool(snapshot.get("successor_boundary", False)) else previous
        if previous_for_learning is not None:
            self._transitions.link_successor(memory_kind, str(previous.get("memory_id", "") or ""), memory_id)
            self._observe_successor_payload_support(previous_for_learning, snapshot)
            self._invalidate_successor_cache(memory_kind, str(previous.get("memory_id", "") or ""))
        if previous_episode is not None:
            previous_episode_id = str(previous_episode.get("memory_id", "") or "")
            if previous_episode_id:
                self._transitions.link_successor(self.EPISODE_SUCCESSOR_KIND, previous_episode_id, memory_id)
                if previous_episode is not previous_for_learning:
                    self._observe_successor_payload_support(previous_episode, snapshot)
                self._invalidate_successor_cache(self.EPISODE_SUCCESSOR_KIND, previous_episode_id)
        self._previous_by_kind[memory_kind] = snapshot
        self._previous_episode_snapshot = snapshot
        learned_vector = list((snapshot.get("vector_spaces", {}) or {}).get("online_learned_vector", []) or [])
        if not learned_vector and self.online_enabled:
            learned_vector = self._online.learned_vector(
                self._vector_tokens_for_index(features),
                limit=self.online_scoring_token_limit,
            )
        snapshot.setdefault("vector_spaces", {"hash_vector": list(vector), "online_learned_vector": list(learned_vector)})
        self._snapshot_learned_vector_by_id[memory_id] = list(learned_vector)
        self._queue_index_job(snapshot=snapshot, features=features, vector=vector, learned_vector=learned_vector, previous=previous_for_learning)
        if process_indexes:
            job = self._pending_index_jobs.get(memory_id)
            if job is not None:
                if replay_learning:
                    self._process_index_job_by_id(memory_id)
                else:
                    self._index_snapshot_job_without_learning(snapshot=snapshot, job=job)
                    self._pending_index_jobs.pop(memory_id, None)
                    self._update_pending_index_stats()
        self._touch_memory_revision()

    def _index_snapshot_job_without_learning(self, *, snapshot: dict, job: dict) -> None:
        saved_online = self.online_enabled
        self.online_enabled = False
        try:
            self._index_snapshot_job(snapshot=snapshot, job=job)
        finally:
            self.online_enabled = saved_online

    def _advance_next_id_from_memory_id(self, memory_id: str) -> None:
        clean = str(memory_id or "")
        if not clean.startswith("mem-"):
            return
        try:
            numeric = int(clean.split("mem-", 1)[1])
        except ValueError:
            return
        self._next_id = max(int(self._next_id), numeric + 1)

    def _persist_snapshot_authoritative(
        self,
        *,
        snapshot: dict,
        features: dict,
        vector: list[float],
        learned_vector: list[float],
        energy_profile: dict[str, float],
        energy_mass: float,
        numeric_features: dict[str, list[float]],
        relation_features: dict,
        previous_memory_id: str,
        transition_edges: list[dict] | None = None,
    ) -> None:
        """
        Send a complete white-box memory event to the authoritative store.

        The persistence layer is not allowed to decide cognition. It receives
        the same snapshot/features that runtime indexes use, so long-term DB
        state can rebuild or audit those indexes later. By default persistence
        failures are recorded but do not break the tick; strict experiments can
        enable persistence_required to make durable writes mandatory.
        """

        result = self._persistence.write_snapshot(
            snapshot=snapshot,
            features=features,
            vector=vector,
            energy_profile=energy_profile,
            energy_mass=energy_mass,
            numeric_features=numeric_features,
            relation_features=relation_features,
            previous_memory_id=previous_memory_id,
            transition_edges=transition_edges,
            learned_vector=learned_vector,
        )
        if result.ok:
            self._persistence_write_count += 1
            return
        self._persistence_error_count += 1
        self._last_persistence_error = str(result.error or "persistence_write_failed")
        if self._persistence_required:
            raise RuntimeError(self._last_persistence_error)

    def _runtime_state_payload(self, *, reason: str) -> dict:
        state = self._online.export_state()
        state["reason"] = str(reason or "")
        state["memory_revision"] = int(self._memory_revision)
        state["persist_suspended"] = int(self._runtime_state_persist_suspended)
        state["last_persisted_revision"] = int(self._runtime_state_last_persist_revision)
        state["relation_restore_high_watermark_tick"] = int(self._runtime_relation_restore_high_watermark_tick)
        return state

    def _persist_runtime_state(self, *, reason: str) -> dict:
        state = self._runtime_state_payload(reason=reason)
        writer = getattr(self._persistence, "write_runtime_state", None)
        if not callable(writer):
            return {"schema_id": "apv21_runtime_state_persist/v1", "written": False, "reason": "persistence_adapter_has_no_runtime_state_writer"}
        result = writer(state=state)
        if result.ok:
            self._runtime_state_last_persist_revision = int(self._memory_revision)
            self._runtime_state_dirty = False
        return {
            "schema_id": "apv21_runtime_state_persist/v1",
            "written": bool(result.ok),
            "backend": getattr(result, "backend", ""),
            "rows_written": int(getattr(result, "rows_written", 0) or 0),
            "reason": str(reason or ""),
            "error": str(getattr(result, "error", "") or ""),
        }

    def flush_runtime_state(self, *, reason: str = "explicit_flush") -> dict:
        if self._runtime_state_persist_suspended > 0:
            return {
                "schema_id": "apv21_runtime_state_flush/v1",
                "written": False,
                "reason": "runtime_state_persist_suspended",
                "memory_revision": int(self._memory_revision),
            }
        if not self._runtime_state_dirty and self._runtime_state_last_persist_revision == self._memory_revision:
            return {
                "schema_id": "apv21_runtime_state_flush/v1",
                "written": False,
                "reason": "runtime_state_clean",
                "memory_revision": int(self._memory_revision),
            }
        trace = self._persist_runtime_state(reason=reason)
        trace["schema_id"] = "apv21_runtime_state_flush/v1"
        return trace

    def _restore_runtime_state_from_persistence(self) -> dict:
        loader = getattr(self._persistence, "load_runtime_state", None)
        if not callable(loader):
            return {"schema_id": "apv21_runtime_state_restore/v1", "restored": False, "reason": "persistence_adapter_has_no_runtime_state_loader"}
        try:
            payload = loader()
        except Exception as exc:
            self._last_persistence_error = str(exc)
            self._persistence_error_count += 1
            self._runtime_state_dirty = False
            return {"schema_id": "apv21_runtime_state_restore/v1", "restored": False, "reason": "loader_error", "error": str(exc)}
        restored = self._online.import_state(payload)
        self._runtime_state_last_persist_revision = int((payload or {}).get("last_persisted_revision", -1) or -1)
        self._runtime_relation_restore_high_watermark_tick = int(
            (payload or {}).get(
                "relation_restore_high_watermark_tick",
                (payload or {}).get("current_tick", -1),
            )
            or -1
        )
        self._runtime_state_dirty = False
        return {
            "schema_id": "apv21_runtime_state_restore/v1",
            "restored": bool(restored.get("restored", False)),
            "entry_count": int(restored.get("entry_count", 0) or 0),
            "promoted_count": int(restored.get("promoted_count", 0) or 0),
            "current_tick": int(restored.get("current_tick", -1) or -1),
            "relation_restore_high_watermark_tick": int(self._runtime_relation_restore_high_watermark_tick),
            "token_limit": int(restored.get("token_limit", self._online.token_limit)),
            "min_support_to_promote": int(restored.get("min_support_to_promote", self._online.min_support_to_promote)),
        }

    def _runtime_state_restore_trace(self) -> dict:
        return {
            "schema_id": "apv21_runtime_state_restore/v1",
            "restored": False,
            "reason": "not_attempted",
        }

    def _query_items_signature(self, items: list[dict], *, memory_kind: str = "") -> str:
        import hashlib

        hasher = hashlib.blake2b(digest_size=16)
        kind = str(memory_kind or "")
        hasher.update(f"K|{kind}\n".encode("utf-8"))
        rows = [item for item in (items or []) if isinstance(item, dict)]
        if kind == "state" and len(rows) > self.state_query_signature_token_limit:
            rows = rows[: self.state_query_signature_token_limit]
        for item in rows:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            hasher.update(label.encode("utf-8"))
            hasher.update(b"|")
            hasher.update(str(item.get("family", "") or "").encode("utf-8"))
            hasher.update(b"|")
            hasher.update(str(item.get("source_type", "") or "").encode("utf-8"))
            hasher.update(b"|")
            hasher.update(str(item.get("position", "") or "").encode("utf-8"))
            hasher.update(b"|")
            numeric = self._extract_numeric_features(item)
            if numeric:
                for channel, vector in sorted(numeric.items()):
                    hasher.update(f"N|{channel}|".encode("utf-8"))
                    for value in vector[:8]:
                        hasher.update(str(round(float(value or 0.0), 3)).encode("utf-8"))
                        hasher.update(b",")
                    hasher.update(b"|")
            if kind == "focus":
                hasher.update(b"1" if bool(item.get("is_focus", False)) else b"0")
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _query_energy_signature(self, items: list[dict], *, memory_kind: str = "") -> str:
        import hashlib

        hasher = hashlib.blake2b(digest_size=16)
        kind = str(memory_kind or "")
        hasher.update(f"K|{kind}\n".encode("utf-8"))
        rows = [item for item in (items or []) if isinstance(item, dict)]
        if len(rows) > self.query_feature_limit:
            rows = rows[: self.query_feature_limit]
        for item in rows:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            hasher.update(label.encode("utf-8"))
            hasher.update(b"|")
            for key in ("query_weight", "real_energy", "virtual_energy", "cognitive_pressure", "attention_gain"):
                try:
                    value = round(float(item.get(key, 0.0) or 0.0), 3)
                except (TypeError, ValueError):
                    value = 0.0
                hasher.update(f"{key}={value}|".encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _build_snapshot_features(self, snapshot: dict) -> dict:
        items = self._snapshot_state_field_items(snapshot)
        if not items:
            items = self._select_state_field_items(list(snapshot.get("items", []) or []), limit=self.core_item_limit)
        labels = [str(item.get("sa_label", "") or "") for item in items if str(item.get("sa_label", "") or "")]
        displays = [str(item.get("display_text", "") or "") for item in items if str(item.get("display_text", "") or "")]
        focus_labels = [str(label or "") for label in (snapshot.get("focus_labels", []) or []) if str(label or "")]
        sequence = dict(snapshot.get("sequence_features", {}) or self._build_sequence_features(items, focus_labels))
        sequence_bigrams = list(sequence.get("sequence_bigrams", []) or [])
        relation_features = dict(snapshot.get("relation_features", {}) or self._relations.build_features(
            memory_kind=str(snapshot.get("memory_kind", "") or ""),
            items=items,
            focus_labels=focus_labels,
        ))
        bigrams = _bigrams(displays)
        return {
            "labels": labels,
            "displays": displays,
            "bigrams": bigrams,
            "sequence_bigrams": sequence_bigrams,
            "relation_features": relation_features,
            "relation_tokens": list(relation_features.get("relation_tokens", []) or []),
            "relation_token_weights": dict(relation_features.get("relation_token_weights", {}) or {}),
            "relation_channels": dict(relation_features.get("relation_channels", {}) or {}),
            "focus_labels": focus_labels,
            "vector_tokens": labels + displays + focus_labels + bigrams + sequence_bigrams + list(relation_features.get("relation_tokens", []) or []),
            "numeric_features": self._numeric_feature_profile(items, limit=self.core_item_limit),
            "label_set": set(labels),
            "display_set": set(displays),
            "bigram_set": set(bigrams),
            "sequence_set": set(sequence_bigrams),
            "focus_set": set(focus_labels),
        }

    def _register_label_document_frequencies(self, snapshot: dict) -> None:
        kind = str((snapshot or {}).get("memory_kind", "") or "")
        memory_id = str((snapshot or {}).get("memory_id", "") or "")
        if not kind or not memory_id:
            return
        features = self._snapshot_features_by_id.get(memory_id) or self._build_snapshot_features(snapshot)
        for label in set(features.get("labels", []) or []):
            clean = str(label or "").strip()
            if clean:
                self._label_document_frequency_by_kind[kind][clean] += 1
        self._document_count_by_kind[kind] += 1
        self._register_token_document_frequencies(kind=kind, features=features, delta=1)

    def _unregister_label_document_frequencies(self, snapshot: dict) -> None:
        kind = str((snapshot or {}).get("memory_kind", "") or "")
        memory_id = str((snapshot or {}).get("memory_id", "") or "")
        if not kind or not memory_id:
            return
        features = self._snapshot_features_by_id.get(memory_id) or self._build_snapshot_features(snapshot)
        counter = self._label_document_frequency_by_kind.get(kind)
        if counter is None:
            return
        for label in set(features.get("labels", []) or []):
            clean = str(label or "").strip()
            if not clean:
                continue
            counter[clean] -= 1
            if counter[clean] <= 0:
                counter.pop(clean, None)
        if self._document_count_by_kind.get(kind, 0) > 0:
            self._document_count_by_kind[kind] -= 1
        self._register_token_document_frequencies(kind=kind, features=features, delta=-1)

    def _register_token_document_frequencies(self, *, kind: str, features: dict, delta: int) -> None:
        field_map = {
            "display": "displays",
            "bigram": "bigrams",
            "sequence": "sequence_bigrams",
        }
        for field_name, feature_key in field_map.items():
            counter = self._token_document_frequency_by_kind_field[(str(kind), field_name)]
            for token in set(features.get(feature_key, []) or []):
                clean = str(token or "").strip()
                if not clean:
                    continue
                counter[clean] += int(delta)
                if counter[clean] <= 0:
                    counter.pop(clean, None)

    def _weighted_label_overlap(
        self,
        *,
        memory_kind: str,
        query_items: list[dict],
        query_label_set: set[str],
        snapshot_label_set: set[str],
    ) -> dict:
        matched = sorted(str(label) for label in (query_label_set & snapshot_label_set) if str(label or ""))
        if not matched:
            return {"score": 0.0, "matches": []}
        query_by_label = {
            str(item.get("sa_label", "") or ""): dict(item)
            for item in query_items or []
            if isinstance(item, dict) and str(item.get("sa_label", "") or "")
        }
        kind = str(memory_kind or "")
        total = max(1, self._document_count_by_kind.get(kind, 0) or len(self._recent_by_kind.get(kind, [])))
        counter = self._label_document_frequency_by_kind.get(kind, Counter())
        # First pass: per-label specificity-weighted contribution.
        contribs: list[tuple[float, dict]] = []
        for label in matched:
            frequency = max(1, int(counter.get(label, 1) or 1))
            specificity = self._label_specificity(total=total, frequency=frequency)
            salience = self._query_label_salience(query_by_label.get(label, {}))
            contribution = specificity * salience
            contribs.append((
                contribution,
                {
                    "label": label,
                    "frequency": frequency,
                    "specificity": _round4(specificity),
                    "query_salience": _round4(salience),
                    "contribution": _round4(contribution),
                },
            ))
        # Saturating accumulation (AP red-line 2): recall must be driven by
        # prediction specificity, not by how many labels happen to overlap. A
        # plain sum lets a snapshot that shares 20+ low-specificity generic SA
        # (action::/feeling::/action_feedback:: residue) out-score a skill
        # snapshot that shares a few high-specificity dialogue/math anchors.
        # We sort contributions descending and apply a geometric rank decay so
        # the high-specificity head keeps near-full weight while the long tail
        # of low-specificity matches saturates (bounded by max/(1-decay)).
        # This is soft and universal -- every label still contributes, just
        # with diminishing marginal weight by rank -- not a blacklist.
        contribs.sort(key=lambda pair: pair[0], reverse=True)
        score = 0.0
        rows = []
        for rank, (contribution, row) in enumerate(contribs):
            effective = contribution * (self._label_overlap_rank_decay ** rank)
            score += effective
            if len(rows) < 16:
                row["effective_contribution"] = _round4(effective)
                row["rank"] = rank
                rows.append(row)
        return {"score": _round4(score), "matches": rows}

    def _weighted_token_overlap(
        self,
        *,
        memory_kind: str,
        field_name: str,
        query_tokens: set[str],
        snapshot_tokens: set[str],
    ) -> dict:
        matched = sorted(str(token) for token in (query_tokens & snapshot_tokens) if str(token or ""))
        if not matched:
            return {"score": 0.0, "matches": []}
        kind = str(memory_kind or "")
        total = max(1, self._document_count_by_kind.get(kind, 0) or len(self._recent_by_kind.get(kind, [])))
        counter = self._token_document_frequency_by_kind_field.get((kind, str(field_name or "")), Counter())
        # Same saturating accumulation as _weighted_label_overlap: short inputs
        # (e.g. "你好") share many generic character bigrams; a plain sum lets a
        # residue snapshot win by matching a long tail of low-specificity
        # tokens. Sort by specificity descending and apply rank decay so the
        # tail saturates while distinctive tokens keep full weight.
        scored = []
        for token in matched:
            frequency = max(1, int(counter.get(token, 1) or 1))
            specificity = self._label_specificity(total=total, frequency=frequency)
            scored.append((specificity, token, frequency))
        scored.sort(key=lambda t: t[0], reverse=True)
        score = 0.0
        rows = []
        for rank, (specificity, token, frequency) in enumerate(scored):
            effective = specificity * (self._label_overlap_rank_decay ** rank)
            score += effective
            if len(rows) < 12:
                rows.append({
                    "token": token,
                    "frequency": frequency,
                    "specificity": _round4(specificity),
                    "rank": rank,
                    "effective": _round4(effective),
                })
        return {"score": _round4(score), "matches": rows}

    def _label_specificity(self, *, total: int, frequency: int) -> float:
        total = max(1, int(total))
        frequency = max(1, int(frequency))
        rarity = exp(-float(frequency - 1) / max(1.0, float(total) * 0.18))
        # Wider dynamic range so ubiquitous generic labels (action::wait,
        # feeling::*) sink toward ~0 while rare high-value anchors stay high.
        # A near-zero floor lets prediction specificity -- not shared-label
        # count -- drive recall, per AP philosophy. Not a blacklist: low-
        # specificity labels still contribute, just proportionally to how
        # little they discriminate.
        return _clamp(0.04 + rarity * 1.56, 0.02, 1.6)

    def _specificity_map_for(self, *, memory_kind: str, labels) -> dict[str, float]:
        """Precompute label -> IDF specificity for a set of labels (the query
        field), so energy/label overlap can be specificity-weighted without
        re-deriving IDF per candidate. Shares the same (total, frequency) sample
        space as _weighted_label_overlap so weighting is consistent."""
        kind = str(memory_kind or "")
        total = max(1, self._document_count_by_kind.get(kind, 0) or len(self._recent_by_kind.get(kind, [])))
        counter = self._label_document_frequency_by_kind.get(kind, Counter())
        out: dict[str, float] = {}
        for label in labels or ():
            clean = str(label or "")
            if not clean or clean in out:
                continue
            frequency = max(1, int(counter.get(clean, 1) or 1))
            out[clean] = self._label_specificity(total=total, frequency=frequency)
        return out

    def _query_label_salience(self, item: dict) -> float:
        if not item:
            return 1.0
        label = str(item.get("sa_label", "") or "")
        source_type = str(item.get("source_type", "") or "")
        family = str(item.get("family", "") or "")
        real = max(0.0, float(item.get("real_energy", 0.0) or 0.0))
        virtual = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
        query = max(0.0, float(item.get("query_weight", 0.0) or 0.0))
        attention = max(0.0, float(item.get("attention_gain", item.get("attention_weight", 0.0)) or 0.0))
        # Attention salience follows positive unresolved pressure: strong
        # current real evidence should draw Bn, while a virtual-only residue is
        # mainly Cn/background expectation and should not become "what I am
        # seeing now" merely because it was predicted again.
        pressure = max(0.0, float(item.get("cognitive_pressure", real - virtual) or 0.0))
        field_weight = real + virtual * 0.24 + query * 0.42 + attention * 0.34 + pressure * 0.36
        base = 0.72 + field_weight * 0.12
        scale = 1.0
        currentness = dict(item.get("query_currentness", {}) or {}) if isinstance(item.get("query_currentness", {}), dict) else {}
        if bool(currentness.get("new_external_turn_residue_softened", False)) or bool(currentness.get("active_external_turn_residue_softened", False)):
            scale *= _clamp(float(currentness.get("factor", 1.0) or 1.0), 0.18, 1.0)
        if source_type in {"context_background"} or family == "context_background" or label.startswith(("domain::", "style::", "learning_phase::")):
            scale *= 0.32
        if source_type.startswith("predicted"):
            scale *= 0.38 if real <= 0.05 else 0.58
        if source_type in {"sensory_echo", "thought_echo", "internal_reply_trace"} or family in {"short_term_echo"}:
            scale *= 0.45
        if source_type == "short_term_slot" or family == "short_term_slot":
            scale *= 0.68
        if source_type == "process_feeling" or family == "process_feeling":
            scale *= 1.28
        return _clamp(base * scale, 0.14, 1.85)

    def _merge_candidates(self, posting_rows: list[dict], vector_rows: list[dict], numeric_rows: list[dict] | None = None) -> list[dict]:
        merged: dict[str, dict] = {}
        for row in posting_rows:
            memory_id = str(row.get("memory_id", "") or "")
            if not memory_id:
                continue
            merged[memory_id] = {
                "memory_id": memory_id,
                "posting_score": float(row.get("posting_score", 0.0) or 0.0),
                "posting_specificity_score": float(row.get("posting_specificity_score", row.get("posting_score", 0.0)) or 0.0),
                "vector_score": 0.0,
                "numeric_score": 0.0,
                "numeric_score_breakdown": {},
                "candidate_sources": list(row.get("candidate_sources", []) or []),
                "matched_tokens": dict(row.get("matched_tokens", {}) or {}),
                "matched_token_weights": dict(row.get("matched_token_weights", {}) or {}),
            }
        for row in vector_rows:
            memory_id = str(row.get("memory_id", "") or "")
            if not memory_id:
                continue
            bucket = merged.setdefault(
                memory_id,
                {
                    "memory_id": memory_id,
                    "posting_score": 0.0,
                    "vector_score": 0.0,
                    "numeric_score": 0.0,
                    "numeric_score_breakdown": {},
                    "candidate_sources": [],
                    "matched_tokens": {},
                    "matched_token_weights": {},
                },
            )
            bucket["vector_score"] = max(float(bucket.get("vector_score", 0.0) or 0.0), float(row.get("vector_score", 0.0) or 0.0))
            # Capture learned-vector candidate signal so learned-only candidates
            # (posting=vector=0) are not sorted to the bottom and cut before
            # scoring. The scoring loop recomputes the precise learned_vector_score.
            if "learned_vector_score" in row:
                bucket["learned_vector_score"] = max(
                    float(bucket.get("learned_vector_score", 0.0) or 0.0),
                    float(row.get("learned_vector_score", 0.0) or 0.0),
                )
            for source in row.get("candidate_sources", []) or []:
                if source not in bucket["candidate_sources"]:
                    bucket["candidate_sources"].append(source)
        for row in numeric_rows or []:
            memory_id = str(row.get("memory_id", "") or "")
            if not memory_id:
                continue
            bucket = merged.setdefault(
                memory_id,
                {
                    "memory_id": memory_id,
                    "posting_score": 0.0,
                    "vector_score": 0.0,
                    "numeric_score": 0.0,
                    "numeric_score_breakdown": {},
                    "candidate_sources": [],
                    "matched_tokens": {},
                    "matched_token_weights": {},
                },
            )
            bucket["numeric_score"] = max(float(bucket.get("numeric_score", 0.0) or 0.0), float(row.get("numeric_score", 0.0) or 0.0))
            breakdown = dict(bucket.get("numeric_score_breakdown", {}) or {})
            for channel, score in dict(row.get("numeric_score_breakdown", {}) or {}).items():
                clean_channel = str(channel or "")
                if not clean_channel:
                    continue
                breakdown[clean_channel] = max(float(breakdown.get(clean_channel, 0.0) or 0.0), float(score or 0.0))
            bucket["numeric_score_breakdown"] = breakdown
            for source in row.get("candidate_sources", []) or []:
                if source not in bucket["candidate_sources"]:
                    bucket["candidate_sources"].append(source)
        ordered = list(merged.values())
        ordered.sort(
            key=lambda item: (
                -(
                    float(item.get("posting_score", 0.0) or 0.0)
                    + float(item.get("vector_score", 0.0) or 0.0)
                    + float(item.get("numeric_score", 0.0) or 0.0)
                    + float(item.get("learned_vector_score", 0.0) or 0.0) * self._learned_vector_candidate_weight
                ),
                str(item.get("memory_id", "") or ""),
            )
        )
        return ordered[: self.candidate_limit]

    def audit_recall(
        self,
        query_items: list[dict],
        *,
        memory_kind: str,
        top_k: int | None = None,
        exact_limit: int | None = None,
        time_context: dict | None = None,
    ) -> dict:
        """
        Compare runtime bounded recall against an exact hot-memory audit.

        Runtime recall is allowed to use ANN and cached candidate pruning. The
        audit path deliberately scores every currently resident snapshot of the
        requested kind. This is not the tick hot path; it is a white-box check
        used to tune ANN/posting/numeric/relation budgets and to catch high-score
        candidates that the fast candidate layer missed.
        """

        limit = self.recall_top_k if top_k is None else max(1, int(top_k))
        audit_limit = max(limit, int(exact_limit or max(limit * 3, 8)))
        runtime_rows = self.recall(query_items, memory_kind=str(memory_kind), top_k=limit, time_context=time_context)
        exact_rows = self._exact_hot_recall(query_items, memory_kind=str(memory_kind), top_k=audit_limit, time_context=time_context)
        runtime_ids = [str(row.get("memory_id", "") or "") for row in runtime_rows if str(row.get("memory_id", "") or "")]
        exact_ids = [str(row.get("memory_id", "") or "") for row in exact_rows if str(row.get("memory_id", "") or "")]
        runtime_set = set(runtime_ids)
        exact_top_set = set(exact_ids[:limit])
        overlap = [memory_id for memory_id in runtime_ids if memory_id in exact_top_set]
        missing = [row for row in exact_rows[:limit] if str(row.get("memory_id", "") or "") not in runtime_set]
        runtime_only = [row for row in runtime_rows if str(row.get("memory_id", "") or "") not in set(exact_ids[:audit_limit])]
        return {
            "schema_id": "apv21_recall_index_audit/v1",
            "memory_kind": str(memory_kind or ""),
            "top_k": int(limit),
            "exact_limit": int(audit_limit),
            "runtime_ids": runtime_ids,
            "exact_ids": exact_ids[:audit_limit],
            "overlap_at_k": len(overlap),
            "overlap_ratio_at_k": _round4(len(overlap) / float(max(1, limit))),
            "missing_exact_high_score": [self._audit_row_preview(row) for row in missing],
            "runtime_only": [self._audit_row_preview(row) for row in runtime_only],
            "runtime_rows": [self._audit_row_preview(row) for row in runtime_rows],
            "exact_rows": [self._audit_row_preview(row) for row in exact_rows[:audit_limit]],
            "candidate_layer": {
                "ann_summary": self.ann_summary(),
                "numeric_summary": self.numeric_summary(),
                "cache_stats": dict(self._cache_stats),
            },
            "meaning": "audit_is_not_tick_hot_path;it_checks_whether_fast_candidates_miss_good_B_objects",
        }

    def _exact_hot_recall(self, query_items: list[dict], *, memory_kind: str, top_k: int, time_context: dict | None = None) -> list[dict]:
        bucket = self._recent_by_kind.get(str(memory_kind or ""), [])
        if not bucket:
            return []
        query_features = self._build_query_features(query_items, memory_kind=str(memory_kind or ""))
        current_tick = self._current_tick_for_temporal(query_items, time_context=time_context)
        query_energy, query_mass, query_real_mass, query_virtual_mass = self._get_or_build_query_energy(
            query_features.get("energy_signature", self._query_energy_signature(query_items, memory_kind=str(memory_kind or ""))),
            query_items,
        )
        rows = []
        for snapshot in bucket:
            score_row = self._score_snapshot_exact(
                snapshot,
                memory_kind=str(memory_kind or ""),
                query_features=query_features,
                query_energy=query_energy,
                query_mass=query_mass,
                time_context=time_context,
                current_tick=current_tick,
            )
            if score_row["score"] <= 0.0:
                continue
            score_row["candidate_sources"] = ["exact_hot_memory_audit"]
            rows.append(score_row)
        rows.sort(key=lambda item: (-float(item.get("score", 0.0) or 0.0), -int(item.get("tick_index", -1) or -1), str(item.get("memory_id", "") or "")))
        return self._annotate_recall_energy(
            rows[: max(1, int(top_k))],
            query_mass=query_mass,
            query_real_mass=query_real_mass,
            query_virtual_mass=query_virtual_mass,
        )

    def _score_snapshot_exact(
        self,
        snapshot: dict,
        *,
        memory_kind: str,
        query_features: dict,
        query_energy: dict[str, float],
        query_mass: float,
        time_context: dict | None,
        current_tick: int | None,
    ) -> dict:
        snapshot_features = self._snapshot_features_by_id.get(snapshot["memory_id"]) or self._build_snapshot_features(snapshot)
        query_label_set = query_features.get("label_set", set())
        query_display_set = query_features.get("display_set", set())
        query_bigram_set = query_features.get("bigram_set", set())
        query_focus_set = query_features.get("focus_set", set())
        query_sequence_set = query_features.get("sequence_set", set())
        snapshot_label_set = snapshot_features.get("label_set", set(snapshot_features["labels"]))
        snapshot_display_set = snapshot_features.get("display_set", set(snapshot_features["displays"]))
        snapshot_bigram_set = snapshot_features.get("bigram_set", set(snapshot_features["bigrams"]))
        snapshot_focus_set = snapshot_features.get("focus_set", set(snapshot_features["focus_labels"]))
        snapshot_sequence_set = snapshot_features.get("sequence_set", set(snapshot_features["sequence_bigrams"]))
        label_overlap = len(query_label_set & snapshot_label_set)
        weighted_label = self._weighted_label_overlap(
            memory_kind=str(snapshot.get("memory_kind", "") or ""),
            query_items=list(query_features.get("selected_items", []) or []),
            query_label_set=query_label_set,
            snapshot_label_set=snapshot_label_set,
        )
        display_overlap = len(query_display_set & snapshot_display_set)
        weighted_display_overlap = self._weighted_token_overlap(
            memory_kind=str(snapshot.get("memory_kind", "") or ""),
            field_name="display",
            query_tokens=query_display_set,
            snapshot_tokens=snapshot_display_set,
        )
        bigram_overlap = len(query_bigram_set & snapshot_bigram_set)
        weighted_bigram_overlap = self._weighted_token_overlap(
            memory_kind=str(snapshot.get("memory_kind", "") or ""),
            field_name="bigram",
            query_tokens=query_bigram_set,
            snapshot_tokens=snapshot_bigram_set,
        )
        focus_overlap = len(query_focus_set & snapshot_focus_set)
        sequence_overlap = len(query_sequence_set & snapshot_sequence_set)
        weighted_sequence_overlap = self._weighted_token_overlap(
            memory_kind=str(snapshot.get("memory_kind", "") or ""),
            field_name="sequence",
            query_tokens=query_sequence_set,
            snapshot_tokens=snapshot_sequence_set,
        )
        snapshot_energy = self._snapshot_energy_by_id.get(snapshot["memory_id"]) or self._energy_profile(snapshot.get("items", []), limit=self.core_item_limit)
        snapshot_mass = self._snapshot_energy_mass_by_id.get(snapshot["memory_id"])
        if snapshot_mass is None:
            snapshot_mass = self._energy_mass(snapshot_energy)
        state_match = min(query_mass, snapshot_mass) / max(1.0, max(query_mass, snapshot_mass))
        energy_overlap = self._energy_overlap(
            query_energy, snapshot_energy, query_mass=query_mass, snapshot_mass=snapshot_mass,
            specificity_by_label=self._specificity_map_for(memory_kind=str(memory_kind), labels=query_energy.keys()),
        )
        query_vector = self._embedder.embed(self._candidate_vector_tokens_for_index(query_features))
        snapshot_vector = self._vector_cache.get(snapshot["memory_id"])
        if snapshot_vector is None:
            snapshot_vector = self._embedder.embed(self._vector_tokens_for_index(snapshot_features))
        vector_score = sum(a * b for a, b in zip(query_vector, snapshot_vector))
        vector_score = max(0.0, float(vector_score))
        query_learned_vector = self._online.learned_vector(
            query_features["vector_tokens"],
            limit=self.online_scoring_token_limit,
        ) if self.online_enabled else []
        snapshot_learned_vector = self._snapshot_learned_vector_by_id.get(snapshot["memory_id"])
        if snapshot_learned_vector is None:
            snapshot_learned_vector = list((snapshot.get("vector_spaces", {}) or {}).get("online_learned_vector", []) or [])
        learned_vector_score = (
            sum(a * b for a, b in zip(query_learned_vector, snapshot_learned_vector))
            if query_learned_vector and snapshot_learned_vector
            else 0.0
        )
        learned_vector_score = max(0.0, float(learned_vector_score))
        relation = self._relations.score(
            memory_kind=str(snapshot.get("memory_kind", "") or ""),
            query_features=dict(query_features.get("relation_features", {}) or {}),
            candidate_memory_id=snapshot["memory_id"],
        )
        relation_score = float(relation.get("score", 0.0) or 0.0)
        learned = (
            self._online.learned_similarity(
                query_features["vector_tokens"],
                snapshot_features["vector_tokens"],
                limit=self.online_scoring_token_limit,
            )
            if self.online_enabled
            else {"score": 0.0, "contributions": []}
        )
        learned_score = float(learned.get("score", 0.0) or 0.0)
        time_match = self._time_match(snapshot=snapshot, time_context=time_context)
        score_before_temporal = (
            weighted_label["score"] * 1.15
            + weighted_display_overlap["score"] * 0.45
            + weighted_bigram_overlap["score"] * 0.9
            + focus_overlap * 0.7
            + state_match * 0.55
            + energy_overlap * 1.35
            + weighted_sequence_overlap["score"] * 0.8
            + vector_score * 0.4
            + relation_score
            + learned_score * self.learned_weight
            + time_match
        )
        temporal = self._temporal_applicability(snapshot, current_tick=current_tick)
        score = score_before_temporal * float(temporal.get("weight", 1.0) or 1.0)
        return {
            "memory_id": snapshot["memory_id"],
            "tick_index": snapshot["tick_index"],
            "query_tick": current_tick,
            "memory_kind": snapshot["memory_kind"],
            "score": _round4(score),
            "raw_score": _round4(score),
            "score_before_temporal": _round4(score_before_temporal),
            "temporal_age_ticks": temporal.get("age_ticks"),
            "temporal_applicability": _round4(float(temporal.get("weight", 1.0) or 1.0)),
            "temporal_applicability_phase": str(temporal.get("phase", "") or ""),
            "temporal_applicability_policy": str(temporal.get("policy", "") or ""),
            "label_overlap": label_overlap,
            "weighted_label_overlap": _round4(float(weighted_label["score"])),
            "weighted_label_matches": weighted_label["matches"],
            "display_overlap": display_overlap,
            "weighted_display_overlap": _round4(float(weighted_display_overlap["score"])),
            "bigram_overlap": bigram_overlap,
            "weighted_bigram_overlap": _round4(float(weighted_bigram_overlap["score"])),
            "focus_overlap": focus_overlap,
            "state_match": _round4(state_match),
            "energy_overlap": _round4(energy_overlap),
            "sequence_overlap": sequence_overlap,
            "weighted_sequence_overlap": _round4(float(weighted_sequence_overlap["score"])),
            "posting_score": 0.0,
            "vector_score": _round4(vector_score),
            "learned_vector_score": _round4(learned_vector_score),
            "numeric_score": 0.0,
            "numeric_score_breakdown": {},
            "relative_relation_score": _round4(relation_score),
            "relative_relation_raw_score": _round4(float(relation.get("raw_score", 0.0) or 0.0)),
            "relation_channels": {
                str(key): _round4(value)
                for key, value in sorted(dict(relation.get("relation_channels", {}) or {}).items())
            },
            "relation_matches": list(relation.get("relation_matches", []) or []),
            "learned_score": _round4(learned_score),
            "time_match": _round4(time_match),
            "learned_contributions": list(learned.get("contributions", []) or []),
            "matched_tokens": {},
            "score_breakdown": {
                "label_overlap": label_overlap,
                "weighted_label_overlap": _round4(float(weighted_label["score"])),
                "weighted_label_matches": weighted_label["matches"],
                "display_overlap": display_overlap,
                "weighted_display_overlap": _round4(float(weighted_display_overlap["score"])),
                "weighted_display_matches": weighted_display_overlap["matches"],
                "bigram_overlap": bigram_overlap,
                "weighted_bigram_overlap": _round4(float(weighted_bigram_overlap["score"])),
                "weighted_bigram_matches": weighted_bigram_overlap["matches"],
                "focus_overlap": focus_overlap,
                "state_match": _round4(state_match),
                "energy_overlap": _round4(energy_overlap),
                "sequence_overlap": sequence_overlap,
                "weighted_sequence_overlap": _round4(float(weighted_sequence_overlap["score"])),
                "weighted_sequence_matches": weighted_sequence_overlap["matches"],
                "vector_score": _round4(vector_score),
                "learned_vector_score": _round4(learned_vector_score),
                "relative_relation_score": _round4(relation_score),
                "learned_score": _round4(learned_score),
                "time_match": _round4(time_match),
                "score_before_temporal": _round4(score_before_temporal),
                "temporal_applicability": _round4(float(temporal.get("weight", 1.0) or 1.0)),
                "temporal_age_ticks": temporal.get("age_ticks"),
                "temporal_phase": str(temporal.get("phase", "") or ""),
                "audit_policy": "exact_scores_all_resident_snapshots_for_this_kind",
            },
            "source_text": str(snapshot.get("source_text", "") or ""),
            "snapshot_ref": {
                "memory_id": snapshot["memory_id"],
                "tick_index": int(snapshot.get("tick_index", -1) or -1),
                "memory_kind": str(snapshot.get("memory_kind", "") or ""),
                "source_text": str(snapshot.get("source_text", "") or ""),
                "item_count": len(snapshot.get("items", []) or []),
                "core_item_count": len(snapshot.get("core_items", []) or []),
                "asset_refs": self._clean_asset_refs(snapshot.get("asset_refs", []) or [])[:8],
            },
            "snapshot_preview": self._snapshot_preview(snapshot),
            "snapshot": snapshot,
        }

    def _audit_row_preview(self, row: dict) -> dict:
        return {
            "memory_id": str(row.get("memory_id", "") or ""),
            "tick_index": int(row.get("tick_index", -1) or -1),
            "source_text": str(row.get("source_text", "") or ""),
            "score": _round4(float(row.get("score", 0.0) or 0.0)),
            "candidate_sources": list(row.get("candidate_sources", []) or []),
            "vector_score": _round4(float(row.get("vector_score", 0.0) or 0.0)),
            "learned_vector_score": _round4(float(row.get("learned_vector_score", 0.0) or 0.0)),
            "learned_score": _round4(float(row.get("learned_score", 0.0) or 0.0)),
            "score_breakdown": dict(row.get("score_breakdown", {}) or {}),
            "snapshot_preview": dict(row.get("snapshot_preview", {}) or {}),
        }

    def _vector_candidates(self, memory_kind: str, query_tokens: list[str], *, query_vector: list[float] | None, posting_rows: list[dict]) -> list[dict]:
        """
        Vector candidate recall without any full scans.

        Policy (theory + performance constraints):
        - If FAISS/HNSW is enabled, use ANN search directly (bounded top_k).
        - If FAISS is not available, do NOT fall back to scanning all vectors.
          Instead, rerank only within posting candidates (bounded by candidate_limit).

        This preserves the APV2 database hard constraint: no operation should require
        a full traversal over all memory vectors.
        """

        base_vector = list(query_vector) if isinstance(query_vector, list) else None
        if base_vector is None:
            base_vector = self._embedder.embed(list(query_tokens or []))
        limit = max(1, int(self.candidate_limit))

        ann = self._ann_for_kind(str(memory_kind or ""))
        if ann is not None and ann.enabled():
            rows = ann.search(base_vector, top_k=limit)
            for row in rows:
                # Make sources stable for explainability.
                sources = list(row.get("candidate_sources", []) or [])
                if "faiss_hnsw_ip" not in sources:
                    sources.append("faiss_hnsw_ip")
                row["candidate_sources"] = sources
            # Filter tombstoned ids (ANN cannot remove online for HNSW in our wheel).
            filtered = []
            for row in rows:
                memory_id = str(row.get("memory_id", "") or "")
                if not memory_id or memory_id in self._ann_tombstones_by_kind.get(str(memory_kind or ""), set()):
                    continue
                snapshot = self._snapshot_by_id.get(memory_id)
                if snapshot and str(snapshot.get("memory_kind", "") or "") != str(memory_kind or ""):
                    continue
                filtered.append(row)
            return filtered

        # Fallback: rerank on posting candidates only (bounded, no scan).
        candidate_ids = [str(row.get("memory_id", "") or "") for row in posting_rows if str(row.get("memory_id", "") or "")]
        if not candidate_ids:
            return []
        rows: list[dict] = []
        for memory_id in candidate_ids[:limit]:
            vec = self._vector_cache.get(memory_id)
            if not vec:
                continue
            score = sum(a * b for a, b in zip(base_vector, vec))
            if score <= 0.0:
                continue
            rows.append({"memory_id": memory_id, "vector_score": _round4(score), "candidate_sources": ["vector_rerank_posting_only"]})
        rows.sort(key=lambda item: (-float(item.get("vector_score", 0.0) or 0.0), str(item.get("memory_id", "") or "")))
        return rows[:limit]

    def _learned_vector_candidates(self, memory_kind: str, query_tokens: list[str], *, posting_rows: list[dict]) -> list[dict]:
        """
        Candidate recall over the online learned-vector ANN (no full scans).

        This is the cross-namespace semantic bridge at the candidate layer: the
        query's tokens pool into a learned coordinate; its ANN neighbors are
        skill snapshots whose learned coordinate is close even when surface
        tokens differ (e.g. a math question -> math-skill snapshots). Without
        this channel, those snapshots never enter the scoring loop because
        posting (literal-token) recall misses them.

        Falls back (no FAISS) to reranking learned vectors of posting candidates
        only -- bounded, never a full scan.
        """
        if not self.online_enabled:
            return []
        query_learned_vector = self._online.learned_vector(
            list(query_tokens or []), limit=self.online_scoring_token_limit
        )
        if not query_learned_vector or not any(query_learned_vector):
            return []
        limit = max(1, int(self.candidate_limit))
        ann = self._ann_for_learned_kind(str(memory_kind or ""))
        if ann is not None and ann.enabled():
            rows = ann.search(query_learned_vector, top_k=limit)
            filtered = []
            tombstones = self._ann_learned_tombstones_by_kind.get(str(memory_kind or ""), set())
            for row in rows:
                memory_id = str(row.get("memory_id", "") or "")
                if not memory_id or memory_id in tombstones:
                    continue
                snapshot = self._snapshot_by_id.get(memory_id)
                if snapshot and str(snapshot.get("memory_kind", "") or "") != str(memory_kind or ""):
                    continue
                sources = list(row.get("candidate_sources", []) or [])
                if "learned_vector_ann" not in sources:
                    sources.append("learned_vector_ann")
                # FAISS labels its IP score "vector_score"; here it is the
                # learned-vector similarity. Relabel so it does not pollute the
                # hash vector_score channel and is captured by the merge ordering.
                learned_sim = float(row.get("vector_score", 0.0) or 0.0)
                filtered.append({
                    "memory_id": memory_id,
                    "learned_vector_score": _round4(learned_sim),
                    "candidate_sources": sources,
                })
            return filtered
        # Fallback: rerank learned vectors of posting candidates only.
        candidate_ids = [str(row.get("memory_id", "") or "") for row in posting_rows if str(row.get("memory_id", "") or "")]
        if not candidate_ids:
            return []
        rows = []
        for memory_id in candidate_ids[:limit]:
            snapshot_learned = self._snapshot_learned_vector_by_id.get(memory_id)
            if not snapshot_learned:
                continue
            score = sum(a * b for a, b in zip(query_learned_vector, snapshot_learned))
            if score <= 0.0:
                continue
            rows.append({
                "memory_id": memory_id,
                "learned_vector_score": _round4(score),
                "candidate_sources": ["learned_vector_rerank_posting_only"],
            })
        rows.sort(key=lambda item: (-float(item.get("learned_vector_score", 0.0) or 0.0), str(item.get("memory_id", "") or "")))
        return rows[:limit]

    def _touch_memory_revision(self) -> None:
        self._memory_revision += 1

    def _invalidate_successor_cache(self, memory_kind: str, memory_id: str) -> None:
        kind = str(memory_kind or "")
        clean = str(memory_id or "")
        if not kind or not clean or not self._successor_cache:
            return
        stale = [key for key in self._successor_cache if len(key) >= 3 and str(key[1]) == kind and str(key[2]) == clean]
        for key in stale:
            self._successor_cache.pop(key, None)

    def _queue_index_job(self, *, snapshot: dict, features: dict, vector: list[float], previous: dict | None = None, learned_vector: list[float] | None = None) -> None:
        memory_id = str(snapshot.get("memory_id", "") or "")
        if not memory_id:
            return
        self._pending_index_jobs[memory_id] = {
            "memory_kind": str(snapshot.get("memory_kind", "") or ""),
            "features": features,
            "vector": vector,
            "learned_vector": list(learned_vector or []),
            "numeric_features": self._snapshot_numeric_by_id.get(memory_id, {}),
            "relation_features": self._snapshot_relations_by_id.get(memory_id, {}),
            "previous_memory_id": str((previous or {}).get("memory_id", "") or ""),
            "heavy": str(snapshot.get("memory_kind", "") or "") == "state" and len(snapshot.get("items", []) or []) > 256,
        }
        self._update_pending_index_stats()

    def _update_pending_index_stats(self) -> None:
        total = len(self._pending_index_jobs)
        heavy = sum(1 for job in self._pending_index_jobs.values() if bool(job.get("heavy", False)))
        self._cache_stats["index_job_pending"] = total
        self._cache_stats["index_job_pending_heavy"] = heavy

    def _process_index_job_by_id(self, memory_id: str) -> dict:
        clean = str(memory_id or "")
        if not clean:
            return {"processed": 0, "skipped": 0}
        job = self._pending_index_jobs.pop(clean, None)
        if job is None:
            return {"processed": 0, "skipped": 0}
        snapshot = self._snapshot_by_id.get(clean)
        if snapshot is None:
            return {"processed": 0, "skipped": 1}
        self._index_snapshot_job(snapshot=snapshot, job=job)
        return {"processed": 1, "skipped": 0}

    def _index_snapshot_job(self, *, snapshot: dict, job: dict) -> None:
        memory_id = str(snapshot.get("memory_id", "") or "")
        if not memory_id or memory_id in self._indexed_snapshot_ids:
            return
        features = job.get("features", {}) or self._build_snapshot_features(snapshot)
        self._posting.add(
            snapshot["memory_kind"],
            memory_id,
            label_tokens=list(features["labels"])[: self.posting_label_token_limit],
            display_tokens=list(features["displays"])[: self.posting_display_token_limit],
            bigram_tokens=list(features["bigrams"])[: self.posting_bigram_token_limit],
            focus_tokens=list(features["focus_labels"])[:64],
            sequence_tokens=list(features["sequence_bigrams"])[: self.posting_sequence_token_limit],
        )
        vector = self._embedder.add_vector(memory_id, job.get("vector", []) or [])
        self._vector_cache[memory_id] = vector
        ann = self._ann_for_kind(snapshot["memory_kind"])
        if ann is not None and ann.enabled():
            ann.add(memory_id, vector)
        # Parallel learned-vector ANN: index the snapshot's online learned vector
        # so it can be recalled as a candidate by learned-vector neighborhood.
        learned_vector = job.get("learned_vector", None)
        if learned_vector is None:
            learned_vector = self._snapshot_learned_vector_by_id.get(memory_id)
        if learned_vector and any(learned_vector):
            learned_ann = self._ann_for_learned_kind(snapshot["memory_kind"])
            if learned_ann is not None and learned_ann.enabled():
                learned_ann.add(memory_id, list(learned_vector))
        if self.numeric_enabled:
            numeric_features = job.get("numeric_features", None)
            if not isinstance(numeric_features, dict):
                numeric_features = self._snapshot_numeric_by_id.get(memory_id, {})
            if numeric_features:
                self._numeric.add(snapshot["memory_kind"], memory_id, numeric_features)
        relation_features = job.get("relation_features", None)
        if not isinstance(relation_features, dict):
            relation_features = self._snapshot_relations_by_id.get(memory_id, {})
        if relation_features:
            self._relations.add_snapshot(
                memory_kind=str(snapshot["memory_kind"]),
                memory_id=memory_id,
                relation_features=relation_features,
                tick_index=int(snapshot.get("tick_index", -1) or -1),
            )
        if self.online_enabled:
            previous_id = str(job.get("previous_memory_id", "") or "")
            previous = self._snapshot_by_id.get(previous_id) if previous_id else None
            self._learn_from_snapshot(snapshot, previous)
        self._indexed_snapshot_ids.add(memory_id)
        self._indexed_count_by_kind[str(snapshot.get("memory_kind", "") or "")] += 1

    def _ann_for_kind(self, memory_kind: str) -> FaissHnswIndex | None:
        if not self._ann_enabled:
            return None
        kind = str(memory_kind or "")
        if not kind:
            return None
        ann = self._ann_by_kind.get(kind)
        if ann is None:
            ann = FaissHnswIndex(config=self._ann_config)
            self._ann_by_kind[kind] = ann
        return ann

    def _ann_for_learned_kind(self, memory_kind: str) -> FaissHnswIndex | None:
        """Parallel ANN index over online learned vectors (same dim/config as
        the hash-vector ANN). Lazily created per kind, like `_ann_for_kind`.

        Note: this is gated only on `_ann_enabled`, NOT `online_enabled`.
        Indexing a snapshot's already-computed learned vector is not "learning"
        (no observe_* / no table mutation), so it must still run during
        warm-load's `_index_snapshot_job_without_learning` (which temporarily
        sets online_enabled=False). Otherwise the learned ANN stays empty."""
        if not self._ann_enabled:
            return None
        kind = str(memory_kind or "")
        if not kind:
            return None
        ann = self._ann_learned_by_kind.get(kind)
        if ann is None:
            ann = FaissHnswIndex(config=self._ann_config)
            self._ann_learned_by_kind[kind] = ann
        return ann

    def _bounded_ordered_dict(self, data: OrderedDict, limit: int) -> None:
        cap = max(1, int(limit))
        while len(data) > cap:
            data.popitem(last=False)

    def _build_query_signature(self, query_features: dict) -> str:
        # Keep it cheap and stable; avoid including volatile floats unless needed.
        # This is an APV2.1 adaptation of the legacy MemoryStoreV2 query_signature.
        import hashlib

        hasher = hashlib.blake2b(digest_size=16)
        for label in query_features.get("labels", []) or []:
            clean = str(label or "")
            if clean:
                hasher.update(f"L|{clean}\n".encode("utf-8"))
        for disp in query_features.get("displays", []) or []:
            clean = str(disp or "")
            if clean:
                hasher.update(f"D|{clean}\n".encode("utf-8"))
        for bigram in query_features.get("bigrams", []) or []:
            clean = str(bigram or "")
            if clean:
                hasher.update(f"B|{clean}\n".encode("utf-8"))
        for seq in query_features.get("sequence_bigrams", []) or []:
            clean = str(seq or "")
            if clean:
                hasher.update(f"S|{clean}\n".encode("utf-8"))
        for rel in query_features.get("relation_tokens", []) or []:
            clean = str(rel or "")
            if clean:
                hasher.update(f"R|{clean}\n".encode("utf-8"))
        for focus in query_features.get("focus_labels", []) or []:
            clean = str(focus or "")
            if clean:
                hasher.update(f"F|{clean}\n".encode("utf-8"))
        return hasher.hexdigest()

    def _build_candidate_signature(self, query_features: dict) -> str:
        import hashlib

        hasher = hashlib.blake2b(digest_size=16)
        for label in self._stable_unique_for_signature(query_features.get("candidate_labels", []) or [], limit=self.posting_label_token_limit):
            clean = str(label or "")
            if clean:
                hasher.update(f"L|{clean}\n".encode("utf-8"))
        for disp in self._stable_unique_for_signature(query_features.get("candidate_displays", []) or [], limit=self.posting_display_token_limit):
            clean = str(disp or "")
            if clean:
                hasher.update(f"D|{clean}\n".encode("utf-8"))
        for bigram in self._stable_unique_for_signature(query_features.get("candidate_bigrams", []) or [], limit=self.posting_bigram_token_limit):
            clean = str(bigram or "")
            if clean:
                hasher.update(f"B|{clean}\n".encode("utf-8"))
        for seq in self._stable_unique_for_signature(query_features.get("candidate_sequence_bigrams", []) or [], limit=self.posting_sequence_token_limit):
            clean = str(seq or "")
            if clean:
                hasher.update(f"S|{clean}\n".encode("utf-8"))
        for focus in self._stable_unique_for_signature(query_features.get("candidate_focus_labels", []) or [], limit=64):
            clean = str(focus or "")
            if clean:
                hasher.update(f"F|{clean}\n".encode("utf-8"))
        return hasher.hexdigest()

    def _stable_unique_for_signature(self, tokens: list[str], *, limit: int) -> list[str]:
        seen = set()
        rows = []
        for token in tokens or []:
            clean = str(token or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(clean)
        rows.sort()
        return rows[: max(1, int(limit))]

    def _get_or_build_query_vector(self, query_signature: str, query_tokens: list[str]) -> list[float]:
        cached = self._query_vector_cache.get(query_signature)
        if cached is not None:
            self._cache_stats["query_vector_hit"] += 1
            self._query_vector_cache.move_to_end(query_signature)
            return list(cached)
        self._cache_stats["query_vector_miss"] += 1
        vector = self._embedder.embed(list(query_tokens or []))
        self._query_vector_cache[query_signature] = list(vector)
        self._bounded_ordered_dict(self._query_vector_cache, self._query_vector_cache_limit)
        return list(vector)

    def _get_or_build_candidates(self, *, memory_kind: str, query_signature: str, query_features: dict) -> tuple[list[dict], list[dict], list[dict]]:
        epoch = int(self._memory_revision) // max(1, int(self._candidate_cache_revision_stride))
        cache_key = (epoch, str(memory_kind or ""), str(query_signature or ""))
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            self._cache_stats["candidate_hit"] += 1
            self._candidate_cache.move_to_end(cache_key)
            posting_rows = self._with_recent_direct_candidates(str(memory_kind), list(cached.get("posting_rows", []) or []))
            vector_rows = list(cached.get("vector_rows", []) or [])
            numeric_rows = self._refresh_numeric_candidate_rows(
                str(memory_kind),
                query_features=query_features,
                posting_rows=posting_rows,
                vector_rows=vector_rows,
                cached_numeric_rows=list(cached.get("numeric_rows", []) or []),
            )
            return self._maybe_extend_with_long_term_candidates(
                memory_kind=str(memory_kind),
                query_features=query_features,
                posting_rows=posting_rows,
                vector_rows=vector_rows,
                numeric_rows=numeric_rows,
            )
        self._cache_stats["candidate_miss"] += 1
        candidate_tokens = self._candidate_vector_tokens_for_index(query_features)
        if not self._has_indexed_snapshots(str(memory_kind)):
            posting_rows = []
            vector_rows = []
            numeric_rows = []
        else:
            posting_rows = self._posting.candidates(
                str(memory_kind),
                label_tokens=list(query_features.get("candidate_labels", []) or [])[: self.posting_label_token_limit],
                display_tokens=list(query_features.get("candidate_displays", []) or [])[: self.posting_display_token_limit],
                bigram_tokens=list(query_features.get("candidate_bigrams", []) or [])[: self.posting_bigram_token_limit],
                focus_tokens=list(query_features.get("candidate_focus_labels", []) or []),
                sequence_tokens=list(query_features.get("candidate_sequence_bigrams", []) or [])[: self.posting_sequence_token_limit],
                limit=self.candidate_limit,
            )
            query_vector = self._get_or_build_query_vector(str(query_signature), candidate_tokens)
            vector_rows = self._vector_candidates(
                str(memory_kind),
                candidate_tokens,
                query_vector=query_vector,
                posting_rows=posting_rows,
            )
            # Learned-vector ANN candidates flow through the same vector_rows
            # channel (so cache + _merge_candidates handle them unchanged). Append
            # only ids not already surfaced by hash-vector recall; the scoring
            # loop reads each snapshot's learned vector regardless of source.
            learned_vector_rows = self._learned_vector_candidates(
                str(memory_kind),
                candidate_tokens,
                posting_rows=posting_rows,
            )
            if learned_vector_rows:
                seen_vector_ids = {str(r.get("memory_id", "") or "") for r in vector_rows}
                for row in learned_vector_rows:
                    if str(row.get("memory_id", "") or "") not in seen_vector_ids:
                        vector_rows.append(row)
            numeric_rows = self._numeric_candidates(str(memory_kind), query_features=query_features, posting_rows=posting_rows, vector_rows=vector_rows)
        payload = {"posting_rows": list(posting_rows), "vector_rows": list(vector_rows), "numeric_rows": list(numeric_rows)}
        self._candidate_cache[cache_key] = payload
        self._bounded_ordered_dict(self._candidate_cache, self._candidate_cache_limit)
        posting_rows = self._with_recent_direct_candidates(str(memory_kind), list(posting_rows))
        numeric_rows = self._refresh_numeric_candidate_rows(
            str(memory_kind),
            query_features=query_features,
            posting_rows=posting_rows,
            vector_rows=list(vector_rows),
            cached_numeric_rows=list(numeric_rows),
        )
        return self._maybe_extend_with_long_term_candidates(
            memory_kind=str(memory_kind),
            query_features=query_features,
            posting_rows=posting_rows,
            vector_rows=list(vector_rows),
            numeric_rows=numeric_rows,
        )

    def _maybe_extend_with_long_term_candidates(
        self,
        *,
        memory_kind: str,
        query_features: dict,
        posting_rows: list[dict],
        vector_rows: list[dict],
        numeric_rows: list[dict],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Let old durable memories participate without full-loading history.

        The runtime first trusts the hot working set. Only when hot candidates
        lack enough evidence do we jump through persistence-side posting indexes,
        rehydrate the matched snapshots, and let the normal B/C scorer decide.
        """

        if (
            not self.long_term_recall_enabled
            or str(memory_kind or "") not in self.long_term_recall_kinds
            or not self._long_term_recall_available()
        ):
            return posting_rows, vector_rows, numeric_rows
        if self._hot_candidates_confident(posting_rows, vector_rows, numeric_rows):
            self._cache_stats["long_term_pruned_hot_confident"] += 1
            return posting_rows, vector_rows, numeric_rows
        tokens_by_field = self._long_term_tokens_by_field(query_features)
        if not any(tokens_by_field.values()):
            self._cache_stats["long_term_posting_no_candidate"] += 1
            return posting_rows, vector_rows, numeric_rows
        finder = getattr(self._persistence, "exact_posting_candidates", None)
        if not callable(finder):
            return posting_rows, vector_rows, numeric_rows
        self._cache_stats["long_term_posting_query"] += 1
        try:
            cold_rows = finder(
                memory_kind=str(memory_kind or ""),
                tokens_by_field=tokens_by_field,
                limit=self.long_term_posting_limit,
            )
        except Exception as exc:
            self._cache_stats["long_term_posting_no_candidate"] += 1
            self._last_persistence_error = str(exc)
            self._persistence_error_count += 1
            return posting_rows, vector_rows, numeric_rows
        cold_rows = [dict(row) for row in list(cold_rows or []) if isinstance(row, dict)]
        if not cold_rows:
            self._cache_stats["long_term_posting_no_candidate"] += 1
            return posting_rows, vector_rows, numeric_rows
        self._cache_stats["long_term_posting_candidate"] += len(cold_rows)
        rehydrated_ids = self._rehydrate_persistent_candidates(
            memory_kind=str(memory_kind or ""),
            candidate_rows=cold_rows,
            limit=self.long_term_rehydrate_limit,
        )
        if not rehydrated_ids:
            return posting_rows, vector_rows, numeric_rows
        existing = {str(row.get("memory_id", "") or "") for row in posting_rows if str(row.get("memory_id", "") or "")}
        merged_posting = list(posting_rows)
        for row in cold_rows:
            memory_id = str(row.get("memory_id", "") or "")
            if not memory_id or memory_id in existing or memory_id not in rehydrated_ids:
                continue
            existing.add(memory_id)
            clean = dict(row)
            sources = list(clean.get("candidate_sources", []) or [])
            if "long_term_rehydrated_posting" not in sources:
                sources.append("long_term_rehydrated_posting")
            clean["candidate_sources"] = sources
            merged_posting.append(clean)
        numeric_rows = self._refresh_numeric_candidate_rows(
            str(memory_kind or ""),
            query_features=query_features,
            posting_rows=merged_posting,
            vector_rows=vector_rows,
            cached_numeric_rows=numeric_rows,
        )
        return merged_posting, vector_rows, numeric_rows

    def _long_term_recall_available(self) -> bool:
        return callable(getattr(self._persistence, "exact_posting_candidates", None)) and callable(getattr(self._persistence, "snapshot_by_id", None))

    def _hot_candidates_confident(self, posting_rows: list[dict], vector_rows: list[dict], numeric_rows: list[dict]) -> bool:
        merged = self._merge_candidates(list(posting_rows or []), list(vector_rows or []), list(numeric_rows or []))
        confident = 0
        for row in merged:
            sources = {str(source or "") for source in list(row.get("candidate_sources", []) or [])}
            if sources and sources.issubset({"recent_direct", "recent_direct_cached_result"}):
                continue
            score = (
                float(row.get("posting_score", 0.0) or 0.0)
                + float(row.get("vector_score", 0.0) or 0.0) * 0.75
                + float(row.get("numeric_score", 0.0) or 0.0) * 0.6
            )
            if score >= self.long_term_hot_confidence_threshold:
                confident += 1
                if confident >= self.long_term_hot_confident_count:
                    return True
        return False

    def _long_term_tokens_by_field(self, query_features: dict) -> dict[str, list[str]]:
        def _pick(values: object, *, limit: int) -> list[str]:
            return self._stable_unique_for_signature(list(values or []), limit=max(1, int(limit)))

        return {
            "sequence": _pick(query_features.get("candidate_sequence_bigrams", []), limit=min(self.posting_sequence_token_limit, 64)),
            "bigram": _pick(query_features.get("candidate_bigrams", []), limit=min(self.posting_bigram_token_limit, 64)),
            "label": _pick(query_features.get("candidate_labels", []), limit=min(self.posting_label_token_limit, 96)),
            "focus": _pick(query_features.get("candidate_focus_labels", []), limit=32),
            "display": _pick(query_features.get("candidate_displays", []), limit=min(self.posting_display_token_limit, 48)),
            "relation": _pick(query_features.get("relation_tokens", []), limit=min(self.relation_token_limit, 64)),
        }

    def _rehydrate_persistent_candidates(self, *, memory_kind: str, candidate_rows: list[dict], limit: int) -> set[str]:
        loaded: set[str] = set()
        skipped = 0
        for row in list(candidate_rows or [])[: max(1, int(limit))]:
            memory_id = str((row or {}).get("memory_id", "") or "")
            if not memory_id:
                skipped += 1
                continue
            if memory_id in self._snapshot_by_id:
                loaded.add(memory_id)
                continue
            if self._rehydrate_persistent_snapshot_by_id(memory_id, expected_kind=memory_kind):
                loaded.add(memory_id)
            else:
                skipped += 1
        self._cache_stats["long_term_rehydrate_skip"] += skipped
        if loaded:
            self._rehydrate_persistent_successors(memory_kind=str(memory_kind or ""), source_memory_ids=sorted(loaded))
        return loaded

    def _rehydrate_persistent_snapshot_by_id(self, memory_id: str, *, expected_kind: str | None = None) -> bool:
        clean = str(memory_id or "")
        if not clean or clean in self._snapshot_by_id:
            return bool(clean and clean in self._snapshot_by_id)
        loader = getattr(self._persistence, "snapshot_by_id", None)
        if not callable(loader):
            return False
        try:
            row = loader(clean)
        except Exception as exc:
            self._last_persistence_error = str(exc)
            self._persistence_error_count += 1
            return False
        if not isinstance(row, dict) or not row:
            return False
        snapshot = self._normalize_persisted_snapshot(row)
        if expected_kind is not None and str(snapshot.get("memory_kind", "") or "") != str(expected_kind or ""):
            return False
        features = self._build_snapshot_features(snapshot)
        vector = self._embedder.embed(self._vector_tokens_for_index(features))
        state_field_items = self._snapshot_state_field_items(snapshot)
        energy = self._energy_profile(state_field_items, limit=self.core_item_limit)
        energy_mass = self._energy_mass(energy)
        numeric = self._numeric_feature_profile(state_field_items, limit=self.core_item_limit)
        relations = dict(snapshot.get("relation_features", {}) or self._relations.build_features(
            memory_kind=str(snapshot.get("memory_kind", "") or ""),
            items=state_field_items,
            focus_labels=snapshot.get("focus_labels", []),
        ))
        snapshot["numeric_features"] = {channel: list(values) for channel, values in numeric.items()}
        snapshot["relation_features"] = relations
        snapshot["prediction_payload_items"] = self._build_prediction_payload_items(snapshot)
        snapshot["action_feedback_items"] = self._extract_action_feedback_items(snapshot.get("items", []), limit=24)
        self._snapshot_by_id[clean] = snapshot
        self._snapshot_features_by_id[clean] = features
        self._snapshot_energy_by_id[clean] = energy
        self._snapshot_energy_mass_by_id[clean] = energy_mass
        self._snapshot_numeric_by_id[clean] = numeric
        self._snapshot_relations_by_id[clean] = relations
        self._register_label_document_frequencies(snapshot)
        self._transitions.register_snapshot(snapshot)
        learned_vector = list((snapshot.get("vector_spaces", {}) or {}).get("online_learned_vector", []) or [])
        if not learned_vector and self.online_enabled:
            learned_vector = self._online.learned_vector(
                self._vector_tokens_for_index(features),
                limit=self.online_scoring_token_limit,
            )
        snapshot.setdefault("vector_spaces", {"hash_vector": list(vector), "online_learned_vector": list(learned_vector)})
        self._snapshot_learned_vector_by_id[clean] = list(learned_vector)
        self._queue_index_job(snapshot=snapshot, features=features, vector=vector, learned_vector=learned_vector, previous=None)
        job = self._pending_index_jobs.get(clean)
        if job is not None:
            self._index_snapshot_job_without_learning(snapshot=snapshot, job=job)
            self._pending_index_jobs.pop(clean, None)
            self._update_pending_index_stats()
        self._long_term_rehydrated_ids[clean] = str(snapshot.get("memory_kind", "") or "")
        self._long_term_rehydrated_ids.move_to_end(clean)
        self._evict_excess_rehydrated_snapshots()
        self._cache_stats["long_term_rehydrated"] += 1
        self._touch_memory_revision()
        return True

    def _evict_excess_rehydrated_snapshots(self) -> None:
        while len(self._long_term_rehydrated_ids) > self.long_term_rehydrated_resident_limit:
            memory_id, memory_kind = self._long_term_rehydrated_ids.popitem(last=False)
            if any(str((snapshot or {}).get("memory_id", "") or "") == memory_id for snapshot in self._recent_by_kind.get(str(memory_kind or ""), [])):
                continue
            snapshot = self._snapshot_by_id.get(memory_id)
            if snapshot is not None:
                self._evict_snapshot(snapshot)

    def _rehydrate_persistent_successors(self, *, memory_kind: str, source_memory_ids: list[str]) -> None:
        edge_loader = getattr(self._persistence, "successor_edges", None)
        if not callable(edge_loader):
            return
        clean_sources = [str(mid or "") for mid in list(source_memory_ids or []) if str(mid or "")]
        if not clean_sources:
            return
        edges = []
        for edge_kind in [str(memory_kind or ""), self.EPISODE_SUCCESSOR_KIND]:
            if not edge_kind:
                continue
            try:
                edges.extend(edge_loader(memory_kind=edge_kind, source_memory_ids=clean_sources, limit_per_source=8) or [])
            except Exception as exc:
                self._last_persistence_error = str(exc)
                self._persistence_error_count += 1
                return
        for edge in list(edges or []):
            if not isinstance(edge, dict):
                continue
            source_id = str(edge.get("source_memory_id", "") or "")
            successor_id = str(edge.get("successor_memory_id", "") or "")
            edge_kind = str(edge.get("memory_kind", memory_kind) or memory_kind)
            if not source_id or not successor_id:
                continue
            if source_id not in self._snapshot_by_id:
                self._rehydrate_persistent_snapshot_by_id(source_id, expected_kind=memory_kind)
            expected_successor_kind = None if edge_kind == self.EPISODE_SUCCESSOR_KIND else memory_kind
            if successor_id not in self._snapshot_by_id and self._rehydrate_persistent_snapshot_by_id(successor_id, expected_kind=expected_successor_kind):
                self._cache_stats["long_term_successor_rehydrated"] += 1
            if source_id in self._snapshot_by_id and successor_id in self._snapshot_by_id:
                self._transitions.register_snapshot(self._snapshot_by_id[source_id])
                self._transitions.register_snapshot(self._snapshot_by_id[successor_id])
                self._transitions.link_successor(edge_kind, source_id, successor_id)
                self._cache_stats["long_term_successor_edge_loaded"] += 1

    def _numeric_candidates(self, memory_kind: str, *, query_features: dict, posting_rows: list[dict], vector_rows: list[dict]) -> list[dict]:
        if not (self.numeric_enabled and self.numeric_candidate_limit > 0):
            return []
        query_numeric = dict(query_features.get("numeric_features", {}) or {})
        if not query_numeric:
            return []
        rows = self._numeric.search(
            str(memory_kind or ""),
            query_numeric,
            top_k_per_channel=self.numeric_top_k_per_channel,
            overall_limit=self.numeric_candidate_limit,
        )
        candidate_ids = [str(row.get("memory_id", "") or "") for row in list(posting_rows or []) + list(vector_rows or []) if str(row.get("memory_id", "") or "")]
        reranked = self._numeric.rerank_candidates(str(memory_kind or ""), query_numeric, candidate_ids)
        merged: dict[str, dict] = {str(row.get("memory_id", "") or ""): dict(row) for row in rows if str(row.get("memory_id", "") or "")}
        for memory_id, row in reranked.items():
            bucket = merged.setdefault(memory_id, {"memory_id": memory_id, "numeric_score": 0.0, "numeric_score_breakdown": {}, "candidate_sources": []})
            bucket["numeric_score"] = max(float(bucket.get("numeric_score", 0.0) or 0.0), float(row.get("numeric_score", 0.0) or 0.0))
            breakdown = dict(bucket.get("numeric_score_breakdown", {}) or {})
            for channel, score in dict(row.get("numeric_score_breakdown", {}) or {}).items():
                breakdown[str(channel)] = max(float(breakdown.get(str(channel), 0.0) or 0.0), float(score or 0.0))
            bucket["numeric_score_breakdown"] = breakdown
            for source in row.get("candidate_sources", []) or []:
                if source not in bucket["candidate_sources"]:
                    bucket["candidate_sources"].append(source)
        ordered = list(merged.values())
        ordered.sort(key=lambda item: (-float(item.get("numeric_score", 0.0) or 0.0), str(item.get("memory_id", "") or "")))
        return ordered[: self.numeric_candidate_limit]

    def _refresh_numeric_candidate_rows(
        self,
        memory_kind: str,
        *,
        query_features: dict,
        posting_rows: list[dict],
        vector_rows: list[dict],
        cached_numeric_rows: list[dict],
    ) -> list[dict]:
        if not (self.numeric_enabled and self.numeric_candidate_limit > 0):
            return []
        query_numeric = dict(query_features.get("numeric_features", {}) or {})
        if not query_numeric:
            return []
        candidate_ids = [str(row.get("memory_id", "") or "") for row in list(posting_rows or []) + list(vector_rows or []) + list(cached_numeric_rows or []) if str(row.get("memory_id", "") or "")]
        reranked = self._numeric.rerank_candidates(str(memory_kind or ""), query_numeric, candidate_ids)
        merged: dict[str, dict] = {str(row.get("memory_id", "") or ""): dict(row) for row in cached_numeric_rows or [] if str(row.get("memory_id", "") or "")}
        for memory_id, row in reranked.items():
            bucket = merged.setdefault(memory_id, {"memory_id": memory_id, "numeric_score": 0.0, "numeric_score_breakdown": {}, "candidate_sources": []})
            bucket["numeric_score"] = max(float(bucket.get("numeric_score", 0.0) or 0.0), float(row.get("numeric_score", 0.0) or 0.0))
            breakdown = dict(bucket.get("numeric_score_breakdown", {}) or {})
            for channel, score in dict(row.get("numeric_score_breakdown", {}) or {}).items():
                breakdown[str(channel)] = max(float(breakdown.get(str(channel), 0.0) or 0.0), float(score or 0.0))
            bucket["numeric_score_breakdown"] = breakdown
            for source in row.get("candidate_sources", []) or []:
                if source not in bucket["candidate_sources"]:
                    bucket["candidate_sources"].append(source)
        ordered = list(merged.values())
        ordered.sort(key=lambda item: (-float(item.get("numeric_score", 0.0) or 0.0), str(item.get("memory_id", "") or "")))
        return ordered[: self.numeric_candidate_limit]

    def _has_indexed_snapshots(self, memory_kind: str) -> bool:
        kind = str(memory_kind or "")
        if not kind:
            return False
        return int(self._indexed_count_by_kind.get(kind, 0) or 0) > 0

    def _with_recent_direct_candidates(self, memory_kind: str, rows: list[dict]) -> list[dict]:
        seen = {str(row.get("memory_id", "") or "") for row in rows if str(row.get("memory_id", "") or "")}
        augmented = list(rows)
        bucket = self._recent_by_kind.get(str(memory_kind or ""), [])
        for snapshot in reversed(bucket[-self._recent_direct_candidate_limit :]):
            memory_id = str((snapshot or {}).get("memory_id", "") or "")
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            augmented.append(
                {
                    "memory_id": memory_id,
                    "posting_score": 0.0,
                    "total_matches": 0,
                    "match_counts": {},
                    "matched_tokens": {},
                    "candidate_sources": ["recent_direct"],
                }
            )
        return augmented[: self.candidate_limit]

    def _vector_tokens_for_index(self, features: dict) -> list[str]:
        tokens: list[str] = []
        budgets = (
            ("labels", max(16, self.vector_token_limit // 2)),
            ("focus_labels", 64),
            ("bigrams", max(16, self.vector_token_limit // 4)),
            ("sequence_bigrams", max(16, self.vector_token_limit // 4)),
            ("relation_tokens", max(16, self.vector_token_limit // 4)),
            ("displays", max(16, self.vector_token_limit // 4)),
        )
        seen = set()
        for key, cap in budgets:
            for token in list(features.get(key, []) or [])[:cap]:
                clean = str(token or "").strip()
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                tokens.append(clean)
                if len(tokens) >= self.vector_token_limit:
                    return tokens
        return tokens[: self.vector_token_limit]

    def _candidate_vector_tokens_for_index(self, features: dict) -> list[str]:
        candidate = dict(features or {})
        if "candidate_vector_tokens" in candidate:
            return list(candidate.get("candidate_vector_tokens", []) or [])[: self.vector_token_limit]
        return self._vector_tokens_for_index(candidate)

    def _items_signature(self, items: list[dict], *, focus_labels: list[str], tick_index: int, memory_kind: str = "") -> str:
        import hashlib

        hasher = hashlib.blake2b(digest_size=16)
        kind = str(memory_kind or "")
        hasher.update(f"K|{kind}\n".encode("utf-8"))
        for label in focus_labels or []:
            clean = str(label or "")
            if clean:
                hasher.update(f"F|{clean}\n".encode("utf-8"))
        for item in items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            hasher.update(f"L|{label}|".encode("utf-8"))
            hasher.update(str(item.get("family", "") or "").encode("utf-8"))
            hasher.update(b"|")
            hasher.update(str(item.get("source_type", "") or "").encode("utf-8"))
            hasher.update(b"|")
            last_seen = int(item.get("last_seen_tick", -1) or -1)
            hasher.update(b"1" if last_seen == int(tick_index) else b"0")
            hasher.update(b"|")
            hasher.update(str(item.get("position", "") or "").encode("utf-8"))
            hasher.update(b"|")
            numeric = self._extract_numeric_features(item)
            if numeric:
                for channel, vector in sorted(numeric.items()):
                    hasher.update(f"N|{channel}|".encode("utf-8"))
                    for value in vector[:8]:
                        hasher.update(str(round(float(value or 0.0), 3)).encode("utf-8"))
                        hasher.update(b",")
                    hasher.update(b"|")
            if kind == "focus":
                hasher.update(b"1" if bool(item.get("is_focus", False)) else b"0")
            for fragment in self._process_anchor_signature_fragments(item):
                hasher.update(f"P|{fragment}|".encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _process_anchor_signature_fragments(self, item: dict) -> list[str]:
        """
        Keep snapshot payload-cache keys sensitive to low-grain process anchors.

        Cn successor payloads must remember "current unread glyph / cursor
        aligned action / post-action feedback" metadata. The ordinary signature
        intentionally stays compact, but if two snapshots share the same labels
        while their process-anchor roles differ, reusing the earlier payload can
        erase digit/colon boundary memories. These fragments are metadata about
        how a token was learned, not hidden OCR output or an answer table.
        """

        if not isinstance(item, dict):
            return []
        meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
        if not meta:
            return []
        label = str(item.get("sa_label", "") or "")
        family = str(item.get("family", "") or "")
        source_type = str(item.get("source_type", "") or "")
        schema_id = str(meta.get("schema_id", "") or "")
        priority = str(meta.get("prediction_payload_priority", "") or "")
        current_role = str(meta.get("current_glyph_role", "") or "")
        same_tick_role = str(meta.get("same_tick_binding_role", "") or "")
        process_role = str(meta.get("process_anchor_role", "") or "")
        readout_role = str(meta.get("readout_semantic_role", "") or "")
        semantic_frame_role = str(meta.get("semantic_frame_role", "") or "")
        current_read = bool(meta.get("current_read_tick", False))
        is_process_anchor = (
            label.startswith(("text::", "text_action::", "text_revision_opportunity::", "text_slot_confirmation::", "action_feedback::"))
            or family in {"text", "text_action", "text_revision_opportunity", "text_slot_confirmation", "action_feedback"}
            or source_type in {"text_action", "action_feedback"}
            or schema_id in {
                "text_revision_opportunity/v1",
                "text_slot_confirmation/v1",
                "text_character_binding/v1",
                "text_action_feedback/v1",
                "text_insert_closure_state/v1",
                "text_reread_closure_state/v1",
                "ui_readout3_whole_region_time_readout_frame/v1",
            }
            or priority.startswith(("current_glyph", "previous_prefix"))
            or "glyph" in current_role
            or "glyph" in same_tick_role
            or bool(process_role)
            or bool(readout_role)
            or bool(semantic_frame_role)
            or current_read
        )
        if not is_process_anchor:
            return []
        fragments: list[str] = []
        allowed_keys = (
            "schema_id",
            "event_type",
            "current_glyph_index",
            "current_glyph_role",
            "same_tick_binding_role",
            "prediction_payload_priority",
            "process_anchor_role",
            "visible_length",
            "cursor_index",
            "cursor",
            "last_visible_token",
            "operation",
            "conflict_kind",
            "span",
            "support",
            "task_id",
            "paradigm_id",
            "region_id",
            "readout_semantic_role",
            "readout_pattern_id",
            "semantic_frame_role",
            "dynamic_slot_role",
            "slot_role",
            "previous_prefix",
            "current_read_tick",
            "feedback_outcome",
            "feedback_correctness",
        )
        for key in allowed_keys:
            if key not in meta:
                continue
            value = meta.get(key)
            if value is None or str(value) == "":
                continue
            if isinstance(value, float):
                clean_value = round(float(value), 3)
            elif isinstance(value, (list, tuple)):
                clean_value = ",".join(str(part) for part in list(value)[:8])
            elif isinstance(value, dict):
                continue
            else:
                clean_value = value
            fragments.append(f"{key}={clean_value}")
        token = self._text_payload_token(item)
        if token and (
            label.startswith(("text_revision_opportunity::", "text_action::", "text_slot_confirmation::"))
            or schema_id in {"text_revision_opportunity/v1", "text_slot_confirmation/v1"}
            or family in {"text_action", "text_revision_opportunity", "text_slot_confirmation"}
        ):
            fragments.append(f"process_token={token}")
        return fragments[:32]

    def _should_rebuild_ann(self, memory_kind: str) -> bool:
        ann = self._ann_by_kind.get(str(memory_kind or ""))
        if not (self._ann_enabled and ann is not None and ann.enabled()):
            return False
        total = max(1, ann.count())
        removed = int(self._ann_removed_since_rebuild_by_kind.get(str(memory_kind or ""), 0) or 0)
        if removed < self._ann_min_removed_before_rebuild:
            return False
        return (removed / float(total)) >= float(self._ann_rebuild_threshold_ratio)

    def _rebuild_ann_index(self, memory_kind: str) -> None:
        kind = str(memory_kind or "")
        ann = self._ann_by_kind.get(kind)
        if not (self._ann_enabled and ann is not None and ann.enabled()):
            return
        ann.rebuild()
        self._ann_tombstones_by_kind[kind].clear()
        self._ann_removed_since_rebuild_by_kind[kind] = 0

    def _evict_snapshot(self, removed: dict) -> None:
        memory_id = str((removed or {}).get("memory_id", "") or "")
        memory_kind = str((removed or {}).get("memory_kind", "") or "")
        if not memory_id:
            return
        # Remove primary snapshot payload.
        self._snapshot_by_id.pop(memory_id, None)
        self._unregister_label_document_frequencies(removed)
        # Remove posting references (bounded via reverse index).
        self._posting.remove(memory_kind, memory_id)
        self._pending_index_jobs.pop(memory_id, None)
        if memory_id in self._indexed_snapshot_ids:
            self._indexed_snapshot_ids.discard(memory_id)
            self._indexed_count_by_kind[memory_kind] = max(0, int(self._indexed_count_by_kind.get(memory_kind, 0) or 0) - 1)
        # Remove embedder vector copy.
        self._embedder.remove(memory_id)
        self._vector_cache.pop(memory_id, None)
        self._snapshot_features_by_id.pop(memory_id, None)
        self._snapshot_energy_by_id.pop(memory_id, None)
        self._snapshot_energy_mass_by_id.pop(memory_id, None)
        self._snapshot_numeric_by_id.pop(memory_id, None)
        self._snapshot_relations_by_id.pop(memory_id, None)
        self._numeric.remove(memory_kind, memory_id)
        self._relations.remove_snapshot(memory_kind=memory_kind, memory_id=memory_id)
        # Remove transition payload/edges best-effort.
        self._transitions.remove_snapshot(memory_kind, memory_id)
        # ANN cannot remove ids for HNSW in our wheel. Track tombstones and rebuild occasionally.
        ann = self._ann_by_kind.get(memory_kind)
        if self._ann_enabled and ann is not None and ann.enabled():
            removed_ok = ann.remove(memory_id)
            if removed_ok:
                self._ann_tombstones_by_kind[memory_kind].add(memory_id)
                self._ann_removed_since_rebuild_by_kind[memory_kind] += 1
                if self._should_rebuild_ann(memory_kind):
                    self._rebuild_ann_index(memory_kind)

    def _learn_from_snapshot(self, snapshot: dict, previous: dict | None) -> None:
        features = self._snapshot_features_by_id.get(str(snapshot.get("memory_id", "") or "")) or self._build_snapshot_features(snapshot)
        tokens = self._unique_tokens(features["vector_tokens"])
        focus_tokens = self._unique_tokens(features["focus_labels"])
        structured_events: list[dict] = []
        energy_events = self._build_energy_learning_events(snapshot)
        structured_events.extend(self._structured_energy_events(snapshot, energy_events))
        energy_learning_tokens = self._energy_learning_token_set(energy_events)
        for event in energy_events:
            source = str(event.get("source", "") or "")
            target = str(event.get("target", "") or "")
            if not source or not target:
                continue
            relation = str(event.get("relation", "positive") or "positive")
            weight = float(event.get("weight", 1.0) or 1.0)
            if relation == "negative":
                self._online.observe_negative_anchor(source, target, weight=weight)
            else:
                self._online.observe_positive_anchor(source, target, weight=weight)
        self._record_energy_learning_events(energy_events)
        relation_events = self._build_relation_learning_events(snapshot)
        structured_events.extend(self._structured_relation_events(snapshot, relation_events))
        for event in relation_events:
            source = str(event.get("source", "") or "")
            target = str(event.get("target", "") or "")
            if not source or not target:
                continue
            self._online.observe_transition_pair(source, target, weight=float(event.get("weight", 1.0) or 1.0))
        self._record_relation_learning_events(relation_events)
        events = self._build_multimodal_learning_events(snapshot, previous)
        structured_events.extend(self._structured_multimodal_events(snapshot, events))
        for event in events:
            source = str(event.get("source", "") or "")
            target = str(event.get("target", "") or "")
            if not source or not target:
                continue
            weight = float(event.get("weight", 1.0) or 1.0)
            if str(event.get("relation", "positive") or "positive") == "transition":
                self._online.observe_transition_pair(source, target, weight=weight)
            else:
                self._online.observe_positive_pair(source, target, weight=weight)
        self._record_multimodal_learning_events(events)
        for idx in range(0, len(focus_tokens)):
            for jdx in range(idx + 1, len(focus_tokens)):
                if focus_tokens[idx] in energy_learning_tokens or focus_tokens[jdx] in energy_learning_tokens:
                    continue
                self._online.observe_positive_pair(focus_tokens[idx], focus_tokens[jdx], weight=0.45)
        for idx in range(0, len(tokens)):
            for jdx in range(idx + 1, min(len(tokens), idx + 3)):
                if tokens[idx] in energy_learning_tokens or tokens[jdx] in energy_learning_tokens:
                    continue
                self._online.observe_positive_pair(tokens[idx], tokens[jdx], weight=0.22)
        if previous is not None:
            previous_features = self._snapshot_features_by_id.get(str(previous.get("memory_id", "") or "")) or self._build_snapshot_features(previous)
            previous_tokens = self._unique_tokens(previous_features["vector_tokens"])
            next_tokens = self._unique_tokens(tokens)
            for source in previous_tokens[:4]:
                for candidate in next_tokens[:4]:
                    if source in energy_learning_tokens or candidate in energy_learning_tokens:
                        continue
                    self._online.observe_transition_pair(source, candidate, weight=0.35)
        self._record_structured_learning_events(structured_events)
        self._runtime_state_dirty = True
        self._maybe_persist_runtime_state()

    def _restore_relations_from_loaded(self, loaded_snapshots: list[dict]) -> int:
        """
        One-time learning-only pass over warm-loaded seed snapshots.

        Rebuilds the online embedder's token co-occurrence/transition table from
        the seed bank by replaying ONLY the relation-learning events
        (`observe_*`). It does not reinject energy, requeue index jobs, or touch
        C* -- so it restores semantic distances (red-line-2 safe) rather than
        recounting occurrences. Returns the number of snapshots learned.

        The embedder normally throttles to `per_tick_update_limit` updates per
        tick (realtime profile = 1). That cap is for the live tick loop; a
        one-time restore must not be throttled, so we lift it for the pass and
        step a synthetic tick per snapshot to keep `begin_tick` bookkeeping sane.
        """

        if not loaded_snapshots:
            return 0
        online = self._online
        saved_limit = getattr(online, "per_tick_update_limit", 1)
        saved_tick = getattr(online, "_current_tick", -1)
        learned = 0
        try:
            # Lift throttle: each snapshot can contribute all its relation pairs.
            online.per_tick_update_limit = 1_000_000
            previous_by_kind: dict[str, dict] = {}
            synthetic_tick = int(saved_tick) + 1
            for snapshot in loaded_snapshots:
                memory_kind = str(snapshot.get("memory_kind", "") or "")
                online.begin_tick(synthetic_tick)
                synthetic_tick += 1
                self._learn_from_snapshot(snapshot, previous_by_kind.get(memory_kind))
                previous_by_kind[memory_kind] = snapshot
                learned += 1
        finally:
            online.per_tick_update_limit = saved_limit
            online._current_tick = saved_tick
        return learned

    def _total_transition_learning_events(self) -> int:
        return int(self._relation_learning_events_total + self._multimodal_learning_events_total)

    def _record_multimodal_learning_events(self, events: list[dict]) -> None:
        if not events:
            return
        clean_events = [dict(event) for event in events[: self._multimodal_event_preview_limit]]
        self._last_multimodal_learning_events = clean_events
        self._multimodal_learning_events_total += len(events)

    def _record_energy_learning_events(self, events: list[dict]) -> None:
        if not events:
            return
        clean_events = [dict(event) for event in events[: self._energy_event_preview_limit]]
        self._last_energy_learning_events = clean_events
        self._energy_learning_events_total += len(events)

    def _record_relation_learning_events(self, events: list[dict]) -> None:
        if not events:
            return
        clean_events = [dict(event) for event in events[: self._multimodal_event_preview_limit]]
        self._last_relation_learning_events = clean_events
        self._relation_learning_events_total += len(events)

    def _record_structured_learning_events(self, events: list[dict]) -> None:
        if not events:
            return
        clean_events = [dict(event) for event in events[: self._energy_event_preview_limit]]
        self._last_structured_learning_events = clean_events
        self._structured_learning_events_total += len(events)
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type", "") or "")
            layer = str(event.get("learning_layer", "") or "")
            writer = str(event.get("writer", "") or "")
            rule = str(event.get("bc_rule_id", "") or "")
            if event_type:
                self._structured_learning_total_by_type[event_type] += 1
            if layer:
                self._structured_learning_total_by_layer[layer] += 1
            if writer:
                self._structured_learning_total_by_writer[writer] += 1
            if rule:
                self._structured_learning_total_by_rule[rule] += 1
        self._maybe_persist_runtime_state()

    def _maybe_persist_runtime_state(self) -> None:
        if self._runtime_state_persist_suspended > 0:
            return
        if not self._runtime_state_dirty:
            return
        if self._runtime_state_last_persist_revision == self._memory_revision:
            return
        queue = getattr(self._persistence, "queue_runtime_state_supplier", None)
        if callable(queue):
            queue(lambda: self._runtime_state_payload(reason="online_learning_tick"), reason="online_learning_tick")
            self._runtime_state_last_persist_revision = int(self._memory_revision)
            self._runtime_state_dirty = False
            return
        self._persist_runtime_state(reason="online_learning_tick")

    def _structured_energy_events(self, snapshot: dict, events: list[dict]) -> list[dict]:
        rows: list[dict] = []
        tick_index = int((snapshot or {}).get("tick_index", -1) or -1)
        memory_id = str((snapshot or {}).get("memory_id", "") or "")
        memory_kind = str((snapshot or {}).get("memory_kind", "") or "")
        for event in events or []:
            relation = str((event or {}).get("relation", "") or "")
            if relation == "negative":
                bc_rule_id = "BC-002"
                event_type = "prediction_error_negative"
                meaning = "negative cognitive pressure means overprediction; push the missed prediction away from actual real-energy anchors"
            else:
                bc_rule_id = "BC-001"
                event_type = "prediction_error_positive"
                meaning = "positive cognitive pressure means underprediction; pull the surprising real subject toward current real-energy anchors"
            rows.append(
                self._structured_learning_event_builder.memory_event(
                    raw_event=event,
                    tick_index=tick_index,
                    memory_id=memory_id,
                    memory_kind=memory_kind,
                    bc_rule_id=bc_rule_id,
                    event_type=event_type,
                    learning_layer="content_recognition_embedding",
                    writer="MemoryStore._learn_from_snapshot",
                    write_mode="direct_online_update",
                    meaning=meaning,
                    guards=self._structured_learning_event_builder.concept_guards(),
                )
            )
        return rows

    def _structured_relation_events(self, snapshot: dict, events: list[dict]) -> list[dict]:
        rows: list[dict] = []
        tick_index = int((snapshot or {}).get("tick_index", -1) or -1)
        memory_id = str((snapshot or {}).get("memory_id", "") or "")
        memory_kind = str((snapshot or {}).get("memory_kind", "") or "")
        for event in events or []:
            rows.append(
                self._structured_learning_event_builder.memory_event(
                    raw_event=event,
                    tick_index=tick_index,
                    memory_id=memory_id,
                    memory_kind=memory_kind,
                    bc_rule_id="BC-003",
                    event_type="order_transition",
                    learning_layer="relation_order_embedding",
                    writer="MemoryStore._learn_from_snapshot",
                    write_mode="direct_online_update",
                    meaning="directed order and relation evidence teaches asymmetric recall channels",
                    guards={"concept_guard": True, "policy": "directed_transition_learning_not_symmetric_concept_merge"},
                )
            )
        return rows

    def _structured_multimodal_events(self, snapshot: dict, events: list[dict]) -> list[dict]:
        rows: list[dict] = []
        tick_index = int((snapshot or {}).get("tick_index", -1) or -1)
        memory_id = str((snapshot or {}).get("memory_id", "") or "")
        memory_kind = str((snapshot or {}).get("memory_kind", "") or "")
        for event in events or []:
            relation = str((event or {}).get("relation", "") or "")
            if relation == "transition":
                bc_rule_id = "BC-003"
                event_type = "order_transition"
                layer = "relation_order_embedding"
                meaning = "handle continuity across ticks teaches successor tendency without replacing spacetime transitions"
            else:
                bc_rule_id = "BC-005"
                event_type = "multimodal_binding"
                layer = "multimodal_binding_embedding"
                meaning = "cross-modal co-presence binds text vision and audio handles as recall evidence"
            rows.append(
                self._structured_learning_event_builder.memory_event(
                    raw_event=event,
                    tick_index=tick_index,
                    memory_id=memory_id,
                    memory_kind=memory_kind,
                    bc_rule_id=bc_rule_id,
                    event_type=event_type,
                    learning_layer=layer,
                    writer="MemoryStore._learn_from_snapshot",
                    write_mode="direct_online_update",
                    meaning=meaning,
                    guards=self._structured_learning_event_builder.concept_guards(),
                )
            )
        return rows

    def _build_relation_learning_events(self, snapshot: dict) -> list[dict]:
        relation_features = dict((snapshot or {}).get("relation_features", {}) or {})
        raw_events = list(relation_features.get("relation_events", []) or [])
        events: list[dict] = []
        for event in raw_events[: self.online_scoring_token_limit]:
            if not isinstance(event, dict):
                continue
            relation_token = str(event.get("relation_token", "") or "")
            source = str(event.get("source_label", "") or "")
            target = str(event.get("target_label", "") or "")
            relation_type = str(event.get("relation_type", "") or "")
            if not relation_token or not source or not target:
                continue
            weight = max(0.03, min(2.0, float(event.get("weight", 0.0) or 0.0)))
            events.append(
                {
                    "event_type": f"{relation_type or 'relation'}_online_transition",
                    "relation": "transition",
                    "source": relation_token,
                    "target": target,
                    "weight": _round4(weight),
                    "evidence": {
                        "source_label": source,
                        "target_label": target,
                        "relation_type": relation_type,
                        "relation_key": str(event.get("relation_key", "") or ""),
                        "white_box_relation_token": relation_token,
                    },
                }
            )
            if source != target:
                events.append(
                    {
                        "event_type": f"{relation_type or 'relation'}_token_binding",
                        "relation": "transition",
                        "source": source,
                        "target": relation_token,
                        "weight": _round4(weight * 0.65),
                        "evidence": {
                            "source_label": source,
                            "target_label": target,
                            "relation_type": relation_type,
                            "relation_key": str(event.get("relation_key", "") or ""),
                            "white_box_relation_token": relation_token,
                        },
                    }
                )
        events.sort(key=lambda row: (-float(row.get("weight", 0.0) or 0.0), str(row.get("event_type", "")), str(row.get("source", ""))))
        return events[: max(4, self.online_scoring_token_limit // 8)]

    def _energy_learning_token_set(self, events: list[dict]) -> set[str]:
        tokens: set[str] = set()
        for event in events or []:
            source = str((event or {}).get("source", "") or "")
            target = str((event or {}).get("target", "") or "")
            if source:
                tokens.add(source)
            if target:
                tokens.add(target)
        return tokens

    def _build_energy_learning_events(self, snapshot: dict) -> list[dict]:
        candidates = self._extract_energy_learning_candidates(snapshot)
        if not candidates:
            return []
        real_context = [
            row
            for row in candidates
            if row["real_energy"] >= self._energy_learning_real_threshold
        ]
        real_context.sort(
            key=lambda row: (
                -float(row.get("real_energy", 0.0) or 0.0),
                -float(row.get("attention_weight", 0.0) or 0.0),
                str(row.get("label", "")),
            )
        )
        real_context = real_context[: self._energy_learning_context_limit]
        positive_subjects = [
            row
            for row in candidates
            if row["cognitive_pressure"] >= self._energy_learning_pressure_threshold
            and row["real_energy"] >= self._energy_learning_real_threshold
        ]
        negative_subjects = [
            row
            for row in candidates
            if row["cognitive_pressure"] <= -self._energy_learning_pressure_threshold
        ]
        positive_subjects.sort(key=lambda row: (-float(row["cognitive_pressure"]), -float(row["real_energy"]), str(row["label"])))
        negative_subjects.sort(key=lambda row: (float(row["cognitive_pressure"]), -float(row["virtual_energy"]), str(row["label"])))
        events: list[dict] = []
        for subject in positive_subjects[: self._energy_learning_subject_limit]:
            for context in real_context:
                if context["label"] == subject["label"]:
                    continue
                weight = self._energy_positive_weight(subject, context)
                if weight <= 0.0:
                    continue
                events.append(
                    self._energy_learning_event(
                        "positive_pressure_real_context",
                        subject,
                        context,
                        relation="positive",
                        weight=weight,
                    )
                )
        for subject in negative_subjects[: self._energy_learning_subject_limit]:
            for context in real_context:
                if context["label"] == subject["label"]:
                    continue
                weight = self._energy_negative_weight(subject, context)
                if weight <= 0.0:
                    continue
                events.append(
                    self._energy_learning_event(
                        "negative_pressure_real_context",
                        subject,
                        context,
                        relation="negative",
                        weight=weight,
                    )
                )
        events.sort(key=lambda row: (-abs(float(row.get("weight", 0.0) or 0.0)), str(row.get("relation", "")), str(row.get("source", "")), str(row.get("target", ""))))
        return events[: max(4, self.online_scoring_token_limit // 8)]

    def _extract_energy_learning_candidates(self, snapshot: dict | None) -> list[dict]:
        if not isinstance(snapshot, dict):
            return []
        rows: list[dict] = []
        source_items = self._snapshot_state_field_items(snapshot)[: self.core_item_limit]
        focus_set = {str(label or "") for label in (snapshot.get("focus_labels", []) or []) if str(label or "")}
        for item in source_items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or not self._is_learning_association_item(item):
                continue
            real_energy = max(0.0, float(item.get("real_energy", 0.0) or 0.0))
            virtual_energy = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
            pressure = float(item.get("cognitive_pressure", real_energy - virtual_energy) or 0.0)
            if real_energy <= 0.0 and virtual_energy <= 0.0 and abs(pressure) <= 0.0:
                continue
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            is_focus = bool(anchor_meta.get("is_focus", False)) or label in focus_set
            attention_weight = float(item.get("attention_weight", item.get("query_weight", 0.0)) or 0.0)
            rows.append(
                {
                    "label": label,
                    "family": str(item.get("family", "") or ""),
                    "source_type": str(item.get("source_type", "") or ""),
                    "real_energy": _round4(real_energy),
                    "virtual_energy": _round4(virtual_energy),
                    "cognitive_pressure": _round4(pressure),
                    "attention_weight": _round4(attention_weight),
                    "is_focus": is_focus,
                    "is_external": self._is_external_evidence_item(item),
                }
            )
            if len(rows) >= max(self._energy_learning_context_limit * 3, 64):
                break
        return rows

    def _is_learning_association_item(self, item: dict) -> bool:
        label = str(item.get("sa_label", "") or "")
        source_type = str(item.get("source_type", "") or "")
        family = str(item.get("family", "") or "")
        if label.startswith(("reward::", "punishment::", "rwd::", "pun::")):
            return False
        if "reward" in label.lower() or "punishment" in label.lower():
            return False
        # APV2.1 P1-K-4: action/feeling/emotion/control SAs are allowed to take
        # part in whole-field intuition learning. We only keep explicit
        # reward/punishment feedback out of this symmetric pressure-learning
        # channel; action outcome memory remains the writer for observed
        # consequence values.
        if source_type in {"action_feedback"} or family in {"action_feedback"} or label.startswith("action_feedback::"):
            return False
        return True

    def _energy_positive_weight(self, subject: dict, context: dict) -> float:
        pressure = max(0.0, float(subject.get("cognitive_pressure", 0.0) or 0.0))
        subject_real = max(0.0, float(subject.get("real_energy", 0.0) or 0.0))
        context_real = max(0.0, float(context.get("real_energy", 0.0) or 0.0))
        subject_signal = self._energy_real_signal(subject_real)
        context_signal = self._energy_real_signal(context_real)
        pressure_signal = self._energy_pressure_signal(pressure)
        if pressure_signal <= 0.0 or subject_signal <= 0.0 or context_signal <= 0.0:
            return 0.0
        weight = pressure_signal * subject_signal * context_signal * 2.4
        weight *= self._energy_context_gate(subject, context)
        return _round4(min(2.4, weight))

    def _energy_negative_weight(self, subject: dict, context: dict) -> float:
        pressure = abs(min(0.0, float(subject.get("cognitive_pressure", 0.0) or 0.0)))
        subject_virtual = max(0.0, float(subject.get("virtual_energy", 0.0) or 0.0))
        context_real = max(0.0, float(context.get("real_energy", 0.0) or 0.0))
        pressure_signal = self._energy_pressure_signal(pressure)
        virtual_signal = self._energy_pressure_signal(subject_virtual)
        context_signal = self._energy_real_signal(context_real)
        if pressure_signal <= 0.0 or context_signal <= 0.0:
            return 0.0
        # Missed predictions have little or no real energy by definition, so
        # negative learning is gated by how strong the actual context is and how
        # strongly the missed object was virtually committed.
        weight = pressure_signal * context_signal * (0.65 + virtual_signal * 0.35) * 2.6
        weight *= self._energy_context_gate(subject, context)
        return _round4(min(2.8, weight))

    def _energy_real_signal(self, real_energy: float) -> float:
        real = max(0.0, float(real_energy or 0.0))
        if real < self._energy_learning_real_threshold:
            return 0.0
        return _round4((real / (real + self._energy_learning_real_softcap)) ** 0.5)

    def _energy_pressure_signal(self, pressure: float) -> float:
        value = max(0.0, float(pressure or 0.0))
        if value < self._energy_learning_pressure_threshold:
            return 0.0
        return _round4((value / (value + self._energy_learning_pressure_softcap)) ** 0.5)

    def _energy_context_gate(self, subject: dict, context: dict) -> float:
        gate = 0.55
        if subject.get("is_focus") or context.get("is_focus"):
            gate += 0.18
        if subject.get("is_external") or context.get("is_external"):
            gate += 0.14
        if subject.get("source_type") == context.get("source_type"):
            gate += 0.06
        elif subject.get("family") == context.get("family"):
            gate += 0.04
        return min(1.0, gate)

    def _energy_learning_event(self, event_type: str, source: dict, target: dict, *, relation: str, weight: float) -> dict:
        return {
            "event_type": str(event_type or ""),
            "relation": str(relation or "positive"),
            "source": str(source.get("label", "") or ""),
            "target": str(target.get("label", "") or ""),
            "weight": _round4(weight),
            "evidence": {
                "source_real": _round4(float(source.get("real_energy", 0.0) or 0.0)),
                "source_virtual": _round4(float(source.get("virtual_energy", 0.0) or 0.0)),
                "source_pressure": _round4(float(source.get("cognitive_pressure", 0.0) or 0.0)),
                "target_real": _round4(float(target.get("real_energy", 0.0) or 0.0)),
                "target_virtual": _round4(float(target.get("virtual_energy", 0.0) or 0.0)),
                "target_pressure": _round4(float(target.get("cognitive_pressure", 0.0) or 0.0)),
                "source_real_signal": self._energy_real_signal(float(source.get("real_energy", 0.0) or 0.0)),
                "target_real_signal": self._energy_real_signal(float(target.get("real_energy", 0.0) or 0.0)),
                "source_pressure_signal": self._energy_pressure_signal(abs(float(source.get("cognitive_pressure", 0.0) or 0.0))),
                "source_focus": bool(source.get("is_focus", False)),
                "target_focus": bool(target.get("is_focus", False)),
                "source_external": bool(source.get("is_external", False)),
                "target_external": bool(target.get("is_external", False)),
            },
        }

    def _build_multimodal_learning_events(self, snapshot: dict, previous: dict | None) -> list[dict]:
        current_handles = self._extract_learning_handles(snapshot)
        if not current_handles:
            return []
        previous_handles = self._extract_learning_handles(previous) if previous is not None else []
        events: list[dict] = []

        text_handles = [row for row in current_handles if row["modality"] == "text"]
        visual_handles = [row for row in current_handles if row["modality"] == "vision"]
        audio_handles = [row for row in current_handles if row["modality"] == "audio"]

        # Cross-modal teaching/co-presence: bounded pair construction.
        for left in text_handles[:6]:
            for right in (visual_handles + audio_handles)[:6]:
                events.append(self._learning_event("cross_modal_copresence_positive", left, right, relation="positive", weight=0.72))
        for left in visual_handles[:4]:
            for right in audio_handles[:4]:
                events.append(self._learning_event("vision_audio_copresence_positive", left, right, relation="positive", weight=0.54))

        # Same-focus / same-scene handle co-presence.
        non_text = (visual_handles + audio_handles)[:6]
        for left, right in combinations(non_text, 2):
            events.append(self._learning_event("same_scene_handle_positive", left, right, relation="positive", weight=0.46))

        # Adjacent tick slot continuity and learned successor.
        previous_by_label = {row["label"]: row for row in previous_handles}
        previous_by_slot = {(row["modality"], row.get("slot_key", "")): row for row in previous_handles if row.get("slot_key")}
        for current in (visual_handles + audio_handles)[:8]:
            prev = previous_by_label.get(current["label"])
            if prev is None and current.get("slot_key"):
                prev = previous_by_slot.get((current["modality"], current.get("slot_key", "")))
            if prev is None:
                continue
            continuity = self._handle_continuity_score(prev, current)
            if continuity <= 0.12:
                continue
            events.append(self._learning_event("handle_continuity_positive", prev, current, relation="positive", weight=continuity))
            events.append(self._learning_event("handle_transition_successor", prev, current, relation="transition", weight=continuity))

        # Keep event construction bounded before it reaches OnlineEmbeddingStore.
        events.sort(key=lambda row: (-float(row.get("weight", 0.0) or 0.0), str(row.get("event_type", "")), str(row.get("source", "")), str(row.get("target", ""))))
        return events[: max(4, self.online_scoring_token_limit // 8)]

    def _extract_learning_handles(self, snapshot: dict | None) -> list[dict]:
        if not isinstance(snapshot, dict):
            return []
        rows: list[dict] = []
        for item in self._snapshot_state_field_items(snapshot)[: self.core_item_limit]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            modality = ""
            if label.startswith(("text::", "phrase::")) or family in {"text", "learned_text_phrase"} or source_type == "external_text":
                modality = "text"
            elif label.startswith("vision_obj::") or family == "vision_object":
                modality = "vision"
            elif label.startswith("audio_event::") or family == "audio_event":
                modality = "audio"
            if not modality:
                continue
            if modality in {"vision", "audio"} and not bool(anchor_meta.get("learnable_handle", False)):
                continue
            rows.append(
                {
                    "label": label,
                    "modality": modality,
                    "slot_key": self._handle_slot_key(item),
                    "tick_index": int(snapshot.get("tick_index", item.get("tick_index", anchor_meta.get("tick_index", -1))) or -1),
                    "energy": _round4(float(item.get("real_energy", item.get("query_weight", 0.0)) or 0.0)),
                    "numeric_features": self._extract_numeric_features(item),
                    "focus": bool(anchor_meta.get("is_focus", False)),
                }
            )
            if len(rows) >= 32:
                break
        return rows

    def _handle_slot_key(self, item: dict) -> str:
        label = str((item or {}).get("sa_label", "") or "")
        anchor_meta = dict((item or {}).get("anchor_meta", {}) or {}) if isinstance((item or {}).get("anchor_meta", {}), dict) else {}
        if label.startswith("vision_obj::slot_"):
            return label.split("vision_obj::", 1)[-1]
        if label.startswith("audio_event::"):
            return label.split("audio_event::", 1)[-1]
        if "track_slot" in anchor_meta:
            return f"slot_{anchor_meta.get('track_slot')}"
        return ""

    def _learning_event(self, event_type: str, source: dict, target: dict, *, relation: str, weight: float) -> dict:
        return {
            "event_type": str(event_type or ""),
            "relation": str(relation or "positive"),
            "source": str(source.get("label", "") or ""),
            "target": str(target.get("label", "") or ""),
            "weight": _round4(weight),
            "evidence": {
                "source_modality": str(source.get("modality", "") or ""),
                "target_modality": str(target.get("modality", "") or ""),
                "source_slot": str(source.get("slot_key", "") or ""),
                "target_slot": str(target.get("slot_key", "") or ""),
                "source_energy": _round4(float(source.get("energy", 0.0) or 0.0)),
                "target_energy": _round4(float(target.get("energy", 0.0) or 0.0)),
            },
        }

    def _handle_continuity_score(self, previous: dict, current: dict) -> float:
        prev_numeric = dict(previous.get("numeric_features", {}) or {})
        cur_numeric = dict(current.get("numeric_features", {}) or {})
        shared = sorted(set(prev_numeric) & set(cur_numeric))
        if not shared:
            return 0.35 if previous.get("label") == current.get("label") else 0.0
        scores = []
        for channel in shared:
            scores.append(self._cosine_numeric(prev_numeric[channel], cur_numeric[channel]))
        base = sum(scores) / max(1, len(scores))
        if previous.get("slot_key") and previous.get("slot_key") == current.get("slot_key"):
            base = min(1.0, base + 0.12)
        return _round4(max(0.0, base))

    def _cosine_numeric(self, left: list[float], right: list[float]) -> float:
        usable = min(len(left or []), len(right or []))
        if usable <= 0:
            return 0.0
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for idx in range(usable):
            a = float(left[idx] or 0.0)
            b = float(right[idx] or 0.0)
            dot += a * b
            left_norm += a * a
            right_norm += b * b
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            return 0.0
        return max(0.0, dot / ((left_norm ** 0.5) * (right_norm ** 0.5)))

    def _build_sequence_features(self, items: list[dict], focus_labels: list[str] | None = None) -> dict:
        ordered_rows = []
        for index, item in enumerate(items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            anchor_meta = dict(item.get("anchor_meta", {}) or {})
            position = item.get("position", anchor_meta.get("position", index))
            try:
                position_key = int(position)
            except (TypeError, ValueError):
                position_key = index
            last_seen = item.get("last_seen_tick", item.get("tick_index", anchor_meta.get("tick_index", -1)))
            try:
                tick_key = int(last_seen)
            except (TypeError, ValueError):
                tick_key = -1
            ordered_rows.append(
                {
                    "sa_label": label,
                    "position": position_key,
                    "tick_index": tick_key,
                    "source_type": str(item.get("source_type", "") or ""),
                    "family": str(item.get("family", "") or ""),
                    "order_index": index,
                }
            )
        ordered_rows.sort(key=lambda row: (int(row.get("tick_index", -1)), int(row.get("position", 0)), int(row.get("order_index", 0)), str(row.get("sa_label", ""))))
        ordered_labels = [str(row.get("sa_label", "") or "") for row in ordered_rows if str(row.get("sa_label", "") or "")]
        focus_order = [str(label or "") for label in (focus_labels or []) if str(label or "")]
        sequence_bigrams = [f"seq::{left}>>{right}" for left, right in zip(ordered_labels, ordered_labels[1:])]
        focus_bigrams = [f"focus_seq::{left}>>{right}" for left, right in zip(focus_order, focus_order[1:])]
        return {
            "ordered_labels": ordered_labels[: self.core_item_limit],
            "focus_order": focus_order[:64],
            "sequence_bigrams": (sequence_bigrams + focus_bigrams)[: self.core_item_limit],
        }

    def _energy_profile(self, items: list[dict], *, limit: int) -> dict[str, float]:
        cap = max(1, int(limit))
        profile: dict[str, float] = {}
        count = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            weight = float(item.get("query_weight", item.get("real_energy", 0.0)) or 0.0)
            weight += float(item.get("virtual_energy", 0.0) or 0.0) * 0.65
            weight += abs(float(item.get("cognitive_pressure", 0.0) or 0.0)) * 0.25
            if weight <= 0.0:
                continue
            profile[label] = profile.get(label, 0.0) + float(weight)
            count += 1
            if count >= cap:
                break
        return profile

    def _energy_mass(self, energy: dict[str, float]) -> float:
        return sum(float(value or 0.0) for value in (energy or {}).values())

    def _query_real_virtual_mass(self, items: list[dict], *, limit: int) -> tuple[float, float]:
        cap = max(1, int(limit))
        real_mass = 0.0
        virtual_mass = 0.0
        count = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            real = max(0.0, float(item.get("real_energy", 0.0) or 0.0))
            virtual = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
            query_weight = max(0.0, float(item.get("query_weight", 0.0) or 0.0))
            attention_gain = max(0.0, float(item.get("attention_gain", 0.0) or 0.0))
            if real <= 0.0 and query_weight > 0.0:
                real = query_weight * float(self.b_drive_fallback_real_factor)
            elif query_weight > real:
                real += (query_weight - real) * float(self.b_attention_real_carry_factor)
            real += attention_gain * float(self.b_attention_real_carry_factor)
            if real <= 0.0 and virtual <= 0.0 and query_weight <= 0.0:
                continue
            real_mass += real
            virtual_mass += virtual
            count += 1
            if count >= cap:
                break
        return real_mass, virtual_mass

    def _energy_overlap(
        self,
        query_energy: dict[str, float],
        snapshot_energy: dict[str, float],
        *,
        query_mass: float | None = None,
        snapshot_mass: float | None = None,
        specificity_by_label: dict[str, float] | None = None,
    ) -> float:
        if not query_energy or not snapshot_energy:
            return 0.0
        numerator = 0.0
        if len(query_energy) <= len(snapshot_energy):
            smaller = query_energy
            larger = snapshot_energy
        else:
            smaller = snapshot_energy
            larger = query_energy
        for label, value in smaller.items():
            other = larger.get(label)
            if other is None:
                continue
            shared = min(float(value or 0.0), float(other or 0.0))
            if specificity_by_label is not None:
                # Weight the shared-energy contribution by how discriminative the
                # label is. Ubiquitous generic labels (action::wait, feeling::*)
                # carry near-zero specificity and stop dominating the overlap by
                # sheer count; rare high-value anchors carry their full weight.
                # AP philosophy: prediction specificity, not shared-label count.
                shared *= float(specificity_by_label.get(label, 1.0))
            numerator += shared
        q_mass = self._energy_mass(query_energy) if query_mass is None else float(query_mass)
        s_mass = self._energy_mass(snapshot_energy) if snapshot_mass is None else float(snapshot_mass)
        denominator = q_mass + s_mass - numerator
        if denominator <= 1e-9:
            return 0.0
        return numerator / denominator

    def _numeric_feature_profile(self, items: list[dict], *, limit: int) -> dict[str, list[float]]:
        cap = max(1, int(limit))
        accum: dict[str, list[float]] = {}
        weights: dict[str, float] = {}
        count = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue
            numeric = self._extract_numeric_features(item)
            if not numeric:
                continue
            weight = float(item.get("query_weight", item.get("real_energy", 1.0)) or 0.0)
            weight += float(item.get("virtual_energy", 0.0) or 0.0) * 0.35
            weight = max(0.05, weight)
            for channel, vector in numeric.items():
                if not vector:
                    continue
                bucket = accum.setdefault(channel, [0.0] * len(vector))
                if len(bucket) < len(vector):
                    bucket.extend([0.0] * (len(vector) - len(bucket)))
                for idx, value in enumerate(vector):
                    bucket[idx] += float(value or 0.0) * weight
                weights[channel] = float(weights.get(channel, 0.0) or 0.0) + weight
            count += 1
            if count >= cap:
                break
        normalized: dict[str, list[float]] = {}
        for channel, vector in accum.items():
            denom = max(1e-9, float(weights.get(channel, 0.0) or 0.0))
            normalized[channel] = [_round4(float(value or 0.0) / denom) for value in vector]
        return normalized

    def _extract_numeric_features(self, item: dict) -> dict[str, list[float]]:
        if not isinstance(item, dict):
            return {}
        anchor_meta = item.get("anchor_meta", {})
        if not isinstance(anchor_meta, dict):
            anchor_meta = {}
        candidates = []
        for payload in (
            item.get("numeric_features", None),
            anchor_meta.get("numeric_features", None),
            anchor_meta.get("features", None),
        ):
            if isinstance(payload, dict):
                candidates.append(payload)
        rows: dict[str, list[float]] = {}
        for payload in candidates:
            for channel, values in payload.items():
                clean_channel = str(channel or "").strip()
                if not clean_channel:
                    continue
                vector = self._coerce_numeric_vector(values)
                if vector:
                    rows[clean_channel] = vector
        return rows

    def _coerce_numeric_vector(self, values: object) -> list[float]:
        if isinstance(values, dict):
            raw = [values[key] for key in sorted(values)]
        elif isinstance(values, (list, tuple)):
            raw = list(values)
        else:
            raw = [values]
        vector: list[float] = []
        for value in raw[:64]:
            try:
                vector.append(float(value))
            except (TypeError, ValueError):
                vector.append(0.0)
        return vector

    def _unique_tokens(self, tokens: list[str]) -> list[str]:
        seen = set()
        rows = []
        for token in tokens:
            clean = str(token or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(clean)
        return rows

    def _time_match(self, *, snapshot: dict, time_context: dict | None) -> float:
        if not time_context:
            return 0.0
        current_tick = time_context.get("current_tick")
        target_delta_t = time_context.get("target_delta_t")
        sigma = time_context.get("time_sigma")
        confidence = float(time_context.get("confidence", 0.0) or 0.0)
        gain = float(time_context.get("gain", 0.0) or 0.0)
        felt_energy = float(time_context.get("felt_energy", confidence) or confidence)
        if current_tick is None or target_delta_t is None or sigma in (None, 0):
            return 0.0
        memory_tick = snapshot.get("tick_index")
        if memory_tick is None:
            return 0.0
        current_tick = self._effective_runtime_tick(current_tick)
        memory_delta_t = max(0.0, float(current_tick) - float(memory_tick))
        sigma_value = max(1.0, float(sigma))
        interval_match = pow(
            2.718281828,
            -(((memory_delta_t - float(target_delta_t)) ** 2) / max(1e-6, 2.0 * sigma_value * sigma_value)),
        )
        return _round4(interval_match * gain * max(0.0, confidence) * max(0.0, felt_energy))

    def _current_tick_for_temporal(self, query_items: list[dict], *, time_context: dict | None = None) -> int | None:
        if time_context and time_context.get("current_tick") is not None:
            try:
                return self._effective_runtime_tick(time_context.get("current_tick"))
            except (TypeError, ValueError):
                return None
        ticks: list[int] = []
        for item in query_items or []:
            if not isinstance(item, dict):
                continue
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            tick_value = item.get("last_seen_tick", item.get("tick_index", anchor_meta.get("tick_index")))
            if tick_value is None:
                continue
            try:
                ticks.append(int(float(tick_value)))
            except (TypeError, ValueError):
                continue
        return self._effective_runtime_tick(max(ticks)) if ticks else None

    def _temporal_applicability(self, snapshot: dict | None, *, current_tick: int | None) -> dict:
        """
        Estimate how applicable a remembered state still is now.

        This is a pre-normalization recall weight, not deletion and not a
        semantic type gate. The whole all-SA snapshot is still a valid memory;
        older memories simply carry less current grasp unless a strong cue
        still makes their base similarity win.
        """

        if not self.temporal_applicability_enabled:
            return {
                "weight": 1.0,
                "age_ticks": None,
                "phase": "disabled",
                "policy": "temporal_applicability_disabled",
            }
        if not isinstance(snapshot, dict) or current_tick is None:
            return {
                "weight": 1.0,
                "age_ticks": None,
                "phase": "unknown_now",
                "policy": "no_current_tick_available",
            }
        memory_tick = snapshot.get("tick_index")
        if memory_tick is None:
            return {
                "weight": 1.0,
                "age_ticks": None,
                "phase": "unknown_memory_tick",
                "policy": "snapshot_has_no_tick_index",
            }
        try:
            age_ticks = max(0, int(float(current_tick)) - int(float(memory_tick)))
        except (TypeError, ValueError):
            return {
                "weight": 1.0,
                "age_ticks": None,
                "phase": "invalid_tick",
                "policy": "tick_parse_failed",
            }

        weight = 1.0
        phase = "fresh"
        if self.temporal_fatigue_window_ticks > 0 and age_ticks < self.temporal_fatigue_window_ticks:
            progress = age_ticks / max(1.0, float(self.temporal_fatigue_window_ticks))
            recovered = progress ** float(self.temporal_fatigue_recovery_exponent)
            weight *= 1.0 - float(self.temporal_fatigue_strength) * (1.0 - recovered)
            phase = "short_fatigue"
        elif age_ticks <= self.temporal_recent_gain_window_ticks:
            progress = age_ticks / max(1.0, float(self.temporal_recent_gain_window_ticks))
            weight *= 1.0 + float(self.temporal_recent_gain) * (1.0 - progress)
            phase = "recent_gain"
        else:
            long_age = age_ticks - self.temporal_recent_gain_window_ticks
            half_life = max(1.0, float(self.temporal_long_half_life_ticks))
            decay = 0.5 ** (float(long_age) / half_life)
            weight *= self.temporal_floor + (1.0 - self.temporal_floor) * decay
            phase = "long_decay"

        lower_bound = 0.02 if phase == "short_fatigue" else self.temporal_floor
        return {
            "weight": _round4(_clamp(weight, lower_bound, 1.0 + self.temporal_recent_gain)),
            "age_ticks": int(age_ticks),
            "phase": phase,
            "fatigue_recovery_exponent": _round4(self.temporal_fatigue_recovery_exponent),
            "policy": "short_fatigue_power_recovery_then_recent_gain_then_long_half_life_floor",
        }

    def _apply_successor_temporal_applicability(
        self,
        rows: list[dict],
        *,
        current_tick: int | None,
        source_b_row: dict | None = None,
    ) -> list[dict]:
        clean_rows = [dict(row) for row in rows or [] if isinstance(row, dict)]
        if not clean_rows:
            return []
        effective_tick = current_tick
        if effective_tick is None and source_b_row:
            snapshot_ref = dict((source_b_row or {}).get("snapshot_ref", {}) or {})
            for value in ((source_b_row or {}).get("query_tick"), snapshot_ref.get("query_tick")):
                if value is None:
                    continue
                try:
                    effective_tick = int(float(value))
                    break
                except (TypeError, ValueError):
                    continue
        for row in clean_rows:
            successor_id = str(row.get("successor_memory_id", "") or "")
            snapshot = self._snapshot_by_id.get(successor_id)
            temporal = self._temporal_applicability(snapshot, current_tick=effective_tick)
            before = float(row.get("score", 0.0) or 0.0)
            weight = float(temporal.get("weight", 1.0) or 1.0)
            row["score_before_temporal"] = _round4(before)
            row["score"] = _round4(before * weight)
            row["temporal_age_ticks"] = temporal.get("age_ticks")
            row["temporal_applicability"] = _round4(weight)
            row["temporal_applicability_phase"] = str(temporal.get("phase", "") or "")
            row["temporal_applicability_policy"] = str(temporal.get("policy", "") or "")
            transfer = dict(row.get("energy_transfer", {}) or {})
            if transfer:
                transfer["temporal_applicability"] = _round4(weight)
                transfer["temporal_age_ticks"] = temporal.get("age_ticks")
                transfer["temporal_phase"] = str(temporal.get("phase", "") or "")
                row["energy_transfer"] = transfer
            scaled_items = []
            for item in list(row.get("predicted_items", []) or []):
                if not isinstance(item, dict):
                    continue
                scaled = dict(item)
                scaled["virtual_energy_before_temporal"] = _round4(float(scaled.get("virtual_energy", 0.0) or 0.0))
                scaled["virtual_energy"] = _round4(float(scaled.get("virtual_energy", 0.0) or 0.0) * weight)
                meta = dict(scaled.get("anchor_meta", {}) or {}) if isinstance(scaled.get("anchor_meta", {}), dict) else {}
                transfer_meta = dict(meta.get("prediction_energy_transfer", {}) or {})
                if transfer_meta:
                    transfer_meta["temporal_applicability"] = _round4(weight)
                    transfer_meta["temporal_age_ticks"] = temporal.get("age_ticks")
                    transfer_meta["temporal_phase"] = str(temporal.get("phase", "") or "")
                    meta["prediction_energy_transfer"] = transfer_meta
                    scaled["anchor_meta"] = meta
                scaled_items.append(scaled)
            if scaled_items:
                row["predicted_items"] = scaled_items
        return clean_rows

    def _build_prediction_payload_items(self, snapshot: dict, *, limit: int | None = None) -> list[dict]:
        cap = max(1, int(limit if limit is not None else self.predict_top_k))
        source_items = snapshot.get("state_field_items", None)
        if not isinstance(source_items, list) or not source_items:
            source_items = snapshot.get("core_items", None)
        if not isinstance(source_items, list) or not source_items:
            source_items = snapshot.get("items", []) or []
        full_items = [row for row in list(snapshot.get("items", []) or []) if isinstance(row, dict)]
        process_source_items = self._current_process_anchor_text_items(full_items, snapshot_tick=snapshot.get("tick_index"))
        if process_source_items:
            merged_source_items = list(process_source_items)
            seen_source_labels = {
                str(row.get("sa_label", "") or "")
                for row in merged_source_items
                if isinstance(row, dict) and str(row.get("sa_label", "") or "")
            }
            for row in list(source_items):
                if not isinstance(row, dict):
                    continue
                label = str(row.get("sa_label", "") or "")
                if label and label in seen_source_labels:
                    continue
                if label:
                    seen_source_labels.add(label)
                merged_source_items.append(row)
            source_items = merged_source_items
        successor_tick = snapshot.get("tick_index")
        external_now = []
        outcome_now = []
        action_now = []
        current_text_now = []
        current_text_action_now = []
        current_revision_now = []
        text_now = []
        text_action_now = []
        revision_now = []
        rest = []
        for row in list(source_items):
            if not isinstance(row, dict):
                continue
            label = str(row.get("sa_label", "") or "")
            family = str(row.get("family", "") or "")
            source_type = str(row.get("source_type", "") or "")
            meta = dict(row.get("anchor_meta", {}) or {}) if isinstance(row.get("anchor_meta", {}), dict) else {}
            if label.startswith("text::") or family in {"text", "learned_text_phrase"}:
                if self._is_negative_feedback_text_payload_item(row):
                    revision_now.append(self._negative_text_payload_as_revision(row))
                    continue
                if self._is_current_teacher_text_payload_item(row, snapshot_tick=successor_tick):
                    current_text_now.append(row)
                else:
                    text_now.append(row)
                continue
            if label.startswith("text_action::") or family == "text_action" or source_type == "text_action":
                if self._is_negative_feedback_text_payload_item(row):
                    revision_now.append(self._negative_text_payload_as_revision(row))
                    continue
                if self._is_current_teacher_text_payload_item(row, snapshot_tick=successor_tick):
                    current_text_action_now.append(row)
                else:
                    text_action_now.append(row)
                continue
            if (
                label.startswith("text_revision_opportunity::")
                or family == "text_revision_opportunity"
                or str(meta.get("schema_id", "") or "") == "text_revision_opportunity/v1"
                or label.startswith("text_slot_confirmation::")
                or family == "text_slot_confirmation"
                or str(meta.get("schema_id", "") or "") == "text_slot_confirmation/v1"
            ):
                if self._is_current_teacher_text_payload_item(row, snapshot_tick=successor_tick):
                    current_revision_now.append(row)
                else:
                    revision_now.append(row)
                continue
            if self._is_action_prediction_item(row):
                action_now.append(row)
                continue
            if (
                successor_tick is not None
                and int(row.get("last_seen_tick", -999999) or -999999) == int(successor_tick)
                and self._is_external_evidence_item(row)
            ):
                external_now.append(row)
            else:
                rest.append(row)
        for row in list(snapshot.get("action_feedback_items", []) or []):
            if not isinstance(row, dict):
                continue
            outcome_now.append(row)
        for row in list(snapshot.get("items", []) or []):
            if not isinstance(row, dict):
                continue
            label = str(row.get("sa_label", "") or "")
            family = str(row.get("family", "") or "")
            source_type = str(row.get("source_type", "") or "")
            is_explicit_signal = label.startswith(("signal::reward", "signal::punishment", "reward::", "punishment::", "rwd::", "pun::"))
            is_action_feedback = label.startswith("action_feedback::") or family == "action_feedback" or source_type == "action_feedback"
            is_action = self._is_action_prediction_item(row)
            if is_action and not is_action_feedback:
                if not any(str(existing.get("sa_label", "") or "") == label for existing in action_now):
                    action_now.append(row)
                continue
            if not (is_explicit_signal or is_action_feedback):
                continue
            if any(str(existing.get("sa_label", "") or "") == label for existing in outcome_now):
                continue
            outcome_now.append(row)
        # P1-K-4: successor payload is another readout of the same all-SA field,
        # not a special escape hatch for non-core action nodes. Keeping actions
        # here lets remembered tendencies refill virtual energy and become drive.
        current_reserved_cap = max(1, min(cap, self.successor_payload_text_reserve_limit + self.successor_payload_text_action_reserve_limit + 1))
        current_companion_tokens = {
            token
            for token in (
                self._text_payload_token(row)
                for row in list(current_revision_now) + list(current_text_action_now)
                if isinstance(row, dict)
            )
            if token
        }
        current_text_now = self._sort_text_process_payload_candidates(current_text_now)
        current_revision_now = self._sort_text_process_payload_candidates(current_revision_now)
        current_text_action_now = self._sort_text_process_payload_candidates(current_text_action_now)
        text_now = self._sort_text_process_payload_candidates(text_now)
        revision_now = self._sort_text_process_payload_candidates(revision_now)
        text_action_now = self._sort_text_process_payload_candidates(text_action_now)

        current_text_from_companions = [
            row
            for row in list(text_now)
            if self._text_payload_token(row) in current_companion_tokens
        ]
        text_payload_candidates = self._augment_current_text_payload_items(
            list(current_text_now)
            + list(current_text_from_companions[: self.successor_payload_text_reserve_limit])
            + list(current_revision_now[: max(2, self.successor_payload_text_reserve_limit)])
            + list(current_text_action_now[: self.successor_payload_text_action_reserve_limit])
        )
        current_reserved = self._dedupe_prediction_payload_items(
            text_payload_candidates,
            limit=current_reserved_cap,
        )
        historical_text_payload_candidates = self._augment_current_text_payload_items(
            list(text_now[: max(0, self.successor_payload_text_reserve_limit - len(current_reserved))])
            + list(revision_now[: max(1, self.successor_payload_text_reserve_limit // 2)])
            + list(text_action_now[: self.successor_payload_text_action_reserve_limit])
        )
        historical_reserved = self._dedupe_prediction_payload_items(
            historical_text_payload_candidates,
            limit=max(0, min(cap, self.successor_payload_text_reserve_limit + self.successor_payload_text_action_reserve_limit)),
            exclude_labels={str(item.get("sa_label", "") or "") for item in current_reserved},
        )
        revision_reserved = self._dedupe_prediction_payload_items(
            list(current_revision_now) + list(revision_now),
            limit=max(1, min(cap, self.successor_payload_text_reserve_limit)),
            exclude_labels={str(item.get("sa_label", "") or "") for item in current_reserved} | {str(item.get("sa_label", "") or "") for item in historical_reserved},
        )
        reserved = self._dedupe_prediction_payload_items(
            list(current_reserved) + list(revision_reserved) + list(historical_reserved),
            limit=max(0, min(cap, self.successor_payload_text_reserve_limit + self.successor_payload_text_action_reserve_limit + 1)),
        )
        reserved = [item for item in reserved if not self._is_negative_feedback_text_payload_item(item)]
        remaining_cap = max(0, cap - len(reserved))
        contextual = self._dedupe_prediction_payload_items(
            list(external_now) + list(outcome_now[:4]) + list(action_now[:4]) + list(rest),
            limit=remaining_cap,
            exclude_labels={str(item.get("sa_label", "") or "") for item in reserved},
        )
        contextual = [
            item
            for item in contextual
            if not self._is_negative_feedback_text_payload_item(item)
        ]
        return reserved + contextual

    def _current_process_anchor_text_items(self, items: list[dict], *, snapshot_tick: int | None) -> list[dict]:
        """
        Recover current low-grain text anchors before state-field truncation.

        UI readout ticks can contain hundreds of vision samples. The bounded
        state-field view may therefore drop the current `text::char` or its
        paired revision/action row even though the full same-tick memory
        snapshot contains it. Successor payloads need these process anchors so
        Cn can learn "this foveated glyph region led to this one-char action".
        This does not add a new answer; it preserves teacher-on material that
        was already written into the ordinary snapshot and remains hidden from
        strict teacher-off input.
        """

        rows: list[dict] = []
        companion_tokens: set[str] = set()
        for row in list(items or []):
            if not isinstance(row, dict):
                continue
            if self._is_current_teacher_text_payload_item(row, snapshot_tick=snapshot_tick) or self._is_negative_feedback_text_payload_item(row):
                token = self._text_payload_token(row)
                if token:
                    companion_tokens.add(token)
                rows.append(dict(row))
        if companion_tokens:
            existing_labels = {str(row.get("sa_label", "") or "") for row in rows if str(row.get("sa_label", "") or "")}
            for row in list(items or []):
                if not isinstance(row, dict):
                    continue
                label = str(row.get("sa_label", "") or "")
                if not label.startswith("text::") or label in existing_labels:
                    continue
                if self._text_payload_token(row) not in companion_tokens:
                    continue
                rows.append(dict(row))
                existing_labels.add(label)
        return self._sort_text_process_payload_candidates(rows)[: max(8, self.successor_payload_text_reserve_limit * 4)]

    def _augment_current_text_payload_items(self, rows: list[dict]) -> list[dict]:
        """
        Preserve charwise process anchors for successor prediction payloads.

        The state pool intentionally merges same-label SA objects, so a later
        `text::0` update may carry only the stable text identity while the
        position/process metadata lives on its paired
        `text_revision_opportunity` or `text_action` row. For foveated UI
        readout, Cn needs that process metadata to compete by "current unread
        glyph" rather than by naked text familiarity. This method copies only
        already-observed low-grain process fields; it does not add labels,
        OCR text, or teacher answers.
        """

        clean_rows = [dict(row) for row in list(rows or []) if isinstance(row, dict)]
        process_meta_by_token: dict[str, dict] = {}
        slot_process_meta_by_token: dict[str, dict] = {}
        for row in clean_rows:
            label = str(row.get("sa_label", "") or "")
            meta = dict(row.get("anchor_meta", {}) or {}) if isinstance(row.get("anchor_meta", {}), dict) else {}
            token = self._text_payload_token(row)
            if not token:
                continue
            if (
                label.startswith("text_slot_confirmation::")
                or str(meta.get("schema_id", "") or "") == "text_slot_confirmation/v1"
                or str(row.get("family", "") or "") == "text_slot_confirmation"
            ):
                slot_process_meta_by_token.setdefault(token, self._text_process_meta_subset(meta))
                process_meta_by_token.setdefault(token, self._text_process_meta_subset(meta))
                continue
            if (
                label.startswith(("text_revision_opportunity::", "text_slot_confirmation::"))
                or str(meta.get("schema_id", "") or "") in {"text_revision_opportunity/v1", "text_slot_confirmation/v1"}
                or str(row.get("family", "") or "") == "text_slot_confirmation"
            ):
                process_meta_by_token.setdefault(token, self._text_process_meta_subset(meta))
                continue
            if label.startswith("text_action::") or str(row.get("family", "") or "") == "text_action" or str(row.get("source_type", "") or "") == "text_action":
                process_meta_by_token.setdefault(token, self._text_process_meta_subset(meta))

        augmented: list[dict] = []
        emitted_text_tokens: set[str] = set()
        for row in clean_rows:
            if self._is_slot_confirmation_process_item(row):
                continue
            if self._is_negative_feedback_text_payload_item(row):
                continue
            token = self._text_payload_token(row)
            if token and str(row.get("sa_label", "") or "").startswith("text::"):
                emitted_text_tokens.add(token)
                process_meta = slot_process_meta_by_token.get(token, {}) or process_meta_by_token.get(token, {})
                if process_meta:
                    meta = dict(row.get("anchor_meta", {}) or {}) if isinstance(row.get("anchor_meta", {}), dict) else {}
                    for key, value in process_meta.items():
                        if key in {
                            "current_glyph_index",
                            "visible_length",
                            "cursor",
                            "cursor_index",
                            "current_glyph_role",
                            "same_tick_binding_role",
                            "prediction_payload_priority",
                            "readout_semantic_role",
                            "readout_pattern_id",
                            "semantic_frame_role",
                            "dynamic_slot_role",
                            "slot_role",
                            "previous_prefix",
                        }:
                            meta[key] = value
                        else:
                            meta.setdefault(key, value)
                    meta.setdefault("prediction_payload_priority", "current_glyph_character")
                    meta.setdefault("current_glyph_role", "read_tick_target")
                    meta["process_meta_restored_from_low_grain_companion"] = True
                    if token in slot_process_meta_by_token:
                        meta["process_meta_restored_from_slot_confirmation"] = True
                    meta["prediction_meta_boundary"] = "process_anchor_metadata_only_no_teacher_answer_or_ocr_text"
                    meta["used_in_strict_teacher_off_input"] = False
                    meta["answer_table_lookup"] = False
                    meta["full_string_or_sentence_action"] = False
                    row = dict(row)
                    row["anchor_meta"] = meta
            augmented.append(row)
        existing_labels = {str(row.get("sa_label", "") or "") for row in augmented if isinstance(row, dict)}
        for token, process_meta in list(slot_process_meta_by_token.items()) + list(process_meta_by_token.items()):
            if not token or token in emitted_text_tokens:
                continue
            if any(
                str((row.get("anchor_meta", {}) or {}).get("source_text_label", "") or "") == f"text::{token}"
                for row in clean_rows
                if isinstance(row, dict)
            ):
                continue
            label = f"text::{token}"
            if label in existing_labels:
                continue
            meta = dict(process_meta or {})
            if not meta:
                continue
            meta.setdefault("schema_id", "predicted_text_payload_from_process_companion/v1")
            meta.setdefault("prediction_payload_priority", "current_glyph_character")
            meta.setdefault("current_glyph_role", "read_tick_target")
            meta["process_meta_restored_from_low_grain_companion"] = True
            meta["prediction_meta_boundary"] = "process_anchor_metadata_only_no_teacher_answer_or_ocr_text"
            meta["used_in_strict_teacher_off_input"] = False
            meta["answer_table_lookup"] = False
            meta["full_string_or_sentence_action"] = False
            augmented.append(
                {
                    "sa_label": label,
                    "display_text": f"process anchored text token {token}",
                    "family": "text",
                    "source_type": "predicted_text_payload_context",
                    "real_energy": 0.20,
                    "virtual_energy": 0.18,
                    "cognitive_pressure": 0.12,
                    "anchor_meta": meta,
                }
            )
            existing_labels.add(label)
            emitted_text_tokens.add(token)
        return augmented

    def _is_slot_confirmation_process_item(self, row: dict) -> bool:
        if not isinstance(row, dict):
            return False
        label = str(row.get("sa_label", "") or "")
        family = str(row.get("family", "") or "")
        meta = dict(row.get("anchor_meta", {}) or {}) if isinstance(row.get("anchor_meta", {}), dict) else {}
        return bool(
            label.startswith("text_slot_confirmation::")
            or family == "text_slot_confirmation"
            or str(meta.get("schema_id", "") or "") == "text_slot_confirmation/v1"
        )

    def _text_process_anchor_can_be_positive_payload(self, row: dict) -> bool:
        """
        Keep process anchors as context unless they are actual insert/replace
        opportunities. Slot confirmations are metadata sources, not text
        candidates; otherwise a repeated digit slot can dominate the Cn payload
        as a naked answer-like item.
        """

        if self._is_slot_confirmation_process_item(row):
            return False
        label = str((row or {}).get("sa_label", "") or "")
        meta = dict((row or {}).get("anchor_meta", {}) or {}) if isinstance((row or {}).get("anchor_meta", {}), dict) else {}
        if label.startswith("text_revision_opportunity::"):
            return str(meta.get("operation", "") or "") in {"insert", "replace"}
        return True

    def _sort_text_process_payload_candidates(self, rows: list[dict]) -> list[dict]:
        """
        Keep current slot/process anchors ahead of generic text familiarity.

        This is a payload-retention rule, not a reading solver. It does not
        choose a character or add new text; it only prevents low-grain process
        metadata that is already present in the snapshot from being truncated
        behind older prefix/reread context.
        """

        clean_rows = [dict(row) for row in list(rows or []) if isinstance(row, dict)]

        def rank(row: dict) -> tuple:
            label = str(row.get("sa_label", "") or "")
            family = str(row.get("family", "") or "")
            meta = dict(row.get("anchor_meta", {}) or {}) if isinstance(row.get("anchor_meta", {}), dict) else {}
            priority = str(meta.get("prediction_payload_priority", "") or "")
            role = str(meta.get("current_glyph_role", "") or "")
            schema_id = str(meta.get("schema_id", "") or "")
            readout = bool(str(meta.get("readout_pattern_id", "") or "") or str(meta.get("readout_semantic_role", "") or ""))
            current = bool(
                meta.get("current_read_tick", False)
                or priority.startswith("current_glyph")
                or role.startswith("current_")
                or role == "read_tick_target"
            )
            if priority == "current_glyph_slot_confirmation" or schema_id == "text_slot_confirmation/v1" or family == "text_slot_confirmation":
                priority_rank = 0
            elif priority == "current_digit_boundary_insert_opportunity":
                priority_rank = 1
            elif priority == "current_glyph_transition_clean_insert_opportunity":
                priority_rank = 2
            elif priority == "current_glyph_insert_opportunity":
                priority_rank = 3
            elif priority == "current_glyph_character":
                priority_rank = 4
            elif priority == "current_cursor_context":
                priority_rank = 5
            elif priority == "previous_prefix_context":
                priority_rank = 8
            else:
                priority_rank = 6
            try:
                glyph_index = int(meta.get("current_glyph_index", 999999))
            except (TypeError, ValueError):
                glyph_index = 999999
            try:
                visible_length = int(meta.get("visible_length", 999999))
            except (TypeError, ValueError):
                visible_length = 999999
            cursor_aligned = bool(glyph_index == visible_length and glyph_index != 999999)
            label_rank = 0 if label.startswith("text::") else (1 if label.startswith("text_slot_confirmation::") else 2)
            return (
                0 if current else 1,
                0 if readout else 1,
                0 if cursor_aligned else 1,
                priority_rank,
                label_rank,
                glyph_index,
                label,
            )

        return sorted(clean_rows, key=rank)

    def _text_payload_token(self, row: dict) -> str:
        label = str((row or {}).get("sa_label", "") or "")
        meta = dict((row or {}).get("anchor_meta", {}) or {}) if isinstance((row or {}).get("anchor_meta", {}), dict) else {}
        if label.startswith("text::"):
            return label.split("::", 1)[-1]
        for key in (
            "candidate_text",
            "expected_text",
            "target_token",
            "feedback_expected_token",
            "teacher_reference_token_post_action_only",
            "token",
            "new_text",
            "candidate_token",
            "expected_token",
            "to_token",
        ):
            value = str(meta.get(key, "") or "")
            if value:
                return value
        return ""

    def _text_process_meta_subset(self, meta: dict) -> dict:
        allowed = {
            "schema_id",
            "event_type",
            "current_glyph_index",
            "current_glyph_role",
            "same_tick_binding_role",
            "prediction_payload_priority",
            "process_anchor_role",
            "visible_length",
            "cursor_index",
            "cursor",
            "last_visible_token",
            "operation",
            "conflict_kind",
            "span",
            "support",
            "task_id",
            "paradigm_id",
            "region_id",
            "readout_semantic_role",
            "readout_pattern_id",
            "semantic_frame_role",
            "dynamic_slot_role",
            "slot_role",
            "previous_prefix",
        }
        return {
            key: value
            for key, value in dict(meta or {}).items()
            if key in allowed and value is not None and str(value) != ""
        }

    def _is_negative_feedback_text_payload_item(self, row: dict) -> bool:
        label = str((row or {}).get("sa_label", "") or "")
        family = str((row or {}).get("family", "") or "")
        source_type = str((row or {}).get("source_type", "") or "")
        meta = dict((row or {}).get("anchor_meta", {}) or {}) if isinstance((row or {}).get("anchor_meta", {}), dict) else {}
        if family == "text_revision_opportunity" or label.startswith("text_revision_opportunity::") or source_type == "text_action_feedback":
            return False
        outcome = str(meta.get("feedback_outcome", "") or "")
        if outcome == "punished":
            return True
        try:
            punishment = float(meta.get("feedback_punishment", 0.0) or 0.0)
            reward = float(meta.get("feedback_reward", 0.0) or 0.0)
            correctness = float(meta.get("feedback_correctness", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        return bool(punishment > max(reward, correctness) and punishment >= 0.18)

    def _negative_text_payload_as_revision(self, row: dict) -> dict:
        label = str((row or {}).get("sa_label", "") or "")
        meta = dict((row or {}).get("anchor_meta", {}) or {}) if isinstance((row or {}).get("anchor_meta", {}), dict) else {}
        token = str(meta.get("token", "") or "")
        if not token and label.startswith("text::"):
            token = label.split("::", 1)[-1]
        clean_token = token or "unknown"
        source_feedback_outcome = str(meta.get("feedback_outcome", "") or "punished")
        derived_notes = list(meta.get("notes", []) or []) + ["negative_feedback_text_not_reserved_as_positive_payload"]
        return {
            "sa_label": f"text_revision_opportunity::negative_feedback::{clean_token}",
            "display_text": f"negative_feedback_repair:{clean_token}",
            "family": "text_revision_opportunity",
            "source_type": "text_action_feedback",
            "real_energy": round(max(0.08, float((row or {}).get("real_energy", 0.0) or 0.0) * 0.45), 4),
            "virtual_energy": round(max(0.12, float(meta.get("feedback_punishment", (row or {}).get("virtual_energy", 0.0)) or 0.0)), 4),
            "anchor_meta": {
                **meta,
                "schema_id": "text_revision_opportunity/v1",
                "event_type": "negative_feedback_repair_context",
                "operation": "repair_or_avoid",
                "token": clean_token,
                "feedback_outcome": "repair",
                "source_feedback_outcome": source_feedback_outcome,
                "prediction_payload_role": "negative_feedback_repair_context",
                "source_text_label": label,
                "positive_text_prediction_allowed": False,
                "used_in_strict_teacher_off_input": False,
                "answer_table_lookup": False,
                "notes": derived_notes,
            },
        }

    def _is_current_teacher_text_payload_item(self, row: dict, *, snapshot_tick: int | None) -> bool:
        meta = dict((row or {}).get("anchor_meta", {}) or {}) if isinstance((row or {}).get("anchor_meta", {}), dict) else {}
        if bool(meta.get("used_in_strict_teacher_off_input", False)):
            return False
        current_role = str(meta.get("current_glyph_role", "") or "")
        priority = str(meta.get("prediction_payload_priority", "") or "")
        same_tick_role = str(meta.get("same_tick_binding_role", "") or "")
        is_current_role = (
            bool(meta.get("current_read_tick", False))
            or current_role in {"read_tick_target", "current_glyph_insert_opportunity", "current_glyph_character", "current_slot_confirmed_character"}
            or priority.startswith("current_glyph")
            or same_tick_role in {"current_glyph_character_sa", "current_glyph_insert_opportunity", "current_glyph_slot_confirmation"}
        )
        if not is_current_role:
            return False
        if snapshot_tick is None:
            return True
        try:
            last_seen = int((row or {}).get("last_seen_tick", (row or {}).get("tick_index", snapshot_tick)) or snapshot_tick)
        except (TypeError, ValueError):
            return True
        try:
            return int(last_seen) == int(snapshot_tick)
        except (TypeError, ValueError):
            return True

    def _dedupe_prediction_payload_items(self, rows: list[dict], *, limit: int, exclude_labels: set[str] | None = None) -> list[dict]:
        clean = []
        cap = int(limit)
        if cap <= 0:
            return clean
        seen = {str(label or "") for label in (exclude_labels or set()) if str(label or "")}
        for item in rows or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            clean.append(dict(item))
            if len(clean) >= cap:
                break
        return clean

    def _extract_action_feedback_items(self, items: list[dict], *, limit: int) -> list[dict]:
        rows = []
        cap = max(1, int(limit))
        for item in items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            family = str(item.get("family", "") or "")
            source_type = str(item.get("source_type", "") or "")
            if not label.startswith("action_feedback::") and family != "action_feedback" and source_type != "action_feedback":
                continue
            rows.append(dict(item))
            if len(rows) >= cap:
                break
        return rows

    def _is_external_evidence_item(self, row: dict) -> bool:
        src = str((row or {}).get("source_type", "") or "")
        if src == "external_text":
            return True
        if src == "external_teacher":
            return True
        if src in {"vision_numeric", "audio_numeric"}:
            return True
        if src.startswith("vision_bridge"):
            return True
        if src.startswith("audio_bridge"):
            return True
        return False

    def _is_action_prediction_item(self, row: dict) -> bool:
        label = str((row or {}).get("sa_label", "") or "")
        family = str((row or {}).get("family", "") or "")
        source_type = str((row or {}).get("source_type", "") or "")
        return label.startswith("action::") or family == "action" or source_type == "action_selection"

    def _select_core_items(self, items: list[dict], *, limit: int = 8) -> list[dict]:
        return self._select_anchor_items(items, limit=limit)

    def _select_state_field_items(self, items: list[dict], *, limit: int = 8) -> list[dict]:
        """
        Main Bn recognition view.

        APV2.1 treats every SA as a first-class citizen. This selector therefore
        does not exclude action, feeling, emotion, feedback, or control labels.
        It only keeps the field bounded and orders stronger/current items first,
        so fast-system recall can form humanlike whole-field intuition.
        """

        cap = max(1, int(limit))
        rows = [dict(item) for item in list(items or []) if isinstance(item, dict) and str(item.get("sa_label", "") or "")]
        rows.sort(key=lambda item: self._state_field_sort_key(item))
        return rows[:cap]

    def _select_anchor_items(self, items: list[dict], *, limit: int = 8) -> list[dict]:
        """
        External-anchor/compatibility view.

        This is not the cognition core. It keeps enough text/vision/audio/body
        anchors available for bounded posting, old persistence tables, and human
        previews while `state_field_items` carries the real all-SA recall field.
        """

        cap = max(1, int(limit))
        external: list[dict] = []
        ordinary: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            row = dict(item)
            if self._is_external_anchor_item(row):
                external.append(row)
            elif self._is_core_item(row):
                ordinary.append(row)
        external.sort(key=lambda item: self._state_field_sort_key(item))
        ordinary.sort(key=lambda item: self._state_field_sort_key(item))
        rows = []
        seen = set()
        for item in external + ordinary:
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            rows.append(item)
            if len(rows) >= cap:
                break
        return rows

    def _items_for_cached_labels(self, items: list[dict], labels: list[str], *, fallback) -> list[dict]:
        current_by_label = {str(item.get("sa_label", "") or ""): item for item in items if isinstance(item, dict) and str(item.get("sa_label", "") or "")}
        rows = [dict(current_by_label[label]) for label in labels if label in current_by_label]
        if rows:
            return rows
        return [dict(item) for item in list(fallback() or []) if isinstance(item, dict)]

    def _snapshot_state_field_items(self, snapshot: dict | None) -> list[dict]:
        if not isinstance(snapshot, dict):
            return []
        rows = snapshot.get("state_field_items", None)
        if isinstance(rows, list) and rows:
            return [dict(item) for item in rows if isinstance(item, dict)]
        return self._select_state_field_items(list(snapshot.get("items", []) or []), limit=self.core_item_limit)

    def _state_field_sort_key(self, item: dict) -> tuple:
        energy = self._item_state_field_weight(item)
        try:
            tick_key = int(item.get("last_seen_tick", item.get("tick_index", 0)) or 0)
        except (TypeError, ValueError):
            tick_key = 0
        try:
            position_key = int(item.get("position", 0) or 0)
        except (TypeError, ValueError):
            position_key = 0
        return (-energy, -tick_key, position_key, str(item.get("sa_label", "") or ""))

    def _item_state_field_weight(self, item: dict) -> float:
        real = max(0.0, float(item.get("real_energy", 0.0) or 0.0))
        virtual = max(0.0, float(item.get("virtual_energy", 0.0) or 0.0))
        pressure = max(0.0, float(item.get("cognitive_pressure", real - virtual) or 0.0))
        query = max(0.0, float(item.get("query_weight", 0.0) or 0.0))
        attention = max(0.0, float(item.get("attention_gain", item.get("attention_weight", 0.0)) or 0.0))
        focus_bonus = 0.18 if bool(dict(item.get("anchor_meta", {}) or {}).get("is_focus", False)) else 0.0
        currentness = dict(item.get("query_currentness", {}) or {}) if isinstance(item.get("query_currentness", {}), dict) else {}
        currentness_factor = 1.0
        if bool(currentness.get("new_external_turn_residue_softened", False)):
            currentness_factor = _clamp(float(currentness.get("factor", 1.0) or 1.0), 0.18, 1.0)
        weight = real * 0.95 + virtual * 0.24 + pressure * 0.40 + query * 0.52 + attention * 0.28 + focus_bonus
        return _round4(weight * currentness_factor)

    def _is_external_anchor_item(self, item: dict) -> bool:
        label = str(item.get("sa_label", "") or "")
        source_type = str(item.get("source_type", "") or "")
        family = str(item.get("family", "") or "")
        if source_type == "external_text" or source_type in {"vision_numeric", "audio_numeric"}:
            return True
        if source_type.startswith(("vision_bridge", "audio_bridge")):
            return True
        if label.startswith(("text::", "phrase::", "vision_obj::", "audio_event::", "body::", "scene::")):
            return True
        if family in {"text", "learned_text_phrase", "vision_object", "audio_event", "body_state", "scene"}:
            return True
        return False

    def _select_aux_items(self, items: list[dict], *, limit: int = 2) -> list[dict]:
        cap = max(0, int(limit))
        if cap <= 0:
            return []
        rows = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not self._is_core_item(item):
                rows.append(dict(item))
                if len(rows) >= cap:
                    break
        if len(rows) <= cap:
            return rows
        rows.sort(
            key=lambda item: (
                -(float(item.get("query_weight", item.get("real_energy", 0.0)) or 0.0) + float(item.get("virtual_energy", 0.0) or 0.0) * 0.25),
                str(item.get("sa_label", "") or ""),
            )
        )
        return rows[:cap]

    def _ordered_focus_labels(self, items: list[dict], *, fallback_labels: list[str]) -> list[str]:
        rows = []
        for index, item in enumerate(items or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            anchor_meta = dict(item.get("anchor_meta", {}) or {}) if isinstance(item.get("anchor_meta", {}), dict) else {}
            if not bool(anchor_meta.get("is_focus", False)):
                continue
            try:
                tick_key = int(item.get("last_seen_tick", item.get("tick_index", 0)) or 0)
            except (TypeError, ValueError):
                tick_key = 0
            try:
                position_key = int(item.get("position", index) or index)
            except (TypeError, ValueError):
                position_key = index
            rows.append((tick_key, position_key, index, label))
        if not rows:
            return [label for label in fallback_labels[:64] if str(label or "")]
        rows.sort(key=lambda row: (int(row[0]), int(row[1]), int(row[2]), str(row[3])))
        ordered = []
        seen = set()
        for _, _, _, label in rows:
            if label in seen:
                continue
            seen.add(label)
            ordered.append(label)
        return ordered[:64]

    def _is_core_item(self, item: dict) -> bool:
        label = str(item.get("sa_label", "") or "")
        if label.startswith(self._non_core_label_prefixes):
            return False
        family = str(item.get("family", "") or "")
        if family in self._non_core_families:
            return False
        source_type = str(item.get("source_type", "") or "")
        if source_type in self._non_core_source_types:
            return False
        return True
