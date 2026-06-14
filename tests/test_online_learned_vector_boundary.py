from __future__ import annotations

from memory.embedding.online_store import OnlineEmbeddingStore
from memory.persistence import RecordingMemoryPersistence, SqliteMemoryPersistence, SqlitePersistenceConfig
from memory.store import MemoryStore


def _item(label: str, *, real: float = 1.0, virtual: float = 0.0, pressure: float | None = None) -> dict:
    return {
        "sa_label": label,
        "display_text": label.split("::")[-1],
        "family": label.split("::", 1)[0] if "::" in label else "text",
        "source_type": "test_probe",
        "real_energy": float(real),
        "virtual_energy": float(virtual),
        "cognitive_pressure": float(real - virtual if pressure is None else pressure),
    }


def test_online_learned_vector_moves_with_pressure_but_transition_stays_directed() -> None:
    store = OnlineEmbeddingStore(dim=32, token_limit=128, min_support_to_promote=1, per_tick_update_limit=32)

    positive_before = store.learned_vector_similarity(["text::apple"], ["vision::round"])["score"]
    store.begin_tick(0)
    store.observe_positive_pair("text::apple", "vision::round", weight=2.0)
    positive_after = store.learned_vector_similarity(["text::apple"], ["vision::round"])["score"]

    store.begin_tick(1)
    store.observe_negative_pair("text::apple", "vision::round", weight=3.0)
    negative_after = store.learned_vector_similarity(["text::apple"], ["vision::round"])["score"]

    transition_before = store.learned_vector_similarity(["text::cue"], ["text::next"])["score"]
    store.begin_tick(2)
    store.observe_transition_pair("text::cue", "text::next", weight=3.0)
    transition_after = store.learned_vector_similarity(["text::cue"], ["text::next"])["score"]
    transition_score = store.learned_transition(["text::cue"], ["text::next"])["score"]

    assert positive_after > positive_before
    assert negative_after < positive_after
    assert transition_after == transition_before
    assert transition_score > 0.0


def test_memory_store_persists_hash_and_online_learned_vector_spaces() -> None:
    recorder = RecordingMemoryPersistence()
    memory = MemoryStore(
        recall_top_k=4,
        predict_top_k=3,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=16,
        candidate_limit=8,
        scoring_candidate_limit=8,
        learned_rerank_limit=8,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        persistence=recorder,
    )

    first = memory.write_snapshot(
        tick_index=0,
        memory_kind="state",
        items=[
            _item("text::math_problem", real=1.2, virtual=0.1, pressure=1.1),
            _item("text::vertical_addition", real=1.0, virtual=0.1, pressure=0.0),
            _item("text::carry_step", real=0.9, virtual=0.1, pressure=0.0),
        ],
        focus_labels=["text::math_problem", "text::vertical_addition"],
        source_text="math problem with vertical addition",
    )
    second = memory.write_snapshot(
        tick_index=1,
        memory_kind="state",
        items=[
            _item("text::math_problem", real=1.1, virtual=0.1, pressure=0.9),
            _item("text::vertical_addition", real=1.0, virtual=0.1, pressure=0.0),
            _item("text::carry_step", real=0.95, virtual=0.1, pressure=0.0),
            _item("text::wrong_story_residue", real=0.0, virtual=1.0, pressure=-1.0),
        ],
        focus_labels=["text::math_problem", "text::carry_step"],
        source_text="similar math problem with carry step",
    )
    memory.process_pending_index_jobs(budget=8, max_ms=50.0, include_heavy=True)

    for snapshot in (first, second):
        spaces = snapshot["vector_spaces"]
        assert set(spaces) == {"hash_vector", "online_learned_vector"}
        assert len(spaces["hash_vector"]) == len(spaces["online_learned_vector"])
        assert any(abs(value) > 1e-9 for value in spaces["online_learned_vector"])

    assert len(recorder.writes) == 2
    assert all({"hash_vector", "online_learned_vector"} <= set(write["vector_spaces"]) for write in recorder.writes)

    audit = memory.audit_recall(
        [_item("text::new_math_problem", real=1.2), _item("text::carry_step", real=1.0)],
        memory_kind="state",
        top_k=4,
        exact_limit=4,
    )
    rows = list(audit.get("exact_rows", []) or [])
    assert rows
    assert any(float(row.get("learned_vector_score", 0.0) or 0.0) > 0.0 for row in rows)
    assert memory.learned_similarity(["text::math_problem"], ["text::vertical_addition"])["score"] != 0.0


def test_learned_vector_candidate_weight_does_not_change_exact_audit_score() -> None:
    query = [_item("text::new_math_problem", real=1.2), _item("text::carry_step", real=1.0)]

    low = MemoryStore(
        recall_top_k=4,
        predict_top_k=3,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=16,
        candidate_limit=8,
        scoring_candidate_limit=8,
        learned_rerank_limit=8,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        learned_vector_candidate_weight=0.0,
    )
    high = MemoryStore(
        recall_top_k=4,
        predict_top_k=3,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=16,
        candidate_limit=8,
        scoring_candidate_limit=8,
        learned_rerank_limit=8,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        learned_vector_candidate_weight=12.0,
    )

    for memory in (low, high):
        memory.write_snapshot(
            tick_index=0,
            memory_kind="state",
            items=[
                _item("text::math_problem", real=1.2, virtual=0.1, pressure=1.1),
                _item("text::vertical_addition", real=1.0, virtual=0.1, pressure=0.0),
                _item("text::carry_step", real=0.9, virtual=0.1, pressure=0.0),
            ],
            focus_labels=["text::math_problem", "text::vertical_addition"],
            source_text="math problem with vertical addition",
        )
        memory.write_snapshot(
            tick_index=1,
            memory_kind="state",
            items=[
                _item("text::math_problem", real=1.1, virtual=0.1, pressure=0.9),
                _item("text::vertical_addition", real=1.0, virtual=0.1, pressure=0.0),
                _item("text::carry_step", real=0.95, virtual=0.1, pressure=0.0),
                _item("text::wrong_story_residue", real=0.0, virtual=1.0, pressure=-1.0),
            ],
            focus_labels=["text::math_problem", "text::carry_step"],
            source_text="similar math problem with carry step",
        )
        memory.process_pending_index_jobs(budget=8, max_ms=50.0, include_heavy=True)

    low_rows = list(low.audit_recall(query, memory_kind="state", top_k=4, exact_limit=4).get("exact_rows", []) or [])
    high_rows = list(high.audit_recall(query, memory_kind="state", top_k=4, exact_limit=4).get("exact_rows", []) or [])
    low_exact = next(row for row in low_rows if str(row.get("memory_id", "") or "") == "mem-2")
    high_exact = next(row for row in high_rows if str(row.get("memory_id", "") or "") == "mem-2")

    assert float(low_exact.get("score", 0.0) or 0.0) == float(high_exact.get("score", 0.0) or 0.0)
    assert float(low_exact.get("learned_vector_score", 0.0) or 0.0) == float(high_exact.get("learned_vector_score", 0.0) or 0.0)


def test_runtime_state_roundtrip_restores_online_similarity(tmp_path) -> None:
    db_path = tmp_path / "runtime_state.sqlite3"
    persistence = SqliteMemoryPersistence(
        SqlitePersistenceConfig(
            path=db_path,
            run_id="runtime-state-run",
            resident_hot_snapshots_per_kind=8,
            warm_prefetch_limit=8,
        )
    )
    writer = MemoryStore(
        recall_top_k=4,
        predict_top_k=3,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=16,
        candidate_limit=8,
        scoring_candidate_limit=8,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
        persistence=persistence,
    )
    writer.write_snapshot(
        tick_index=0,
        memory_kind="state",
        items=[
            _item("text::math_problem", real=1.2, virtual=0.1, pressure=1.1),
            _item("text::vertical_addition", real=1.0, virtual=0.1, pressure=0.0),
        ],
        focus_labels=["text::math_problem"],
        source_text="math problem with vertical addition",
    )
    writer.write_snapshot(
        tick_index=1,
        memory_kind="state",
        items=[
            _item("text::math_problem", real=1.1, virtual=0.1, pressure=0.9),
            _item("text::carry_step", real=0.95, virtual=0.1, pressure=0.0),
        ],
        focus_labels=["text::math_problem", "text::carry_step"],
        source_text="math problem with carry step",
    )
    persistence.flush()
    before = writer.learned_similarity(["text::math_problem"], ["text::vertical_addition"])["score"]
    runtime_state = persistence.load_runtime_state()
    assert runtime_state is not None
    assert runtime_state.get("entries")

    reader = MemoryStore(
        recall_top_k=4,
        predict_top_k=3,
        prediction_energy_scale=0.55,
        max_snapshots_per_kind=16,
        candidate_limit=8,
        scoring_candidate_limit=8,
        online_min_support_to_promote=1,
        online_per_tick_update_limit=64,
    )
    reader._persistence = SqliteMemoryPersistence(
        SqlitePersistenceConfig(
            path=db_path,
            run_id="runtime-state-run",
            resident_hot_snapshots_per_kind=8,
            warm_prefetch_limit=8,
        )
    )
    warm = reader.warm_load_from_persistence(memory_kind="state", limit_per_kind=8)
    after = reader.learned_similarity(["text::math_problem"], ["text::vertical_addition"])["score"]

    assert warm["runtime_state"]["restored"] is True
    assert after == before
