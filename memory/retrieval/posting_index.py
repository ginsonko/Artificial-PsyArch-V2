from __future__ import annotations

from collections import defaultdict
from math import exp


def _round4(value: float) -> float:
    return round(float(value), 4)


class PostingIndex:
    def __init__(self) -> None:
        self._postings: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
            lambda: {
                "label": defaultdict(set),
                "display": defaultdict(set),
                "bigram": defaultdict(set),
                "focus": defaultdict(set),
                "sequence": defaultdict(set),
            }
        )
        self._ids_by_kind: dict[str, set[str]] = defaultdict(set)
        # Reverse index to support bounded eviction without scanning postings.
        # kind -> memory_id -> field -> set(tokens)
        self._tokens_by_id: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(lambda: defaultdict(dict))
        self.matched_token_preview_limit = 12

    def add(
        self,
        memory_kind: str,
        memory_id: str,
        *,
        label_tokens: list[str],
        display_tokens: list[str],
        bigram_tokens: list[str],
        focus_tokens: list[str],
        sequence_tokens: list[str] | None = None,
    ) -> None:
        kind = str(memory_kind or "")
        clean_id = str(memory_id or "")
        if not kind or not clean_id:
            return
        self._ids_by_kind[kind].add(clean_id)
        self._add_tokens(kind, "label", label_tokens, clean_id)
        self._add_tokens(kind, "display", display_tokens, clean_id)
        self._add_tokens(kind, "bigram", bigram_tokens, clean_id)
        self._add_tokens(kind, "focus", focus_tokens, clean_id)
        self._add_tokens(kind, "sequence", list(sequence_tokens or []), clean_id)

    def candidates(
        self,
        memory_kind: str,
        *,
        label_tokens: list[str],
        display_tokens: list[str],
        bigram_tokens: list[str],
        focus_tokens: list[str],
        sequence_tokens: list[str] | None = None,
        limit: int,
    ) -> list[dict]:
        kind = str(memory_kind or "")
        if not kind:
            return []
        rows: dict[str, dict] = {}
        cap = max(1, int(limit))
        stop_rows = max(cap * 3, cap + 16)
        self._accumulate(rows, kind, "sequence", list(sequence_tokens or []), 1.05, stop_rows=stop_rows, prefer_low_frequency=True)
        self._accumulate(rows, kind, "focus", focus_tokens, 0.7, stop_rows=stop_rows, prefer_low_frequency=True)
        self._accumulate(rows, kind, "bigram", bigram_tokens, 0.9, stop_rows=stop_rows, prefer_low_frequency=True)
        self._accumulate(rows, kind, "label", label_tokens, 1.0, stop_rows=stop_rows, prefer_low_frequency=True)
        self._accumulate(rows, kind, "display", display_tokens, 0.55, stop_rows=stop_rows, prefer_low_frequency=True)
        ordered = list(rows.values())
        ordered.sort(
            key=lambda item: (
                -float(item.get("posting_score", 0.0) or 0.0),
                -int(item.get("total_matches", 0) or 0),
                str(item.get("memory_id", "") or ""),
            )
        )
        result = ordered[:cap]
        for row in result:
            row["posting_score"] = _round4(float(row.get("posting_score", 0.0) or 0.0))
        return result

    def ids_for_kind(self, memory_kind: str) -> list[str]:
        return sorted(self._ids_by_kind.get(str(memory_kind or ""), set()))

    def remove(self, memory_kind: str, memory_id: str) -> None:
        kind = str(memory_kind or "")
        clean_id = str(memory_id or "")
        if not kind or not clean_id:
            return
        self._ids_by_kind.get(kind, set()).discard(clean_id)
        token_entry = self._tokens_by_id.get(kind, {}).pop(clean_id, None)
        if not isinstance(token_entry, dict):
            return
        for field_name, tokens in token_entry.items():
            if field_name not in {"label", "display", "bigram", "focus", "sequence"}:
                continue
            posting_bucket = self._postings[kind][field_name]
            for token in set(tokens or set()):
                if not token:
                    continue
                id_set = posting_bucket.get(token)
                if not id_set:
                    continue
                id_set.discard(clean_id)
                if not id_set:
                    posting_bucket.pop(token, None)

    def _add_tokens(self, memory_kind: str, field_name: str, tokens: list[str], memory_id: str) -> None:
        posting_bucket = self._postings[memory_kind][field_name]
        reverse_bucket = self._tokens_by_id[memory_kind].setdefault(memory_id, {}).setdefault(field_name, set())
        seen: set[str] = set()
        for token in tokens:
            clean = str(token or "").strip()
            if not clean or clean in seen:
                continue
            posting_bucket[clean].add(memory_id)
            reverse_bucket.add(clean)
            seen.add(clean)

    def _accumulate(
        self,
        rows: dict[str, dict],
        memory_kind: str,
        field_name: str,
        tokens: list[str],
        field_weight: float,
        *,
        stop_rows: int | None = None,
        prefer_low_frequency: bool = False,
    ) -> None:
        posting_bucket = self._postings[memory_kind][field_name]
        seen: set[str] = set()
        clean_tokens = []
        for token in tokens:
            clean = str(token or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            id_set = posting_bucket.get(clean, set())
            if not id_set:
                continue
            clean_tokens.append((clean, len(id_set), id_set))
        if prefer_low_frequency:
            clean_tokens.sort(key=lambda item: (int(item[1]), str(item[0])))
        source_tag = f"posting:{field_name}"
        total_ids = max(1, len(self._ids_by_kind.get(memory_kind, set())))
        for clean, frequency, id_set in clean_tokens:
            specificity = self._posting_specificity(total_ids=total_ids, token_frequency=int(frequency))
            contribution = float(field_weight) * float(specificity)
            for memory_id in id_set:
                row = rows.setdefault(
                    memory_id,
                    {
                        "memory_id": memory_id,
                        "posting_score": 0.0,
                        "posting_specificity_score": 0.0,
                        "total_matches": 0,
                        "match_counts": {"label": 0, "display": 0, "bigram": 0, "focus": 0, "sequence": 0},
                        "matched_tokens": {"label": [], "display": [], "bigram": [], "focus": [], "sequence": []},
                        "matched_token_weights": {"label": [], "display": [], "bigram": [], "focus": [], "sequence": []},
                        "candidate_sources": [],
                    },
                )
                row["posting_score"] = float(row["posting_score"]) + contribution
                row["posting_specificity_score"] = float(row["posting_specificity_score"]) + contribution
                row["total_matches"] = int(row["total_matches"]) + 1
                row["match_counts"][field_name] = int(row["match_counts"][field_name]) + 1
                if len(row["matched_tokens"][field_name]) < self.matched_token_preview_limit:
                    row["matched_tokens"][field_name].append(clean)
                    row["matched_token_weights"][field_name].append(
                        {
                            "token": clean,
                            "frequency": int(frequency),
                            "specificity": _round4(specificity),
                            "contribution": _round4(contribution),
                        }
                    )
                if source_tag not in row["candidate_sources"]:
                    row["candidate_sources"].append(source_tag)
            if stop_rows is not None and len(rows) >= int(stop_rows):
                break

    def _posting_specificity(self, *, total_ids: int, token_frequency: int) -> float:
        total = max(1, int(total_ids))
        frequency = max(1, int(token_frequency))
        # Common tokens are still useful anchors, but AP-style recall should feel
        # their repetition as adapted background. Rare/current process tokens
        # keep more drive without becoming a keyword route.
        rarity = exp(-float(frequency - 1) / max(1.0, float(total) * 0.18))
        return max(0.22, min(1.6, 0.22 + rarity * 1.38))
