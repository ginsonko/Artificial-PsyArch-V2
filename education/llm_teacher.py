from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from education.intervention import normalize_education_intervention


DEFAULT_SYSTEM_PROMPT = """You are an external APV2.1 teacher, not AP's core mind.
AP is like a very young learner: it only knows what has been demonstrated as building blocks.
Give soft teaching interventions only: state_items, action_biases with parameters, and feedback.
Do not directly complete the final task for AP. Encourage reusable process, reread, compare, revise,
and reasonable new combinations built from taught blocks. Output strict JSON only."""

DEFAULT_POSTHOC_JUDGE_PROMPT = """You are an external APV2.1 post-output judge, not AP's core mind.
AP has already committed its draft before you see it. Judge only the committed output.
Return strict JSON with grade, feedback, and rationale. Do not output action_biases, state_items,
text_insert params, text_replace params, or any next-answer suggestion."""


@dataclass
class LLMTeacherConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen3-32b"
    timeout_seconds: float = 20.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    posthoc_judge_prompt: str = DEFAULT_POSTHOC_JUDGE_PROMPT

    @classmethod
    def from_env(cls) -> "LLMTeacherConfig":
        return cls(
            base_url=os.environ.get("APV21_LLM_TEACHER_BASE_URL", "").strip(),
            api_key=os.environ.get("APV21_LLM_TEACHER_API_KEY", "").strip(),
            model=os.environ.get("APV21_LLM_TEACHER_MODEL", "qwen3-32b").strip() or "qwen3-32b",
            timeout_seconds=float(os.environ.get("APV21_LLM_TEACHER_TIMEOUT", "20") or 20),
        )

    def public_trace(self) -> dict:
        return {
            "schema_id": "llm_teacher_config_trace/v1",
            "base_url_configured": bool(self.base_url),
            "api_key_configured": bool(self.api_key),
            "model": self.model,
            "timeout_seconds": float(self.timeout_seconds),
            "key_storage_policy": "env_only_never_write_to_repo_or_artifacts",
        }


class LLMTeacherClient:
    """OpenAI-compatible external teacher adapter.

    The adapter is intentionally thin and optional. It is suitable for qwen,
    gpt-mini, a local proxy, or a fake test client. It never stores secrets in
    payloads; traces expose only whether a key was configured.
    """

    def __init__(self, config: LLMTeacherConfig | None = None) -> None:
        self.config = config or LLMTeacherConfig.from_env()

    def build_messages(self, *, goal: str, taught_blocks: list[dict], ap_trace_summary: dict, allowed_actions: list[str]) -> list[dict]:
        user_payload = {
            "goal": str(goal or ""),
            "taught_blocks": list(taught_blocks or []),
            "ap_trace_summary": dict(ap_trace_summary or {}),
            "allowed_actions": list(allowed_actions or []),
            "output_schema": {
                "schema_id": "education_intervention/v1",
                "source": "llm_teacher",
                "teacher_kind": "llm",
                "goal": "string",
                "state_items": "list of AP state items",
                "action_biases": "list of {action_id, drive_delta, params, notes}",
                "feedback": "{reward, punishment, correctness, confidence}",
                "notes": "list of short explanations",
            },
            "hard_rules": [
                "Do not output secrets.",
                "Do not directly replace AP core cognition.",
                "Use only allowed action_id values.",
                "Prefer process hints over final answers.",
                "Reward reasonable generalization from taught blocks.",
            ],
        }
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def build_posthoc_judge_messages(
        self,
        *,
        goal: str,
        taught_blocks: list[dict],
        target_text: str,
        committed_text: str,
        near_variants: list[str] | None = None,
        trace_summary: dict | None = None,
    ) -> list[dict]:
        """Build the strict post-output judge prompt.

        The payload intentionally contains no allowed actions. This keeps the
        LLM in the teacher/judge role: it can grade what AP already committed,
        but it cannot steer the next tick or write text back into AP.
        """

        user_payload = {
            "goal": str(goal or ""),
            "taught_blocks": list(taught_blocks or []),
            "target_text": str(target_text or ""),
            "committed_text": str(committed_text or ""),
            "near_variants": [str(row) for row in list(near_variants or []) if str(row)],
            "trace_summary": dict(trace_summary or {}),
            "output_schema": {
                "schema_id": "llm_posthoc_judge/v1",
                "grade": "one of exact, near, wrong",
                "feedback": "{reward, punishment, correctness, confidence}",
                "rationale": "short explanation after the answer already exists",
            },
            "hard_rules": [
                "AP has already committed the answer before you judge.",
                "Do not output action_biases.",
                "Do not output state_items.",
                "Do not output params for text_insert/text_replace/text_commit.",
                "Do not suggest the next answer.",
                "Grade exact only for identical target text.",
                "Grade near for meaning-preserving word-order or particle differences.",
                "Grade wrong for wrong color/object/position binding.",
            ],
        }
        return [
            {"role": "system", "content": self.config.posthoc_judge_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def suggest_intervention(
        self,
        *,
        goal: str,
        taught_blocks: list[dict],
        ap_trace_summary: dict,
        allowed_actions: list[str],
        tick_index: int | None = None,
    ) -> dict:
        raw_text = self._chat_completion(
            self.build_messages(
                goal=goal,
                taught_blocks=taught_blocks,
                ap_trace_summary=ap_trace_summary,
                allowed_actions=allowed_actions,
            )
        )
        parsed = self._parse_json_object(raw_text)
        parsed.setdefault("source", "llm_teacher")
        parsed.setdefault("teacher_kind", "llm")
        parsed.setdefault("goal", str(goal or ""))
        normalized = normalize_education_intervention(parsed, tick_index=tick_index)
        normalized["llm_teacher"] = {
            **self.config.public_trace(),
            "raw_response_chars": len(raw_text),
            "parse_ok": bool(parsed),
        }
        return normalized

    def judge_post_output(
        self,
        *,
        goal: str,
        taught_blocks: list[dict],
        target_text: str,
        committed_text: str,
        near_variants: list[str] | None = None,
        trace_summary: dict | None = None,
    ) -> dict:
        """Ask a configured LLM to grade an already committed AP output.

        Only judge fields are returned. If an LLM replies with intervention
        fields, they are counted and discarded so the caller cannot accidentally
        turn a judge response into an AP action.
        """

        raw_text = self._chat_completion(
            self.build_posthoc_judge_messages(
                goal=goal,
                taught_blocks=taught_blocks,
                target_text=target_text,
                committed_text=committed_text,
                near_variants=near_variants,
                trace_summary=trace_summary,
            )
        )
        parsed = self._parse_json_object(raw_text)
        sanitized = self._sanitize_posthoc_judge(parsed)
        sanitized["llm_teacher"] = {
            **self.config.public_trace(),
            "raw_response_chars": len(raw_text),
            "parse_ok": bool(parsed),
        }
        return sanitized

    def _sanitize_posthoc_judge(self, parsed: dict) -> dict:
        forbidden = [key for key in ("action_biases", "state_items", "params", "allowed_actions") if key in dict(parsed or {})]
        grade = str((parsed or {}).get("grade", "") or "").strip().lower()
        if grade not in {"exact", "near", "wrong"}:
            grade = "wrong"
        feedback = dict((parsed or {}).get("feedback", {}) or {})
        if grade == "exact":
            defaults = {"reward": 0.68, "punishment": 0.0, "correctness": 0.80, "confidence": 0.70}
        elif grade == "near":
            defaults = {"reward": 0.36, "punishment": 0.02, "correctness": 0.52, "confidence": 0.66}
        else:
            defaults = {"reward": 0.0, "punishment": 0.30, "correctness": 0.08, "confidence": 0.66}
        for key, value in defaults.items():
            try:
                numeric = float(feedback.get(key, value))
            except (TypeError, ValueError):
                numeric = float(value)
            feedback[key] = round(max(0.0, min(1.0, numeric)), 4)
        return {
            "schema_id": "llm_posthoc_judge/v1",
            "available": True,
            "judge_kind": "live_llm",
            "grade": grade,
            "feedback": feedback,
            "rationale": str((parsed or {}).get("rationale", (parsed or {}).get("reason", "")) or ""),
            "forbidden_fields_discarded": forbidden,
            "meaning": "post-output judge only; no action or answer fields are returned to AP",
        }

    def _chat_completion(self, messages: list[dict]) -> str:
        if not self.config.base_url or not self.config.api_key:
            raise RuntimeError("LLM teacher is not configured; set APV21_LLM_TEACHER_BASE_URL and APV21_LLM_TEACHER_API_KEY.")
        base = self.config.base_url.rstrip("/")
        url = f"{base}/v1/chat/completions"
        body = json.dumps(
            {
                "model": self.config.model,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.timeout_seconds)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM teacher HTTP error {exc.code}: {detail}") from exc
        content = (
            ((payload.get("choices", [{}]) or [{}])[0].get("message", {}) or {}).get("content", "")
            if isinstance(payload, dict)
            else ""
        )
        return str(content or "")

    def _parse_json_object(self, text: str) -> dict:
        clean = str(text or "").strip()
        if not clean:
            return {}
        try:
            parsed = json.loads(clean)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            start = clean.find("{")
            end = clean.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(clean[start : end + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
        return {}
