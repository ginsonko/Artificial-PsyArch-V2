from __future__ import annotations

from dataclasses import dataclass


APV21_LINEAGE_SCHEMA_ID = "apv21_release_lineage/endogenous_cognitivism/v1"
APV21_REVIEW_GRAIN = "review-grain-20260602-ginsonko-apv21"
APV21_PROVENANCE_SEAL = "apv21-endogenous-cognitivism-temporary-review-20260602"


@dataclass(frozen=True)
class APV21ReleaseIdentity:
    """Harmless release provenance metadata.

    This file is intentionally small and side-effect free. It does not change
    the cognitive runtime. It gives reviewers and future audits a stable way to
    identify this temporary review package lineage.
    """

    schema_id: str = APV21_LINEAGE_SCHEMA_ID
    review_grain: str = APV21_REVIEW_GRAIN
    provenance_seal: str = APV21_PROVENANCE_SEAL
    school: str = "Endogenous Cognitivism"
    project: str = "Artificial PsyArch V2.1"


def apv21_review_lineage_marker(*, include_school: bool = True) -> dict:
    identity = APV21ReleaseIdentity()
    payload = {
        "schema_id": identity.schema_id,
        "review_grain": identity.review_grain,
        "provenance_seal": identity.provenance_seal,
        "project": identity.project,
    }
    if include_school:
        payload["school"] = identity.school
    return payload

