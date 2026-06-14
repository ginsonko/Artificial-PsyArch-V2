# APV2.1 Temporary Review Release Manifest

Release date: 2026-06-02

Repository target: `ginsonko/Artificial-PsyArch-V2.1`

## Included

- Core APV2.1 source:
  - `channels/`
  - `config/`
  - `core/`
  - `education/`
  - `memory/`
  - `observatory/`
  - `sensors/`
  - `scripts/`
  - `tests/`
  - `data/`

- Compatibility/reference layer:
  - selected `legacy_apv2/` source, sensors, observatory, schemas, scripts, and evidence.
  - legacy generated outputs are excluded.

- Experiment sources:
  - all top-level `experiments/*.py` files.

- Selected generated reports:
  - public showcase HTML/PNG/MD reports,
  - LongRun dashboards,
  - P1-J attention reports,
  - P1-L multimodal/teacher-off/uncertainty reports,
  - Math-0 through Math-20 reports and skill packages.

- Theory and review docs:
  - `README.md`
  - `APV21_FULL_PROJECT_GUIDE_FOR_AI.md`
  - `EDUCATION_PROTOCOL.md`
  - `EXPERIMENT_INDEX.md`
  - `SKILL_PACKAGES.md`
  - `SECURITY_AND_PRIVACY.md`
  - `NOTICE.md`
  - `LICENSE`
  - selected historical cold-save docs under `docs/`.

- Provenance metadata:
  - `core/release_identity.py`
  - release verification in `scripts/verify_release_package.py`.

## Excluded

- Python bytecode and cache folders.
- `.pytest_cache` and other local caches.
- Local logs.
- Temporary probe folders.
- Full `outputs/` directories.
- Very large raw tick traces over normal GitHub file-size limits.
- Real API keys or private API endpoint configs.
- Legacy generated long multimodal datasets that are not required for current review.

## Release Identity

This release package contains harmless provenance metadata:

```text
apv21_release_lineage/endogenous_cognitivism/v1
review-grain-20260602-ginsonko-apv21
apv21-endogenous-cognitivism-temporary-review-20260602
```

These identifiers do not affect cognitive runtime behavior. They preserve authorship continuity for review and later source-audit comparison.

## Recommended Validation

```powershell
python scripts\verify_release_package.py
python scripts\run_math20_two_digit_divisor_random_retention.py
python -m pytest -q tests\test_math20_two_digit_divisor_random_retention.py
```

Broader math regression:

```powershell
python -m pytest -q tests\test_math0_digit_quantity_sequence_blocks.py tests\test_math1_single_digit_add_sub_persistence.py tests\test_math2_random_feedback_revision_accumulation.py tests\test_math3_ten_within_random_retention.py tests\test_math4_carry_borrow_process_bridge.py tests\test_math5_vertical_two_digit_add_sub.py tests\test_math6_two_digit_vertical_random_retention.py tests\test_math7_multi_digit_vertical_add_sub.py tests\test_math8_multi_digit_random_retention.py tests\test_math9_multiplication_bricks.py tests\test_math10_two_digit_vertical_multiplication.py tests\test_math11_two_digit_multiplication_random_retention.py tests\test_math12_three_by_two_vertical_multiplication.py tests\test_math13_three_by_two_random_retention.py tests\test_math14_division_prebricks.py tests\test_math15_one_digit_vertical_division.py tests\test_math16_one_digit_division_random_retention.py tests\test_math17_zero_quotient_long_division.py tests\test_math18_two_digit_divisor_trial_adjust.py tests\test_math19_two_digit_divisor_vertical_division.py tests\test_math20_two_digit_divisor_random_retention.py
```

## Known Release Caveats

- Some older Chinese cold-save docs may reflect earlier checkpoints. Prefer this manifest, README, and full AI guide for the current public-facing explanation.
- Large raw traces are intentionally omitted. Regenerate them locally when needed.
- This is a temporary review package, not a polished product release.

