# Security And Privacy Notes

## Release Scope

This package was prepared from a local APV2.1 prototype checkout. The release copy intentionally excludes transient outputs, cache folders, Python bytecode, local logs, temporary probes, and very large raw tick traces.

Included materials are intended for review:

- source code,
- tests,
- core design documents,
- selected generated HTML/PNG/Markdown experiment reports,
- selected JSON evidence files under GitHub-size limits,
- and skill experience packages that are useful for inspection.

## API Keys

No real API key is intentionally included in this release.

The optional LLM teacher reads configuration from environment variables:

```powershell
$env:APV21_LLM_TEACHER_BASE_URL="https://api.example.invalid/v1"
$env:APV21_LLM_TEACHER_API_KEY="redacted-local-review-key"
$env:APV21_LLM_TEACHER_MODEL="model-name"
```

Do not commit real keys. Do not paste real keys into docs or generated artifacts.

## Test Secret Strings

Some tests simulate redaction behavior. In this release package, simulated values should not use real-looking `sk-` strings. They are intentionally replaced with `redacted-test-secret-*` style values.

In the active local development tree, some tests may still contain fake `sk-*` fixtures to prove redaction behavior. Treat those strings as test-only fixtures. They must not appear in cleaned review packages or reviewer-facing generated artifacts.

## Local-Only Review

Run the package locally. Do not connect it to real user accounts, money-bearing accounts, sensitive desktop applications, or private cameras/microphones without a separate safety review.

## Local Readiness Verification

Run this during local development:

```powershell
python scripts\verify_local_review_readiness.py
```

This lighter checker allows expected local caches and fake test fixtures while still checking reviewer-facing docs, required reports, private endpoint markers, and real-looking non-test API keys.

## Release Verification

Run this before sharing a cleaned review package:

```powershell
python scripts\verify_release_package.py
```

The verifier checks for common release mistakes:

- real-looking API keys,
- accidental temporary folders,
- Python bytecode/cache files,
- oversized files that GitHub rejects,
- and release identity metadata.
