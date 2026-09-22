# Agent instructions for GEMS-CONTEST

Primary objective: maximize reproducible competition performance while staying within the official GEMS rules and data licenses.

- Never commit raw competition data, external source tiles, checkpoints, or submission GeoTIFFs.
- Preserve spatially honest validation; do not introduce random-pixel leakage.
- Implement objective tests for metrics, geospatial transforms, masks and submission format.
- Every experiment must be reproducible from a commit + immutable config + data fingerprint.
- Keep `docs/AI_DISCLOSURE.md` current for material generative-AI use.
- Keep `docs/EXTERNAL_DATA_LEDGER.md` current before external data is used.
- Prefer mergeable, CI-gated changes over prose-only suggestions.
- Do not add human-review blockers as a workflow requirement; rely on tests/schema/CI when possible.
- Treat leaderboard results as evidence, not ground truth. Do not optimize via repeated public-LB probing.
