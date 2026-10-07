# Agent instructions for GEMS-CONTEST

Primary objective: maximize reproducible competition performance while staying within the official GEMS rules and data licenses.

## Svyable's operating goal (2026-10-07)

Advance this repository toward winning the GEMS Prize as Svyable. Prioritize
measured unknown-fault discovery, robust held-out performance, and reproducible
finalist assets over PR volume or speculative claims. Use recent OpenAI mathematical
research as a source of candidate ideas and verification methods; require an
explicit connection to GEMS inputs/objective before adopting a theorem or algorithm.
Track that connection and its verification status in `docs/MATH_RESEARCH.md`.

At each development run, inspect existing work, choose the highest-value remaining
gap, and complete a tested change where inputs permit. Distinguish measured gains
from infrastructure, fit scores, hypotheses, and unverified research. Report what
changed, evidence, prize relevance, the remaining blocker, and the next experiment.
Do not initiate paid compute or API use without an authorized budget.

- Never commit raw competition data, external source tiles, checkpoints, or submission GeoTIFFs.
- Preserve spatially honest validation; do not introduce random-pixel leakage.
- Implement objective tests for metrics, geospatial transforms, masks and submission format.
- Every experiment must be reproducible from a commit + immutable config + data fingerprint.
- Keep `docs/AI_DISCLOSURE.md` current for material generative-AI use.
- Keep `docs/EXTERNAL_DATA_LEDGER.md` current before external data is used.
- Prefer mergeable, CI-gated changes over prose-only suggestions.
- Do not add human-review blockers as a workflow requirement; rely on tests/schema/CI when possible.
- Treat leaderboard results as evidence, not ground truth. Do not optimize via repeated public-LB probing.
