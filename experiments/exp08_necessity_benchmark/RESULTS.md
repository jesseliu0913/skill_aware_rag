# Exp08 — Necessity benchmark: results

**Status:** Not yet run. 🔴 new-data — the benchmark is not yet curated or
validated (see `SPEC.md` §5, `README.md` "Required next steps"). Gold answers
emitted by the scaffold are PROVISIONAL/UNVALIDATED; no numbers may be reported
until human validation is complete.

## Benchmark construction summary
_Not yet run._ (Fill from `candidates/build_stats.json` after building + validating:
#collision aliases, #candidates, #discriminative, #validated, per-source counts,
mean surface-ambiguity score.)

## Per-arm results

| Arm | token-F1 | exact | source-selection acc (incl. source A) | notes |
|---|---:|---:|---:|---|
| uniform RAG (bm25/hybrid) | — | — | — | Not yet run. |
| dense RAG (dense_bge/rrf) | — | — | — | Not yet run. |
| oracle-source RAG (upper bound) | — | — | — | Not yet run. |
| wrong-skill RAG (negative control) | — | — | — | Not yet run. |
| SkillRAG (ours) | — | — | — | Not yet run. |

**Expected ordering (hypothesis, SPEC.md §3):**
`wrong-skill < uniform ≈ dense < SkillRAG ≤ oracle-source`, with SkillRAG closest
to the oracle-source bound without gold evidence or a gold source label.

## Findings
_Not yet run._
