# Exp08 — Necessity benchmark (source-ambiguous, retrieval-sensitive)

**Serves:** the *necessity* argument (storyline §10). **Paper:** §5.5 core.
**Status:** 🔴 **new-data** — net-new curation. Neither FDARxBench nor
Mol-Instructions satisfies the ambiguity criterion, so this benchmark must be
built and **human-validated before any run**. This branch ships the *design* and
the *mechanical construction scaffold* only.

## Claim under test
Choosing the right knowledge *source* is causally load-bearing: on items where
the query surface is ambiguous across ≥2 KB sources and the answer is not in the
input, a skill-routed retriever picks the appropriate source more often than
uniform RAG, closing most of the gap to an oracle-source upper bound — **without
gold evidence or a gold source label**.

See **`SPEC.md`** for the full design: the four requirements (R1–R4), the five
comparison arms, the expected ordering, the decoy design, and the
acceptance/validation criteria.

## Files
- `SPEC.md` — the design (read this first).
- `build_necessity_probe.py` — construction **scaffold**: finds cross-source
  collision aliases, forms gold(A)/decoy(B) pairs, emits candidates in the
  standard normalized schema. **Mechanical assembly only** — gold answers are
  PROVISIONAL/UNVALIDATED.
- `test_build_necessity_probe.py` — offline stdlib unittest (tiny inline KB).
- `RESULTS.md` — empty template ("Not yet run.").

## How to run

Offline unit tests (no KB / no `outputs/` needed):
```bash
cd experiments/exp08_necessity_benchmark
python -m unittest test_build_necessity_probe.py -v
```

Build candidates from the unified KB (tiny slice shown; requires the KB to
exist locally — it is git-ignored):
```bash
python experiments/exp08_necessity_benchmark/build_necessity_probe.py \
  --kb-dir outputs/knowledge_bank/unified \
  --output-dir experiments/exp08_necessity_benchmark/candidates \
  --max-facts 5000 --limit 200 --min-decoy-similarity 0.1
```
Writes `candidates/candidates.jsonl`, `candidates/provenance.jsonl` (A/B source
pair per item), and `candidates/build_stats.json`. The `candidates/` dir is
git-ignored (data artifact); only code + `SPEC.md`/`README.md`/`RESULTS.md` are
committed.

## Required next steps (NOT DONE — this is 🔴 new-data)
1. **Human validation pass** over `candidates.jsonl` per SPEC.md §5: verify gold,
   R1–R4, question naturalness/source-ambiguity; drop or fix failing items. No
   auto-accepted gold ships.
2. Freeze the validated set and its `provenance.jsonl`.
3. Run the five arms (uniform / dense / oracle-source / wrong-skill / SkillRAG)
   through the standard pipeline (augment → RAG-SFT → infer → metrics). The
   oracle-source and wrong-skill arms hard-filter the retriever to the A / B
   source recorded in provenance; SkillRAG routes from the question only.
4. Report task-level metrics + source-selection accuracy per arm into `RESULTS.md`.

## Downstream integration (not modified here)
The emitted records are pipeline-consumable as-is. The oracle-source / wrong-skill
arms need a source hard-filter, which overlaps with the planned `oracle_source`
augment variant (Exp02) and the skill-override hook (Exp06); this experiment
reuses those once they land rather than forking the augmenter.
