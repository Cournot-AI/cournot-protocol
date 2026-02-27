# Handoff: CollectorGeminiGroundedStrict v3 — FotMob Direct Extraction

## Goal
Make `CollectorGeminiGroundedStrict` reliably find and cite required-domain pages (e.g. fotmob.com) by adding Serper URL discovery, Gemini UrlContext, and **deterministic stat extraction from FotMob match pages**.

## Current State
- **Strict collector upgraded to v3** with three-phase approach:
  1. **Phase 1: Serper pre-search** — uses LLM-generated discovery query (`_generate_discovery_query`) to find top 3 URLs on the required domain via `google.serper.dev/search`.
  2. **Phase 1.5: FotMob direct extraction** (NEW) — if a discovered URL matches `fotmob.com/matches/*`, fetches the HTML page and extracts stats from the `__NEXT_DATA__` JSON (Next.js SSR data). Resolves the market directly without Gemini. **Deterministic and reliable.**
  3. **Phase 2: Gemini UrlContext + GoogleSearch** — fallback if Phase 1.5 is not applicable or fails. Passes discovered URLs to Gemini via `UrlContext` tool for full page ingestion.
- **50 tests pass** (48 new code + 2 pre-existing failures from missing `google.genai` module).
- **Live result**: Serper finds correct fotmob.com match page. Direct extraction gets **Shots outside box = 8** (Real Madrid, away team). Market correctly resolved as "Yes" (8 >= 5). Confidence 0.95.

## Key Finding: `__NEXT_DATA__` Solves JS-Rendering Problem
FotMob is a Next.js app. The HTML page embeds ALL match data (496KB JSON) in a `<script id="__NEXT_DATA__">` tag. A plain HTTP GET with browser headers returns this data — no Cloudflare challenge, no JS rendering needed. All 7 stat categories (top_stats, shots, expected_goals, passes, defence, duels, discipline) are present.

## Files Changed (this session)
- `agents/collector/fotmob.py` — NEW: FotMob stat extractor module with `fetch_match_stats()`, `find_stat()`, `match_team()`, dataclasses (`FotMobMatchData`, `FotMobStat`, `FotMobExtractionError`)
- `agents/collector/gemini_grounded_strict_agent.py` — v2→v3: added `_try_fotmob_direct_extraction()` method, Phase 1.5 block in `_collect_requirement()`, fotmob imports
- `tests/test_fotmob.py` — NEW: 16 unit tests for fotmob module (fetch, find_stat, match_team)
- `tests/test_fotmob_live.py` — NEW: 2 live integration tests against real fotmob.com
- `tests/test_collector_gemini_grounded_strict.py` — added 4 tests for fotmob direct extraction (TestFotMobDirectExtraction)
- `docs/plans/2026-02-27-fotmob-extractor-design.md` — design document
- `docs/plans/2026-02-27-fotmob-extractor-plan.md` — implementation plan

## Test Market for Next Session

Run the full API pipeline (step-by-step flow from `api/README.md`) with `CollectorGeminiGroundedStrict` to verify end-to-end resolution.

**Market:** "Will Real Madrid make 5+ shots outside box vs Osasuna on Feb 21?"

**Expected result:**
- Serper discovers: `https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz`
- Phase 1.5 extracts: Shots outside box = 8 (Real Madrid, away)
- Resolution: **Yes** (8 >= 5), confidence 0.95, tier 1

**How to run (step-by-step via API):**

```bash
# 1. Start the API server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# 2. Step 1: Prompt — compile the market query
#    NOTE: The prompt engineer must produce a PromptSpec with:
#    - source_targets pointing to fotmob.com
#    - expected_fields: ["Shots outside box"]
#    - prediction_semantics.target_entity: "Real Madrid"
#    - prediction_semantics.threshold: "5"
#    If the LLM prompt engineer doesn't produce these fields correctly,
#    you may need to manually construct or patch the prompt_spec.
curl -s -X POST http://localhost:8000/step/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Will Real Madrid make 5+ shots outside box vs Osasuna on Feb 21?\n\nThis market will resolve to YES if Real Madrid records 5 or more shots outside the box during the La Liga match against Osasuna scheduled for February 21, 2026, 17:30 UTC. Otherwise, it will resolve to NO.\n\nThe shots outside the box count will be determined using Real Madrid'\''s Shots outside box value shown on https://www.fotmob.com/ on the match page under Top stats -> Shots for this specific game.\n\nIf the match is not played and completed with an official result by March 21, 2026, 23:59 UTC, the market will resolve to NO.",
    "strict_mode": true
  }' -o prompt_out.json

# 3. Step 2: Collect — use CollectorGeminiGroundedStrict
curl -s -X POST http://localhost:8000/step/collect \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
r = json.load(open('prompt_out.json'))
print(json.dumps({
    'prompt_spec': r['prompt_spec'],
    'tool_plan': r['tool_plan'],
    'collectors': ['CollectorGeminiGroundedStrict'],
    'include_raw_content': False
}))
")" -o collect_out.json

# 4. Check the collect output for fotmob direct extraction
python3 -c "
import json
r = json.load(open('collect_out.json'))
for bundle in r.get('evidence_bundles', []):
    for item in bundle.get('items', []):
        ef = item.get('extracted_fields', {})
        print(f'outcome: {ef.get(\"outcome\")}')
        print(f'direct_extraction: {ef.get(\"direct_extraction\")}')
        print(f'fotmob_stat_title: {ef.get(\"fotmob_stat_title\")}')
        print(f'fotmob_stat_value: {ef.get(\"fotmob_stat_value\")}')
        print(f'confidence: {ef.get(\"confidence_score\")}')
"

# 5. Continue with audit → judge → bundle as in api/README.md
```

**Key requirement for Phase 1.5 to activate:**
The PromptSpec must have:
1. `data_requirements[].source_targets[].uri` containing `fotmob.com`
2. `data_requirements[].expected_fields` containing `"Shots outside box"`
3. `prediction_semantics.target_entity` = `"Real Madrid"`
4. `prediction_semantics.threshold` = `"5"`

If the LLM prompt engineer doesn't produce these fields, the fotmob direct extraction will not fire and it will fall back to Gemini Phase 2.

## Resolved Issues
1. ~~Non-deterministic stat values~~ — RESOLVED for fotmob markets via direct `__NEXT_DATA__` extraction
2. ~~JS-rendered sub-stats invisible~~ — RESOLVED via `__NEXT_DATA__` JSON
3. ~~Playwright needed~~ — NOT NEEDED

## Remaining Issues
1. **`google.genai` not installed locally** — 2 tests fail (pre-existing)
2. **FotMob-only** — direct extraction only works for fotmob.com
3. **Prompt engineer dependency** — the LLM prompt engineer must produce the right `source_targets`, `expected_fields`, `target_entity`, and `threshold` for Phase 1.5 to activate. If it doesn't, need to verify/patch the prompt_spec manually.

## Next Steps
1. **Test end-to-end via API** — run the step-by-step flow above with `CollectorGeminiGroundedStrict` and verify the collect step returns `direct_extraction: true` with `fotmob_stat_value: 8`.
2. **Verify prompt engineer output** — check that the LLM prompt engineer produces the right fields for fotmob markets. If not, consider adding fotmob-specific hints to the prompt engineer.
3. **LLM fallback for stat field identification** — when `expected_fields` is empty, use LLM to derive the stat name from the market question.
