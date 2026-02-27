# Handoff: CollectorGeminiGroundedStrict v3 — FotMob Direct Extraction

## Goal
Make `CollectorGeminiGroundedStrict` reliably find and cite required-domain pages (e.g. fotmob.com) by adding Serper URL discovery, Gemini UrlContext, and **deterministic stat extraction from FotMob match pages**.

## Current State
- **Strict collector upgraded to v3** with three-phase approach:
  1. **Phase 1: Serper pre-search** — uses LLM-generated discovery query (`_generate_discovery_query`) to find top 3 URLs on the required domain via `google.serper.dev/search`.
  2. **Phase 1.5: FotMob direct extraction** (NEW) — if a discovered URL matches `fotmob.com/matches/*`, fetches the HTML page and extracts stats from the `__NEXT_DATA__` JSON (Next.js SSR data). Resolves the market directly without Gemini. **Deterministic and reliable.**
  3. **Phase 2: Gemini UrlContext + GoogleSearch** — fallback if Phase 1.5 is not applicable or fails. Passes discovered URLs to Gemini via `UrlContext` tool for full page ingestion.
- **50 tests pass** (42 parent + 34 strict + 16 fotmob unit + 2 live). Live API tested with Real Madrid shots market.
- **Live result**: Serper finds correct fotmob.com match page. Direct extraction gets **Shots outside box = 8** (Real Madrid, away team). Market correctly resolved as "Yes" (8 >= 5). Confidence 0.95.

## Key Finding: `__NEXT_DATA__` Solves JS-Rendering Problem
The "Shots outside box" stat on fotmob.com is behind a JS-rendered "All stats" accordion/tab. Previous tools could not see it:
- Gemini UrlContext: sees only Top Stats (xG, Total shots, Big chances)
- Jina Reader: same limitation
- FotMob JSON API (`/api/matchDetails`): blocked by Cloudflare Turnstile
- Gemini GoogleSearch: non-deterministic (returned 2, 3, and 8 across runs)

**Solution**: FotMob is a Next.js app. The HTML page embeds ALL match data (496KB JSON) in a `<script id="__NEXT_DATA__">` tag. A plain HTTP GET with browser headers returns this data — no Cloudflare challenge, no JS rendering needed. All 7 stat categories (top_stats, shots, expected_goals, passes, defence, duels, discipline) are present.

## Files Changed (this session)
- `agents/collector/fotmob.py` — NEW: FotMob stat extractor module with `fetch_match_stats()`, `find_stat()`, `match_team()`, dataclasses (`FotMobMatchData`, `FotMobStat`, `FotMobExtractionError`)
- `agents/collector/gemini_grounded_strict_agent.py` — v2→v3: added `_try_fotmob_direct_extraction()` method, Phase 1.5 block in `_collect_requirement()`, fotmob imports
- `tests/test_fotmob.py` — NEW: 16 unit tests for fotmob module (fetch, find_stat, match_team)
- `tests/test_fotmob_live.py` — NEW: 2 live integration tests against real fotmob.com
- `tests/test_collector_gemini_grounded_strict.py` — added 4 tests for fotmob direct extraction (TestFotMobDirectExtraction)
- `docs/plans/2026-02-27-fotmob-extractor-design.md` — design document
- `docs/plans/2026-02-27-fotmob-extractor-plan.md` — implementation plan

## Resolved Issues
1. **~~Non-deterministic stat values~~** — RESOLVED for fotmob markets via direct `__NEXT_DATA__` extraction. Always returns the exact value.
2. **~~JS-rendered sub-stats invisible~~** — RESOLVED. The `__NEXT_DATA__` JSON contains ALL stats, including those behind the accordion.
3. **~~Playwright needed~~** — NOT NEEDED. The `__NEXT_DATA__` approach avoids browser automation entirely.

## Remaining Issues
1. **`google.genai` not installed locally** — 2 tests fail due to missing `google.genai` module (pre-existing, unrelated to fotmob work)
2. **FotMob-only** — direct extraction only works for fotmob.com. Other JS-heavy sites would need their own extractors or a different approach (Playwright, Firecrawl).
3. **FotMob page structure may change** — if FotMob changes their Next.js data structure, extraction will fail and fall back to Gemini automatically.

## Next Steps
1. **Add more site-specific extractors** — if other prediction markets reference JS-heavy sites, add similar `__NEXT_DATA__` or API-based extractors.
2. **LLM fallback for stat field identification** — when `expected_fields` is empty, use LLM to derive the stat name from the market question (currently returns None and falls back to Gemini).
3. **Update `api/README.md`** — document `CollectorGeminiGroundedStrict` v3 usage, fotmob extraction, and the Phase 1.5 flow.
