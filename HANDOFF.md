# Handoff: Temporal Guard — Benchmarking Phase

## Branch: `feat/dashboard-dispute-contract`

## What was implemented

An opt-in temporal constraint system that prevents the LLM auditor/judge from confidently answering YES/NO for future or in-progress events. The temporal status (FUTURE/ACTIVE/PAST) is computed **at resolution time** by comparing `event_time` against the current clock, not at prompt creation time.

## Architecture

```
/step/prompt (LLM detects temporal signals)
    → prompt_spec.extra["temporal_constraint"] = {
        "enabled": true,
        "event_time": "2026-09-23T00:00:00Z",
        "reason": "Match scheduled for Sep 23 2026"
      }
      (NO status field — PE only detects relevance + event_time)

/step/audit or /step/judge (receives temporal_constraint)
    → ctx.extra["temporal_context"] = temporal_constraint
    → _resolve_temporal_status(event_time, ctx.now()) computes FUTURE/ACTIVE/PAST
    → Injects "## TEMPORAL ADVISORY" into LLM messages
    → FUTURE → forces INVALID; ACTIVE → INVALID unless concluded; PAST → normal
```

## Status computation logic (`_resolve_temporal_status`)

Located in both `agents/auditor/llm_reasoner.py` and `agents/judge/agent.py`:

```python
event_time > current_time           → FUTURE
current_time - event_time < 24h     → ACTIVE
current_time - event_time >= 24h    → PAST
parse failure                       → UNKNOWN
```

## Files changed

| File | What changed |
|------|-------------|
| `agents/prompt_engineer/prompts.py` | LLM schema includes optional `temporal_constraint: { enabled, event_time, reason }` |
| `agents/prompt_engineer/llm_compiler.py` | Passes `temporal_constraint` into `prompt_spec.extra` when present |
| `api/routes/steps.py` | `AuditRequest` and `JudgeRequest` have new optional `temporal_constraint` field; handlers wire into `ctx.extra["temporal_context"]` |
| `agents/auditor/prompts.py` | System prompt has "CRITICAL — Temporal Reasoning" section |
| `agents/auditor/llm_reasoner.py` | `_resolve_temporal_status()` helper + `_build_temporal_context_prompt(ctx, now)` method; injected after quality context |
| `agents/judge/prompts.py` | "Temporal Validity Check" subsection under Decision Guidelines |
| `agents/judge/agent.py` | Same `_resolve_temporal_status()` helper + `_build_temporal_context_prompt(ctx, now)` |
| `orchestrator/pipeline.py` | `_step_audit` and `_step_judge` extract `temporal_constraint` from `prompt_spec.extra` → inject into `ctx.extra["temporal_context"]` |
| `benchmarks/run_false_negative_benchmark.py` | Extracts `temporal_constraint` from `prompt_spec["extra"]` after prompt step, passes to audit/judge payloads |
| `tests/unit/test_temporal_context_injection.py` | 21 tests: `_resolve_temporal_status`, prompt builders, injection presence/absence/ordering, e2e judge run |

## Test status

```
123 passed, 18 skipped, 0 failures (tests/unit/)
```

## How temporal_constraint flows through the system

### Path 1: Full pipeline (`orchestrator/pipeline.py`)
`prompt_spec.extra["temporal_constraint"]` → `_step_audit`/`_step_judge` read it → inject into `ctx.extra["temporal_context"]` → auditor/judge `_build_temporal_context_prompt(temporal_ctx, ctx.now())`

### Path 2: Step-by-step API (`api/routes/steps.py`)
Frontend calls `/step/prompt` → response includes `prompt_spec.extra.temporal_constraint` → frontend passes it back as `temporal_constraint` field in `/step/audit` and `/step/judge` request bodies → handler injects into `ctx.extra["temporal_context"]`

### Path 3: Benchmark runner (`benchmarks/run_false_negative_benchmark.py`)
After `/step/prompt`, extracts `prompt_spec["extra"]["temporal_constraint"]` → passes as `temporal_constraint` in audit/judge payloads.

## What needs to happen next: Benchmarking

### 1. Run existing false-negative benchmark
The benchmark runner already wires `temporal_constraint` through. Start the server and run it:
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
python benchmarks/run_false_negative_benchmark.py
```

### 2. Create a temporal-specific benchmark
Create `benchmarks/benchmarks_temporal.json` with events that have clear future/past dates:
- **Future events** that should return INVALID (e.g. "Will Team X win the 2027 World Cup final?")
- **Past events** that should still resolve YES/NO normally
- **Active/recent events** to test the 24h window

### 3. Manual spot-checks
```bash
# Call /step/prompt with a future event
curl -s -X POST http://localhost:8000/step/prompt \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Will Arsenal win the Champions League final on May 31 2027?"}' \
  -o prompt_out.json

# Verify prompt_spec.extra.temporal_constraint exists in output
python3 -c "import json; r=json.load(open('prompt_out.json')); print(json.dumps(r['prompt_spec']['extra'].get('temporal_constraint'), indent=2))"

# Then run collect → audit → judge and verify INVALID outcome
```

### 4. Key things to verify
- Future event → `temporal_constraint.enabled=true` in prompt_spec → audit/judge compute FUTURE → outcome=INVALID
- Past event → either no `temporal_constraint` or status=PAST → normal YES/NO resolution
- No `temporal_constraint` at all → completely unchanged behavior (backwards compatible)

## Key design decisions
- **No new Pydantic models** — `temporal_constraint` is a plain dict in `extra`
- **Soft advisory only** — LLM is instructed to return INVALID for FUTURE, not a hard code-level block
- **Backwards compatible** — absent `temporal_constraint` changes nothing
- **Status computed at resolution time** — prompt engineer provides `event_time`, auditor/judge compute FUTURE/ACTIVE/PAST against `ctx.now()`
- **24h ACTIVE window** — events within 24h of `event_time` are "ACTIVE" (may still be in progress)
