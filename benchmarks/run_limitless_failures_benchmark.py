#!/usr/bin/env python3
"""
Benchmark runner for limitless_failures_34 benchmark.

Runs the full pipeline (prompt -> collect -> quality_check -> audit -> judge)
for each event. Dynamically selects collectors based on prompt_spec output:
  - If data requirements have source_targets with URLs -> CollectorSitePinned
  - If data requirements have deferred_source_discovery -> CollectorOpenSearch

Usage:
    python benchmarks/run_limitless_failures_benchmark.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
BENCHMARK_FILE = Path(__file__).parent / "benchmark_limitless_failures_34.json"
RESULTS_FILE = Path(__file__).parent / "limitless_failures_34_results.json"
MAX_WORKERS = 5  # Lower parallelism to avoid rate limits
REQUEST_TIMEOUT = 240.0  # seconds per HTTP call
MAX_QUALITY_RETRIES = 2


def select_collectors(prompt_spec: dict) -> list[str]:
    """Select collectors based on data requirements in prompt_spec.

    - If any data requirement has source_targets with URIs -> CollectorSitePinned
    - If any data requirement has deferred_source_discovery -> CollectorOpenSearch
    - Returns a list with duplicates for multi-run reliability.
    """
    has_pinned = False
    has_deferred = False

    data_requirements = prompt_spec.get("data_requirements", [])
    for req in data_requirements:
        targets = req.get("source_targets", [])
        deferred = req.get("deferred_source_discovery", False)

        if deferred:
            has_deferred = True
        if targets and any(t.get("uri") for t in targets):
            has_pinned = True

    collectors = []
    if has_pinned:
        collectors.extend(["CollectorSitePinned", "CollectorSitePinned"])
    if has_deferred:
        collectors.extend(["CollectorOpenSearch", "CollectorOpenSearch"])

    # Fallback: if neither detected, use both
    if not collectors:
        collectors = ["CollectorSitePinned", "CollectorOpenSearch"]

    return collectors


def run_pipeline(event: dict) -> dict:
    """Run prompt -> collect -> quality_check -> audit -> judge for a single event."""
    event_id = event["event_id"]
    title = event["title"]
    description = event["description"]
    ground_truth = event["result"]
    original_match_result = event.get("match_result", "unknown")
    original_ai_result = event.get("ai_result", "unknown")

    user_input = f"{title}\n\n{description}"
    client = httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT)

    result = {
        "event_id": event_id,
        "title": title,
        "ground_truth": ground_truth,
        "original_match_result": original_match_result,
        "original_ai_result": original_ai_result,
        "ai_outcome": None,
        "ai_confidence": None,
        "match": None,
        "was_invalid": False,
        "error": None,
        "steps_completed": [],
        "collectors_used": [],
        "quality_retries": 0,
    }

    try:
        # Step 1: Prompt
        resp = client.post("/step/prompt", json={
            "user_input": user_input,
            "strict_mode": True,
        })
        resp.raise_for_status()
        prompt_data = resp.json()
        if not prompt_data.get("ok"):
            result["error"] = f"Prompt failed: {prompt_data.get('error')}"
            return result
        result["steps_completed"].append("prompt")

        prompt_spec = prompt_data["prompt_spec"]
        tool_plan = prompt_data["tool_plan"]

        # Select collectors based on data requirements
        collectors = select_collectors(prompt_spec)
        result["collectors_used"] = collectors

        # Step 2: Collect
        resp = client.post("/step/collect", json={
            "prompt_spec": prompt_spec,
            "tool_plan": tool_plan,
            "collectors": collectors,
            "include_raw_content": False,
        })
        resp.raise_for_status()
        collect_data = resp.json()
        if not collect_data.get("ok"):
            result["error"] = f"Collect failed: {collect_data.get('errors')}"
            return result
        result["steps_completed"].append("collect")

        evidence_bundles = collect_data["evidence_bundles"]

        # Step 2.5: Quality check + retry loop
        qc_data = {}
        for retry in range(MAX_QUALITY_RETRIES):
            qc_resp = client.post("/step/quality_check", json={
                "prompt_spec": prompt_spec,
                "evidence_bundles": evidence_bundles,
            })
            qc_resp.raise_for_status()
            qc_data = qc_resp.json()

            if not qc_data.get("ok"):
                break

            if qc_data.get("meets_threshold", False):
                break

            retry_hints = (qc_data.get("scorecard") or {}).get("retry_hints", {})
            if not retry_hints:
                break

            resp = client.post("/step/collect", json={
                "prompt_spec": prompt_spec,
                "tool_plan": tool_plan,
                "collectors": collectors,
                "include_raw_content": False,
                "quality_feedback": retry_hints,
            })
            resp.raise_for_status()
            retry_data = resp.json()
            if retry_data.get("ok"):
                evidence_bundles.extend(retry_data.get("evidence_bundles", []))

        result["quality_retries"] = retry if MAX_QUALITY_RETRIES > 0 else 0
        result["steps_completed"].append("quality_check")

        quality_scorecard = qc_data.get("scorecard") if qc_data.get("ok") else None
        temporal_constraint = (prompt_spec.get("extra") or {}).get("temporal_constraint")

        # Step 3: Audit
        audit_payload = {
            "prompt_spec": prompt_spec,
            "evidence_bundles": evidence_bundles,
        }
        if quality_scorecard:
            audit_payload["quality_scorecard"] = quality_scorecard
        if temporal_constraint:
            audit_payload["temporal_constraint"] = temporal_constraint

        resp = client.post("/step/audit", json=audit_payload)
        resp.raise_for_status()
        audit_data = resp.json()
        if not audit_data.get("ok"):
            result["error"] = f"Audit failed: {audit_data.get('errors')}"
            return result
        result["steps_completed"].append("audit")

        reasoning_trace = audit_data["reasoning_trace"]
        result["preliminary_outcome"] = reasoning_trace.get("preliminary_outcome")

        # Step 4: Judge
        judge_payload = {
            "prompt_spec": prompt_spec,
            "evidence_bundles": evidence_bundles,
            "reasoning_trace": reasoning_trace,
        }
        if quality_scorecard:
            judge_payload["quality_scorecard"] = quality_scorecard
        if temporal_constraint:
            judge_payload["temporal_constraint"] = temporal_constraint

        resp = client.post("/step/judge", json=judge_payload)
        resp.raise_for_status()
        judge_data = resp.json()
        if not judge_data.get("ok"):
            result["error"] = f"Judge failed: {judge_data.get('errors')}"
            return result
        result["steps_completed"].append("judge")

        ai_outcome = judge_data.get("outcome", "UNKNOWN")
        ai_confidence = judge_data.get("confidence", 0.0)

        result["ai_outcome"] = ai_outcome
        result["ai_confidence"] = ai_confidence
        result["was_invalid"] = ai_outcome == "INVALID"

        gt_normalized = ground_truth.upper()
        result["match"] = ai_outcome == gt_normalized

    except Exception as e:
        result["error"] = str(e)

    finally:
        client.close()

    return result


def main():
    with open(BENCHMARK_FILE) as f:
        events = json.load(f)

    print(f"Loaded {len(events)} events from {BENCHMARK_FILE.name}")
    print(f"Running with {MAX_WORKERS} parallel threads")
    print(f"Collector selection: sitepinned for URL sources, opensearch for deferred")
    print(f"{'='*90}")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_event = {
            executor.submit(run_pipeline, event): event
            for event in events
        }

        for i, future in enumerate(as_completed(future_to_event), 1):
            event = future_to_event[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "event_id": event["event_id"],
                    "title": event["title"],
                    "ground_truth": event["result"],
                    "original_match_result": event.get("match_result", "unknown"),
                    "original_ai_result": event.get("ai_result", "unknown"),
                    "ai_outcome": None,
                    "ai_confidence": None,
                    "match": None,
                    "was_invalid": False,
                    "error": str(e),
                    "steps_completed": [],
                    "collectors_used": [],
                    "quality_retries": 0,
                }

            results.append(result)

            status = "MATCH" if result["match"] else (
                "INVALID" if result["was_invalid"] else (
                    "MISMATCH" if result["match"] is False else "ERROR"
                )
            )
            icon = {"MATCH": "+", "INVALID": "?", "MISMATCH": "X", "ERROR": "!"}[status]
            conf = result.get("ai_confidence") or 0.0
            ai_out = str(result.get("ai_outcome") or "ERR")
            orig = result.get("original_ai_result", "?")
            collectors = ",".join(set(result.get("collectors_used", [])))
            print(
                f"[{i:2d}/{len(events)}] [{icon}] {status:8s} | "
                f"GT={result['ground_truth']:3s} AI={ai_out:7s} "
                f"(was:{orig:7s}) conf={conf:.2f} | "
                f"[{collectors}] | "
                f"{result['title'][:50]}"
            )

    elapsed = time.time() - start_time

    # Summary
    total = len(results)
    matches = sum(1 for r in results if r["match"] is True)
    mismatches = sum(1 for r in results if r["match"] is False and not r["was_invalid"])
    invalids = sum(1 for r in results if r["was_invalid"])
    errors = sum(1 for r in results if r["error"])

    # Improvement analysis vs original benchmark
    improved = sum(
        1 for r in results
        if r["match"] is True and r["original_match_result"] != "match"
    )
    regressed = sum(
        1 for r in results
        if r["match"] is False and not r["was_invalid"]
        and r["original_match_result"] != "mismatch"
    )
    was_invalid_now_match = sum(
        1 for r in results
        if r["match"] is True and r["original_match_result"] == "invalid"
    )
    was_mismatch_now_match = sum(
        1 for r in results
        if r["match"] is True and r["original_match_result"] == "mismatch"
    )
    was_mismatch_now_invalid = sum(
        1 for r in results
        if r["was_invalid"] and r["original_match_result"] == "mismatch"
    )
    still_mismatch = sum(
        1 for r in results
        if r["match"] is False and not r["was_invalid"]
        and r["original_match_result"] == "mismatch"
    )
    still_invalid = sum(
        1 for r in results
        if r["was_invalid"] and r["original_match_result"] == "invalid"
    )

    print(f"\n{'='*90}")
    print(f"BENCHMARK RESULTS — Limitless Failures 34")
    print(f"{'='*90}")
    print(f"Total events:    {total}")
    print(f"Matches:         {matches} ({matches/total*100:.1f}%)")
    print(f"Mismatches:      {mismatches} ({mismatches/total*100:.1f}%)")
    print(f"INVALID:         {invalids} ({invalids/total*100:.1f}%)")
    print(f"Errors:          {errors}")
    print(f"Time:            {elapsed:.1f}s")
    print()
    print(f"IMPROVEMENT vs ORIGINAL BENCHMARK:")
    print(f"  Was mismatch, now MATCH:    {was_mismatch_now_match}")
    print(f"  Was mismatch, now INVALID:  {was_mismatch_now_invalid}")
    print(f"  Was invalid,  now MATCH:    {was_invalid_now_match}")
    print(f"  Still mismatch:             {still_mismatch}")
    print(f"  Still invalid:              {still_invalid}")
    print(f"  Regressions:                {regressed}")
    print()

    original_mismatches = sum(1 for e in events if e.get("match_result") == "mismatch")
    original_invalids = sum(1 for e in events if e.get("match_result") == "invalid")
    print(f"ORIGINAL BENCHMARK: {original_mismatches} mismatches, {original_invalids} invalids")
    print(f"CURRENT RUN:        {mismatches} mismatches, {invalids} invalids, {matches} matches")

    if still_mismatch > 0:
        print(f"\nSTILL MISMATCHED ({still_mismatch}):")
        for r in results:
            if r["match"] is False and not r["was_invalid"] and r["original_match_result"] == "mismatch":
                print(f"  - [{r['event_id']}] GT={r['ground_truth']} AI={r['ai_outcome']} | {r['title'][:70]}")

    if still_invalid > 0:
        print(f"\nSTILL INVALID ({still_invalid}):")
        for r in results:
            if r["was_invalid"] and r["original_match_result"] == "invalid":
                print(f"  - [{r['event_id']}] GT={r['ground_truth']} AI={r['ai_outcome']} | {r['title'][:70]}")

    if errors > 0:
        print(f"\nERRORS ({errors}):")
        for r in results:
            if r["error"]:
                print(f"  - [{r['event_id']}] {r['error'][:100]} | {r['title'][:50]}")

    # Save detailed results
    output = {
        "benchmark": "limitless_failures_34",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "max_workers": MAX_WORKERS,
            "max_quality_retries": MAX_QUALITY_RETRIES,
            "collector_selection": "dynamic (sitepinned for URLs, opensearch for deferred)",
        },
        "summary": {
            "total": total,
            "matches": matches,
            "mismatches": mismatches,
            "invalids": invalids,
            "errors": errors,
            "was_mismatch_now_match": was_mismatch_now_match,
            "was_mismatch_now_invalid": was_mismatch_now_invalid,
            "was_invalid_now_match": was_invalid_now_match,
            "still_mismatch": still_mismatch,
            "still_invalid": still_invalid,
            "regressions": regressed,
            "elapsed_seconds": elapsed,
        },
        "original_benchmark_summary": {
            "total": total,
            "mismatches": original_mismatches,
            "invalids": original_invalids,
        },
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to {RESULTS_FILE}")

    sys.exit(0 if matches / total > 0.3 else 1)


if __name__ == "__main__":
    main()
