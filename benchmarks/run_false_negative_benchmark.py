#!/usr/bin/env python3
"""
Benchmark runner for false-negative bias fix verification.

Runs the full pipeline (prompt -> collect -> audit -> judge) for each event
in benchmarks_false_nagative.json using 10 parallel threads, then compares
the AI outcome against the ground truth.

Usage:
    python benchmarks/run_false_negative_benchmark.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
BENCHMARK_FILE = Path(__file__).parent / "benchmarks_false_nagative.json"
RESULTS_FILE = Path(__file__).parent / "false_negative_benchmark_results.json"
COLLECTORS = ["CollectorOpenSearch", "CollectorOpenSearch", "CollectorOpenSearch"]
MAX_WORKERS = 10
REQUEST_TIMEOUT = 180.0  # seconds per HTTP call
MAX_QUALITY_RETRIES = 2  # quality check retry loop iterations


def run_pipeline(event: dict) -> dict:
    """Run prompt -> collect -> audit -> judge for a single event."""
    event_id = event["event_id"]
    title = event["title"]
    description = event["description"]
    ground_truth = event["result"]

    # Build user_input from title + description
    user_input = f"{title}\n\n{description}"

    client = httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT)

    result = {
        "event_id": event_id,
        "title": title,
        "ground_truth": ground_truth,
        "ai_outcome": None,
        "ai_confidence": None,
        "match": None,
        "was_invalid": False,
        "error": None,
        "steps_completed": [],
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

        # Step 2: Collect (with quality check retry loop)
        resp = client.post("/step/collect", json={
            "prompt_spec": prompt_spec,
            "tool_plan": tool_plan,
            "collectors": COLLECTORS,
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
        for retry in range(MAX_QUALITY_RETRIES):
            qc_resp = client.post("/step/quality_check", json={
                "prompt_spec": prompt_spec,
                "evidence_bundles": evidence_bundles,
            })
            qc_resp.raise_for_status()
            qc_data = qc_resp.json()

            if not qc_data.get("ok"):
                break  # quality check itself failed, proceed anyway

            if qc_data.get("meets_threshold", False):
                break  # quality is good enough

            # Retry collect with quality feedback
            retry_hints = (qc_data.get("scorecard") or {}).get("retry_hints", {})
            if not retry_hints:
                break  # no actionable hints

            resp = client.post("/step/collect", json={
                "prompt_spec": prompt_spec,
                "tool_plan": tool_plan,
                "collectors": COLLECTORS,
                "include_raw_content": False,
                "quality_feedback": retry_hints,
            })
            resp.raise_for_status()
            retry_data = resp.json()
            if retry_data.get("ok"):
                evidence_bundles.extend(retry_data.get("evidence_bundles", []))

        result["quality_retries"] = retry

        # Get final quality scorecard to pass to auditor/judge
        quality_scorecard = qc_data.get("scorecard") if qc_data.get("ok") else None

        # Extract temporal_constraint from prompt_spec if auto-detected
        temporal_constraint = (prompt_spec.get("extra") or {}).get("temporal_constraint")

        # Step 3: Audit (with quality scorecard + temporal constraint)
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

        # Step 4: Judge (with quality scorecard + temporal constraint)
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

        # Compare: ground truth "Yes" -> "YES", "No" -> "NO"
        gt_normalized = ground_truth.upper()
        result["match"] = ai_outcome == gt_normalized

    except Exception as e:
        result["error"] = str(e)

    finally:
        client.close()

    return result


def main():
    # Load benchmark
    with open(BENCHMARK_FILE) as f:
        events = json.load(f)

    print(f"Loaded {len(events)} events from {BENCHMARK_FILE.name}")
    print(f"Running with {MAX_WORKERS} parallel threads, collectors={COLLECTORS}")
    print(f"{'='*80}")

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
                    "ai_outcome": None,
                    "ai_confidence": None,
                    "match": None,
                    "was_invalid": False,
                    "error": str(e),
                    "steps_completed": [],
                }

            results.append(result)

            # Status indicator
            status = "MATCH" if result["match"] else (
                "INVALID" if result["was_invalid"] else (
                    "MISMATCH" if result["match"] is False else "ERROR"
                )
            )
            icon = {
                "MATCH": "+",
                "INVALID": "?",
                "MISMATCH": "X",
                "ERROR": "!",
            }[status]
            conf = result.get('ai_confidence') or 0.0
            ai_out = str(result.get('ai_outcome') or 'ERR')
            print(
                f"[{i:2d}/{len(events)}] [{icon}] {status:8s} | "
                f"GT={result['ground_truth']:3s} AI={ai_out:7s} "
                f"conf={conf:.2f} | "
                f"{result['title'][:60]}"
            )

    elapsed = time.time() - start_time

    # Summary
    total = len(results)
    matches = sum(1 for r in results if r["match"] is True)
    mismatches = sum(1 for r in results if r["match"] is False and not r["was_invalid"])
    invalids = sum(1 for r in results if r["was_invalid"])
    errors = sum(1 for r in results if r["error"])

    # For false negatives specifically: how many flipped from NO->YES or NO->INVALID?
    fn_fixed_to_yes = sum(
        1 for r in results
        if r["ground_truth"].upper() == "YES" and r["ai_outcome"] == "YES"
    )
    fn_fixed_to_invalid = sum(
        1 for r in results
        if r["ground_truth"].upper() == "YES" and r["ai_outcome"] == "INVALID"
    )
    fn_still_no = sum(
        1 for r in results
        if r["ground_truth"].upper() == "YES" and r["ai_outcome"] == "NO"
    )

    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS — False Negative Bias Fix")
    print(f"{'='*80}")
    print(f"Total events:    {total}")
    print(f"Matches:         {matches} ({matches/total*100:.1f}%)")
    print(f"Mismatches:      {mismatches} ({mismatches/total*100:.1f}%)")
    print(f"INVALID:         {invalids} ({invalids/total*100:.1f}%)")
    print(f"Errors:          {errors}")
    print(f"Time:            {elapsed:.1f}s")
    print()
    print(f"FALSE NEGATIVE BREAKDOWN (ground truth = YES):")
    print(f"  Fixed to YES:     {fn_fixed_to_yes}")
    print(f"  Fixed to INVALID: {fn_fixed_to_invalid} (honest answer — absence of evidence)")
    print(f"  Still NO:         {fn_still_no} (bias not fixed for these)")
    print()

    # The fix is successful if fn_still_no decreased
    # Previously all 34 were NO (false negative). Now we expect most to be YES or INVALID.
    improvement = fn_fixed_to_yes + fn_fixed_to_invalid
    print(f"IMPROVEMENT: {improvement}/{total} cases no longer false negative ({improvement/total*100:.1f}%)")
    if fn_still_no > 0:
        print(f"REMAINING FALSE NEGATIVES ({fn_still_no}):")
        for r in results:
            if r["ground_truth"].upper() == "YES" and r["ai_outcome"] == "NO":
                print(f"  - [{r['event_id']}] {r['title'][:70]}")

    # Save detailed results
    output = {
        "benchmark": "false_negative_bias_fix",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "collectors": COLLECTORS,
            "max_workers": MAX_WORKERS,
        },
        "summary": {
            "total": total,
            "matches": matches,
            "mismatches": mismatches,
            "invalids": invalids,
            "errors": errors,
            "fn_fixed_to_yes": fn_fixed_to_yes,
            "fn_fixed_to_invalid": fn_fixed_to_invalid,
            "fn_still_no": fn_still_no,
            "improvement_rate": improvement / total if total else 0,
            "elapsed_seconds": elapsed,
        },
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to {RESULTS_FILE}")

    # Exit code: 0 if improvement > 50%, 1 otherwise
    sys.exit(0 if improvement / total > 0.5 else 1)


if __name__ == "__main__":
    main()
