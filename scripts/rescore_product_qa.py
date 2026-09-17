#!/usr/bin/env python
"""CLI: re-score a saved product QA run with the CURRENT grader, without
calling the model again.

A grader change and a generation change both move the pass rate, and
comparing one run before a change to one run after it cannot tell you which
one did the moving. This replays stored answers through the current
evaluate_turn(), so:

    old answers + old grader   the number in the saved run
    old answers + NEW grader   this script          -> the grader's effect
    new answers + NEW grader   a fresh harness run  -> generation's effect

It also makes a grader change auditable after the fact: any run ever saved
to output/ can be re-scored, and the model is never invoked, so the answers
being scored are fixed.

Usage:
    python scripts/rescore_product_qa.py output/product_qa_results.json
    python scripts/rescore_product_qa.py a.json b.json      # compare two runs
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.product_qa import aggregate, evaluate_turn


def rescore(run: dict) -> tuple[dict, list[dict]]:
    """Re-grade every stored turn with the current grader. Returns
    (summary, case_results) in the same shape aggregate() expects."""
    cases = []
    for case in run["cases"]:
        turns = [
            {
                "spec": turn["spec"],
                "result": turn["result"],
                "metrics": evaluate_turn(turn["spec"], turn["result"], case["category"]),
            }
            for turn in case["turns"]
        ]
        cases.append({**case, "turns": turns})
    return aggregate(cases), cases


def _failing(cases: list[dict]) -> set[tuple[str, int]]:
    return {
        (case["id"], i)
        for case in cases
        for i, turn in enumerate(case["turns"])
        if not turn["metrics"]["passed"]
    }


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_run(label: str, path: str, run: dict) -> list[dict]:
    stored = run.get("summary", {}).get("overall", {})
    summary, cases = rescore(run)
    o = summary["overall"]

    print(f"\n{label}: {path}")
    if stored:
        print(f"  as originally scored : {stored['passed']}/{stored['n_turns']} "
              f"({stored['pass_rate'] * 100:.0f}%)")
    print(f"  re-scored (current)  : {o['passed']}/{o['n_turns']} ({o['pass_rate'] * 100:.0f}%)")
    print(f"  fact coverage        : {(o['avg_fact_coverage'] or 0) * 100:.0f}%")
    print(f"  validation clean     : {o['validation_ok_rate'] * 100:.0f}%")

    failing = sorted(_failing(cases))
    print(f"  failing turns        : {', '.join(f'{i}t{t}' for i, t in failing) or 'none'}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score saved product QA runs with the current grader.")
    parser.add_argument("runs", nargs="+", help="Saved product QA result JSON file(s).")
    args = parser.parse_args()

    all_cases = []
    for i, path in enumerate(args.runs):
        label = f"RUN {i + 1}"
        all_cases.append(_print_run(label, path, _load(path)))

    if len(all_cases) == 2:
        first, second = _failing(all_cases[0]), _failing(all_cases[1])
        print("\nDIFFERENCE (both re-scored with the current grader)")
        fixed = sorted(first - second)
        broken = sorted(second - first)
        print(f"  fixed in run 2  : {', '.join(f'{i}t{t}' for i, t in fixed) or 'none'}")
        print(f"  broken in run 2 : {', '.join(f'{i}t{t}' for i, t in broken) or 'none'}")
        print(f"  failing in both : {', '.join(f'{i}t{t}' for i, t in sorted(first & second)) or 'none'}")


if __name__ == "__main__":
    main()
