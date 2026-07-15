#!/usr/bin/env python3
"""Print the current LLM cost for an in-progress or completed run."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _discover_cost_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]

    direct_file = path / "llm_cost.json"
    if direct_file.exists() and not recursive:
        return [direct_file]

    return sorted(path.rglob("llm_cost.json"))


def _read_cost_file(path: Path) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    skipped_lines = 0

    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped_lines += 1
    except FileNotFoundError:
        return [], 0

    return entries, skipped_lines


def _empty_bucket() -> dict[str, float]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "duration_seconds": 0.0,
    }


def collect_cost(path: Path, recursive: bool) -> dict[str, Any]:
    cost_files = _discover_cost_files(path, recursive)
    summary = _empty_bucket()
    by_function: defaultdict[str, dict[str, float]] = defaultdict(_empty_bucket)
    by_model: defaultdict[str, dict[str, float]] = defaultdict(_empty_bucket)
    skipped_lines = 0
    files_read = 0

    for cost_file in cost_files:
        entries, skipped = _read_cost_file(cost_file)
        skipped_lines += skipped
        if not entries and not cost_file.exists():
            continue

        files_read += 1
        for entry in entries:
            function_name = entry.get("FunctionName") or "unknown_function"
            model = entry.get("Model") or "unknown_model"
            prompt_tokens = _number(entry.get("PromptTokens"))
            completion_tokens = _number(entry.get("CompletionTokens"))
            cost = _number(entry.get("Cost"))
            duration = _number(entry.get("EndTime")) - _number(entry.get("StartTime"))
            if duration < 0:
                duration = 0.0

            for bucket in (summary, by_function[function_name], by_model[model]):
                bucket["calls"] += 1
                bucket["prompt_tokens"] += prompt_tokens
                bucket["completion_tokens"] += completion_tokens
                bucket["cost"] += cost
                bucket["duration_seconds"] += duration

    summary["total_tokens"] = summary["prompt_tokens"] + summary["completion_tokens"]

    return {
        "source": str(path),
        "files_read": files_read,
        "skipped_malformed_lines": skipped_lines,
        "summary": summary,
        "by_function": dict(by_function),
        "by_model": dict(by_model),
    }


def _format_int(value: float) -> str:
    return f"{int(value):,}"


def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def print_report(report: dict[str, Any], top: int) -> None:
    summary = report["summary"]
    print("Current LLM cost")
    print(f"Source: {report['source']}")
    print(f"Files read: {report['files_read']}")
    if report["skipped_malformed_lines"]:
        print(f"Skipped malformed/incomplete lines: {report['skipped_malformed_lines']}")
    print(f"Calls: {_format_int(summary['calls'])}")
    print(f"Prompt tokens: {_format_int(summary['prompt_tokens'])}")
    print(f"Completion tokens: {_format_int(summary['completion_tokens'])}")
    print(f"Total tokens: {_format_int(summary['total_tokens'])}")
    print(f"Duration: {_format_duration(summary['duration_seconds'])}")
    print(f"Cost: ${summary['cost']:.6f}")

    if report["by_model"]:
        print("\nBy model:")
        rows = sorted(report["by_model"].items(), key=lambda item: item[1]["cost"], reverse=True)
        for model, values in rows[:top]:
            print(
                f"  {model}: ${values['cost']:.6f}, "
                f"{_format_int(values['calls'])} calls, "
                f"{_format_int(values['prompt_tokens'] + values['completion_tokens'])} tokens"
            )

    if report["by_function"]:
        print("\nTop functions:")
        rows = sorted(report["by_function"].items(), key=lambda item: item[1]["cost"], reverse=True)
        for function_name, values in rows[:top]:
            print(
                f"  {function_name}: ${values['cost']:.6f}, "
                f"{_format_int(values['calls'])} calls, "
                f"{_format_duration(values['duration_seconds'])}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the current cost from llm_cost.json while a run is still active."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="results/neo4j_queryRetrieve_tools_v2_3",
        help="Path to llm_cost.json, a run directory, or a results directory.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search recursively for llm_cost.json when path is a directory.",
    )
    parser.add_argument("--watch", type=float, default=0, help="Refresh every N seconds.")
    parser.add_argument("--top", type=int, default=10, help="Number of model/function rows to print.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)

    while True:
        report = collect_cost(path, args.recursive)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_report(report, args.top)

        if args.watch <= 0:
            break
        time.sleep(args.watch)
        print("\n" + "-" * 72 + "\n")


if __name__ == "__main__":
    main()
