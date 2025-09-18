"""Command line interface for running SWE-Fixer via LangGraph."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import PipelineConfig
from .graph import run_swe_fixer_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SWE-Fixer pipeline with LangGraph.")
    parser.add_argument("--api_key", type=str, default="token-abc123", help="API key for the model provider.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL for the model provider.")
    parser.add_argument("--model_name", type=str, default="SWE-Fixer", help="Model identifier to record in outputs.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="tiktoken encoding name or model id used for token counting (e.g. cl100k_base).",
    )
    parser.add_argument("--instance_file_path", type=str, required=True, help="Path to the instance metadata (JSON/JSONL).")
    parser.add_argument("--code_structure_path", type=str, required=True, help="Directory containing repository structure JSON files.")
    parser.add_argument("--bm25_retrieval_files", type=str, required=True, help="Input dataset for BM25 retrieval prompts.")
    parser.add_argument("--bm25_input_dataset", type=str, required=True, help="Where to store retrieval input dataset records.")
    parser.add_argument("--bm25_results_file", type=str, required=True, help="Where to store retrieval predictions.")
    parser.add_argument("--editing_dataset_path", type=str, required=True, help="Where to store editing dataset records.")
    parser.add_argument("--editing_result_path", type=str, required=True, help="Where to store editing predictions.")
    parser.add_argument("--post_process", action="store_true", help="Enable post-processing heuristics during inference.")
    parser.add_argument("--run_retrieval", action="store_true", help="Execute the retrieval stage.")
    parser.add_argument("--run_editing", action="store_true", help="Execute the editing stage.")
    return parser


def cli(argv: Any | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path,
        instance_file_path=args.instance_file_path,
        code_structure_path=args.code_structure_path,
        bm25_retrieval_files=args.bm25_retrieval_files,
        bm25_input_dataset=args.bm25_input_dataset,
        bm25_results_file=args.bm25_results_file,
        editing_dataset_path=args.editing_dataset_path,
        editing_result_path=args.editing_result_path,
        post_process=args.post_process,
        run_retrieval=args.run_retrieval,
        run_editing=args.run_editing,
    )

    try:
        final_state = run_swe_fixer_pipeline(config)
    except Exception as exc:  # pragma: no cover - surfaced to the user
        print(f"[swe-fixer] Pipeline execution failed: {exc}", file=sys.stderr)
        return 1

    summary = final_state.get("pipeline_summary")
    if summary:
        print("[swe-fixer] Summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()
