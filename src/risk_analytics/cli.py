"""Command line entry points (FR-19).

    python -m risk_analytics ingest              index the registered corpus
    python -m risk_analytics ask "question"      write a cited analysis note

Ingest and analysis are separate commands so an indexed corpus can be queried
repeatedly without paying to rebuild it.
"""

from __future__ import annotations

import argparse
import sys

from . import config, indexing, llm, note as note_module


def cmd_ingest(args) -> int:
    report = indexing.ingest(force=args.force)
    print(f"ingested: {', '.join(report.ingested) or 'none'}")
    print(f"skipped (unchanged): {', '.join(report.skipped) or 'none'}")
    print(f"chunks added: {report.chunks_added}")
    print(f"chunks in store: {indexing.collection().count()}")
    return 0


def cmd_ask(args) -> int:
    if indexing.collection().count() == 0:
        print("the index is empty; run `ingest` first", file=sys.stderr)
        return 2

    budget = llm.Budget(ceiling_usd=args.ceiling)
    model = llm.LLM(budget=budget)

    where = {"kind": indexing.TABLE} if args.tables_only else None
    result = note_module.analyse(
        args.question, model, k=args.k, where=where, doc_ids=args.doc or None
    )
    markdown_path, trace_path = note_module.write(result, budget)

    print(f"note:  {markdown_path}")
    print(f"trace: {trace_path}")
    print(
        f"specialists: {', '.join(result.routing.selected) or 'none'} | "
        f"iterations: {result.trace.iteration_count} | "
        f"terminated: {result.trace.termination} | "
        f"cost: ${budget.spent_usd:.4f}"
    )
    if result.dropped_claims:
        print(
            f"warning: {len(result.dropped_claims)} claim(s) dropped for citing "
            "evidence that was not retrieved",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk_analytics",
        description="Multi-agent RAG over public financial filings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="classify, extract, chunk and embed the corpus")
    ingest.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="answer an analyst question with a cited note")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=8, help="chunks retrieved per query")
    ask.add_argument("--tables-only", action="store_true", help="restrict retrieval to table pages")
    ask.add_argument(
        "--doc", action="append", metavar="DOC_ID",
        help="restrict retrieval to this manifest document; repeatable. "
             "Omitted, the issuer is inferred from the question.",
    )
    ask.add_argument(
        "--ceiling", type=float, default=config.RUN_COST_CEILING_USD,
        help="spend ceiling in USD for this run",
    )
    ask.set_defaults(func=cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except config.MissingCredentialError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
