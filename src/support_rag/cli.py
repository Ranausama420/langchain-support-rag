import argparse
import json
from pathlib import Path

from support_rag.config import Settings
from support_rag.evaluation import evaluate
from support_rag.service import Service


def main():
    parser = argparse.ArgumentParser(description="NimbusDesk Support RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--model")
    compare = sub.add_parser("compare")
    compare.add_argument("question")
    compare.add_argument("--models", nargs="+", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--dataset", type=Path, default=Path("examples/evaluation.json"))
    evaluation.add_argument("--model")
    evaluation.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    service = Service(Settings())
    if args.command == "ingest":
        result = service.store.ingest()
    elif args.command == "ask":
        result = service.ask(args.question, args.model).model_dump()
    elif args.command == "compare":
        result = service.compare(args.question, args.models)
    else:
        result = evaluate(service, args.dataset, args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.command == "evaluate" and result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
