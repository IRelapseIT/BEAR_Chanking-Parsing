import argparse
import json
import sys
import tracemalloc
from pathlib import Path

from .bootstrap import lock_dependencies, prepare_images, prepare_model
from .comparison import compare_results
from .errors import BearError
from .models import DocumentContext, ProcessingProfile
from .runtime import Services
from .serialization import load_result, result_adapter


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="BEAR: парсинг и адаптивный чанкинг")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--model-dir", default="models/e5")
    parser.add_argument("--offline", action="store_true", help="Не выполнять семантический инференс")
    parser.add_argument("--test-counter", action="store_true", help="Тестовый счётчик; индексация запрещена")
    parser.add_argument("--allow-cp1251", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Проверить установку без сети и запуска сервисов")
    doctor.add_argument("--require-model", action="store_true")
    process = sub.add_parser("process")
    process.add_argument("file", type=Path)
    process.add_argument("--tenant", default="local")
    process.add_argument("--space", default="demo")
    process.add_argument("--document", required=True)
    process.add_argument("--version", type=int, default=1)
    process.add_argument("--output", type=Path)
    process.add_argument("--mode", choices=["adaptive", "fixed", "token", "semantic"], default="adaptive")
    process.add_argument("--benchmark", action="store_true")
    index = sub.add_parser("index")
    index.add_argument("result", nargs="?", type=Path)
    index.add_argument("--generation")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--tenant", default="local")
    search.add_argument("--space", default="demo")
    search.add_argument("--top-k", type=int, default=5)
    compare = sub.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    prepare = sub.add_parser("prepare-model")
    prepare.add_argument("--weights", action="store_true")
    freeze = sub.add_parser("lock")
    freeze.add_argument("--output", type=Path, default=Path("requirements.lock"))
    images = sub.add_parser("prepare-images")
    images.add_argument("--output", type=Path, default=Path("compose.lock.env"))
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("corpus", type=Path)
    evaluation.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    runtime = None
    try:
        if args.command == "doctor":
            from .doctor import diagnose
            report = diagnose(args.model_dir)
            print_json(report)
            return 1 if args.require_model and not report["model"]["tokenizer_ready"] else 0
        if args.command == "prepare-model":
            print_json(prepare_model(Path(args.model_dir), weights=args.weights))
            return 0
        if args.command == "prepare-images":
            prepare_images(args.output)
            print_json({"lock": str(args.output)})
            return 0
        if args.command == "lock":
            lock_dependencies(args.output)
            return 0
        if args.command == "compare":
            print_json(compare_results(load_result(json.loads(args.left.read_text(encoding="utf-8"))),
                                       load_result(json.loads(args.right.read_text(encoding="utf-8")))))
            return 0
        runtime = Services(args.state_dir, args.model_dir, test_counter=args.test_counter,
                           offline=args.offline, allow_cp1251=args.allow_cp1251)
        if args.command == "process":
            if args.benchmark:
                tracemalloc.start()
            context = DocumentContext(args.tenant, args.space, args.document, args.version, args.file.name)
            with args.file.open("rb") as file:
                result = runtime.process_stream(file, context, ProcessingProfile(mode=args.mode))
            if args.benchmark:
                tracemalloc.stop()
            serialized = result_adapter().dump_json(result, indent=2).decode()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized, encoding="utf-8")
                print_json({"generation_id": result.generation_id, "chunks": len(result.chunks),
                            "completeness": result.completeness, "output": str(args.output)})
            else:
                print(serialized)
        elif args.command == "index":
            if bool(args.result) == bool(args.generation):
                raise ValueError("Укажите файл результата либо --generation")
            if args.generation:
                print_json(runtime.indexer.retry(args.generation))
            else:
                print_json(runtime.indexer.index(load_result(json.loads(args.result.read_text(encoding="utf-8")))))
        elif args.command == "search":
            from dataclasses import asdict
            print_json([asdict(hit) for hit in runtime.indexer.search(args.query, args.tenant, args.space, args.top_k)])
        elif args.command == "evaluate":
            from .evaluation import evaluate
            report = evaluate(runtime, args.corpus)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print_json(report)
        return 0
    except (BearError, ValueError, OSError) as exc:
        print_json({"error": type(exc).__name__, "detail": str(exc)})
        return 1
    finally:
        if runtime:
            runtime.close()


if __name__ == "__main__":
    sys.exit(main())
