"""Command entrypoint for versioned local inference experiments."""
import argparse
import json
import sys
from .preflight import load_config, main as preflight_main
from .acquisition import acquire_model
from .runtime import smoke


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "preflight":
        return preflight_main(argv)
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("acquire-model")
    download.add_argument("--config", required=True)
    download.add_argument("--destination", required=True)
    run = commands.add_parser("smoke")
    run.add_argument("--config", required=True)
    run.add_argument("--binary", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "acquire-model":
        print(json.dumps(acquire_model(config, args.destination)))
        return 0
    record, directory = smoke(config, args.binary, args.model, args.output_root)
    print(json.dumps({"run_id": record["run_id"], "status": record["status"], "directory": str(directory)}))
    return 0 if record["status"] == "success" else 2
