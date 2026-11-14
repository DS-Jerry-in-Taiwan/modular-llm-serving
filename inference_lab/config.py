"""Validated immutable experiment configuration."""
import json
from pathlib import Path
import re

SCHEMA_VERSION = 1


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported configuration schema_version")
    for key in ("experiment_id", "model", "runtime", "resources", "budgets"):
        if key not in config:
            raise ValueError(f"missing {key}")
    model = config["model"]
    for key in ("repository", "revision", "filename", "sha256", "size_bytes"):
        if key not in model:
            raise ValueError(f"missing model.{key}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repository"]):
        raise ValueError("model repository must be owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("model revision must be immutable 40-character commit")
    if not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
        raise ValueError("model sha256 must be 64 hexadecimal characters")
    if not isinstance(model["filename"], str) or Path(model["filename"]).name != model["filename"]:
        raise ValueError("model filename must be a basename")
    for group, key in ((model, "size_bytes"), (config["resources"], "disk_reserve_bytes"),
                       (config["budgets"], "download_bytes"), (config["budgets"], "smoke_seconds"),
                       (config["budgets"], "experiment_seconds")):
        if type(group.get(key)) is not int or group[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if model["size_bytes"] > config["budgets"]["download_bytes"]:
        raise ValueError("model exceeds download budget")
    if not re.fullmatch(r"[0-9a-f]{40}", config["runtime"].get("revision", "")):
        raise ValueError("runtime revision must be immutable")
    for key in ("context_size", "gpu_layers", "threads", "parallel"):
        if type(config["runtime"].get(key)) is not int or config["runtime"][key] <= 0:
            raise ValueError(f"runtime.{key} must be a positive integer")
    if config["runtime"]["parallel"] != 1:
        raise ValueError("only one inference slot is supported")
    return config


