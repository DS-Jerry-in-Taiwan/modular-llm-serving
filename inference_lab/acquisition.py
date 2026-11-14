"""Bounded, checksum-verified acquisition of a configured public GGUF artifact."""
from __future__ import annotations

from pathlib import Path
import time
import os
import urllib.request

from .artifacts import sha256_file


def verify_model(path, model):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != model["size_bytes"]:
        raise ValueError("model size does not match pinned identity")
    digest = sha256_file(path)
    if digest != model["sha256"]:
        raise ValueError("model SHA-256 does not match pinned identity")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest}


def acquire_model(config, destination, timeout=1800):
    """No credentials; immutable public source only, atomic rename after validation."""
    model = config["model"]
    destination = Path(destination)
    if destination.exists():
        return verify_model(destination, model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError("partial download already exists; inspect before retry")
    url = f'https://huggingface.co/{model["repository"]}/resolve/{model["revision"]}/{model["filename"]}?download=true'
    deadline = time.monotonic() + timeout
    size = 0
    with urllib.request.urlopen(url, timeout=min(timeout, 30)) as response, partial.open("xb") as handle:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("model download deadline exceeded")
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > min(model["size_bytes"], config["budgets"]["download_bytes"]):
                raise ValueError("download exceeds pinned size or budget")
            handle.write(chunk)
    result = verify_model(partial, model)
    if destination.exists():
        raise FileExistsError("destination appeared during download")
    os.link(partial, destination)  # atomic no-replace publish on this filesystem
    partial.unlink()
    return {**result, "path": str(destination.resolve()), "source_url": url}
