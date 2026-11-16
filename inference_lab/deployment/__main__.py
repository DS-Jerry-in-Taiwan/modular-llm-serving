"""CLI entrypoint for ``python -m inference_lab.deployment``."""
from .preflight import main

raise SystemExit(main())
