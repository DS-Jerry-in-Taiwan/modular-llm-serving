"""Usefulness suites, repeated measurements, and reviewed comparisons."""

def __getattr__(name):
    # Keep the optional V1/V2 adapter out of the V1 measurement source
    # closure.  Import it lazily when a caller explicitly requests it.
    if name in __all__:
        from importlib import import_module
        module = import_module(".adapter", __name__)
        return getattr(module, name)
    raise AttributeError(name)

__all__ = [
    "AdapterError", "InferenceAdapter", "InferenceBackend", "RunIdentityConflict",
    "RunMetrics", "RunRecord", "RunRequest",
]
