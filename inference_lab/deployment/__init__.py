"""Fail-closed local deployment identity and manifest seam.

This package records the host and GPU observations needed before starting the
local inference service.  It is deliberately independent of serving and
evaluation: a blocked manifest is evidence that deployment must not proceed,
not a claim that a GPU run was attempted.
"""

from .manifest import (
    DEPLOYMENT_SCHEMA_VERSION,
    ManifestError,
    build_manifest,
    load_manifest,
    manifest_digest,
    validate_manifest,
    write_manifest_new,
)

__all__ = [
    "DEPLOYMENT_SCHEMA_VERSION",
    "ManifestError",
    "build_manifest",
    "load_manifest",
    "manifest_digest",
    "validate_manifest",
    "write_manifest_new",
]
