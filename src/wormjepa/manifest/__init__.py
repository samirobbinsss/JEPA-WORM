"""Pre-registration manifest infrastructure for JEPA-WORM.

The manifest mechanism is the structural keystone of the project's pre-registration
discipline (Epic 4). Every reportable computation downstream consults
``pre-registration/MANIFEST.lock`` to confirm that frozen artifacts have not
drifted; SHA mismatches surface as :class:`wormjepa.PreRegistrationViolation`.
"""

from wormjepa.manifest.canonicalize import (
    CanonicalizationMethod,
    canonicalize_doi_manifest,
    canonicalize_doi_string,
    canonicalize_python,
    canonicalize_text,
    canonicalize_yaml,
    sha256_of_canonicalized,
)

__all__ = [
    "CanonicalizationMethod",
    "canonicalize_doi_manifest",
    "canonicalize_doi_string",
    "canonicalize_python",
    "canonicalize_text",
    "canonicalize_yaml",
    "sha256_of_canonicalized",
]
