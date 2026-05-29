"""Canonicalization of frozen artifacts for SHA-stable hashing.

Each method produces a byte representation of an artifact's *semantic* content,
stripped of incidental formatting variation (key order, line endings, comments,
whitespace). The lock script hashes the canonical bytes with SHA-256.

Schema version 1 methods (file-path based):

- ``yaml_sorted_keys_lf``  — YAML re-emitted with sorted keys + LF newlines.
- ``python_ast_normalized``  — Python parsed via :mod:`ast` and re-emitted with
  :func:`ast.unparse`. Whitespace-only diffs produce identical hashes.
- ``doi_manifest``  — JSON metadata for a DOI-pinned dataset (single DOI).
- ``text_lf``  — Plain text, LF-normalized, trailing-whitespace stripped.

Schema version 2 methods (structured-data based; the data lives in the
``data/sources/<name>`` SPEC module, not on disk):

- ``dandi_federation``  — Multi-dandiset corpora. The canonical bytes are the
  sorted-key JSON of the dandiset list. Used by WormID (7 federated dandisets).
- ``zenodo_subset``  — Per-experiment Zenodo archives where a specific subset
  is pre-committed. The canonical bytes are the sorted-key JSON of the chosen
  records. Used by WormBehaviorDB and OpenWormMovementDB.
- ``github_commit_pin``  — Generator or code-only repos where bytes lack a DOI
  but commits + configs do. The canonical bytes are the sorted-key JSON of
  ``{repo, commit_sha, config_path, config_sha256}``. Used by BAAIWorm.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml

CanonicalizationMethod = Literal[
    # Schema v1 (file-path based):
    "yaml_sorted_keys_lf",
    "python_ast_normalized",
    "doi_manifest",
    "text_lf",
    # Schema v2 (structured-data based):
    "dandi_federation",
    "zenodo_subset",
    "github_commit_pin",
]


def canonicalize_yaml(path: Path) -> bytes:
    """Read a YAML file and return canonical UTF-8 bytes.

    Canonical form: ``yaml.safe_dump(..., sort_keys=True, default_flow_style=False)``
    with LF line endings. Comments are dropped (semantic content only).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return b""
    text = yaml.safe_dump(
        data,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        line_break="\n",
    )
    # yaml.safe_dump always uses LF when line_break="\n"; ensure trailing \n only.
    return text.encode("utf-8")


def canonicalize_text(path: Path) -> bytes:
    """Canonicalize a plain-text file (e.g., Markdown) for SHA-stable hashing.

    Canonical form: UTF-8, LF line endings (CRLF/CR converted), trailing
    whitespace stripped per line, final trailing newline. Used for
    human-readable frozen documents (PRE-REGISTRATION.md, CITATIONS.bib).
    """
    raw = path.read_text(encoding="utf-8")
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in normalized.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    text = "\n".join(lines) + "\n"
    return text.encode("utf-8")


def canonicalize_python(path: Path) -> bytes:
    """Read a Python file and return ``ast.unparse``-normalized UTF-8 bytes.

    Whitespace-only or comment-only edits leave the AST unchanged, so the
    canonical hash stays stable across formatter churn.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    canonical = ast.unparse(tree)
    if not canonical.endswith("\n"):
        canonical += "\n"
    return canonical.encode("utf-8")


def canonicalize_doi_manifest(path: Path) -> bytes:
    """Read a JSON DOI-metadata file and return canonical UTF-8 bytes.

    Canonical form: ``json.dumps(..., sort_keys=True, separators=(',', ':'))``
    + trailing LF. Plain-string DOI strings (without a wrapping JSON object)
    are accepted and canonicalized as a single-field document.
    """
    raw = path.read_text(encoding="utf-8").strip()
    return canonicalize_doi_string(raw) if raw else b""


def canonicalize_doi_string(doi_or_json: str) -> bytes:
    """Canonicalize a bare DOI string or pre-fetched DOI metadata JSON.

    Used by :func:`wormjepa.manifest.lock_check.verify_manifest` to recompute
    a dataset entry's hash from the current ``data/sources/<name>.SPEC.doi``
    in-memory (no scratch file needed).
    """
    if not doi_or_json:
        return b""
    try:
        data: object = json.loads(doi_or_json)
    except json.JSONDecodeError:
        data = {"doi": doi_or_json}
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def canonicalize_dandi_federation(dandisets: list[dict[str, str]]) -> bytes:
    """Canonicalize a federated set of DANDI dandiset pins.

    Each input dict must contain ``dandiset_id``, ``version``, ``doi``. The
    list is sorted by ``dandiset_id`` and re-emitted as compact sorted-key
    JSON. Used for WormID, whose corpus is 7 federated dandisets.
    """
    normalized = sorted(
        (
            {"dandiset_id": d["dandiset_id"], "version": d["version"], "doi": d["doi"]}
            for d in dandisets
        ),
        key=lambda d: d["dandiset_id"],
    )
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def canonicalize_zenodo_subset(records: list[dict[str, str]]) -> bytes:
    """Canonicalize a pre-committed subset of per-experiment Zenodo records.

    Each input dict must contain ``zenodo_record_id``, ``doi``; ``description``
    is optional and included if present. The list is sorted by
    ``zenodo_record_id`` and re-emitted as compact sorted-key JSON. Used for
    archives like WormBehaviorDB and OpenWormMovementDB where the canonical
    "dataset" is a chosen subset of per-experiment records.
    """

    def _entry(r: dict[str, str]) -> dict[str, str]:
        e = {"zenodo_record_id": r["zenodo_record_id"], "doi": r["doi"]}
        if r.get("description"):
            e["description"] = r["description"]
        return e

    normalized = sorted((_entry(r) for r in records), key=lambda r: r["zenodo_record_id"])
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def canonicalize_github_commit_pin(
    repo: str,
    commit_sha: str,
    config_path: str,
    config_sha256: str,
) -> bytes:
    """Canonicalize a GitHub commit + generator-config pin.

    Used for code-only releases where bytes have no DOI but the commit and
    generator configuration do. Emitted as compact sorted-key JSON of
    ``{repo, commit_sha, config_path, config_sha256}``.
    """
    payload = {
        "repo": repo,
        "commit_sha": commit_sha,
        "config_path": config_path,
        "config_sha256": config_sha256,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


_PATH_BASED: frozenset[CanonicalizationMethod] = frozenset(
    {"yaml_sorted_keys_lf", "python_ast_normalized", "doi_manifest", "text_lf"}
)
"""Canonicalization methods that operate on a file path.

Methods outside this set (``dandi_federation``, ``zenodo_subset``,
``github_commit_pin``) accept structured data and are dispatched separately
by callers in :mod:`wormjepa.manifest.lock_check`.
"""


def sha256_of_canonicalized(path: Path, method: CanonicalizationMethod) -> str:
    """Compute the lowercase hex SHA-256 of ``path`` canonicalized via ``method``.

    Only file-based canonicalizations are supported here. Schema v2 methods
    (``dandi_federation``, ``zenodo_subset``, ``github_commit_pin``) operate
    on structured data and must use the per-method helper functions directly.
    """
    if method == "yaml_sorted_keys_lf":
        canonical = canonicalize_yaml(path)
    elif method == "python_ast_normalized":
        canonical = canonicalize_python(path)
    elif method == "doi_manifest":
        canonical = canonicalize_doi_manifest(path)
    elif method == "text_lf":
        canonical = canonicalize_text(path)
    elif method in {"dandi_federation", "zenodo_subset", "github_commit_pin"}:
        msg = (
            f"Canonicalization method {method!r} is structured-data-based and "
            f"does not accept a file path. Use the per-method canonicalize_* "
            f"function directly."
        )
        raise ValueError(msg)
    else:  # pragma: no cover  # Literal enforced by type checker
        msg = f"Unknown canonicalization method: {method!r}"
        raise ValueError(msg)
    return hashlib.sha256(canonical).hexdigest()
