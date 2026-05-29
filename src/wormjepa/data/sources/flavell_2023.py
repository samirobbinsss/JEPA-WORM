"""Atanas/Flavell-2023 corpus source spec.

**Story 9.1 finding (2026-05-18):** The Zenodo record `10.5281/zenodo.8150515`
pinned here at v2 lock time contains **CePNEM model analysis artefacts +
ANTSUN segmentation neural-network weights**, not the raw paired
video + neural corpus the JEPA loader needs. The canonical raw corpus
for Atanas et al. 2023 *Cell* is **DANDI dandiset 000776** — the same
dandiset already pinned in the WormID `dandi_federation` SPEC under
the "SF" (Sprague-Flavell) label.

`Flavell-2023` is therefore **conceptually redundant with the WormID
000776 federation member**. Phase 0 had been treating it as a separate
fifth dataset; in reality the Zenodo record is the analysis companion
to that same dandiset.

Implication: Phase 0 cannot materialize Flavell-2023 raw data without
either (a) materializing WormID 000776, which was gamma-deferred on
2026-05-18 for disk-capacity + content-shape reasons (see
`_bmad-output/implementation-artifacts/phase0-wormid-content-discovery-2026-05-18.md`),
or (b) accepting the Zenodo analysis-only payload as "the dataset",
which fails the FR8 contract (no video).

**Story 9.1 resolution: defer Flavell-2023 to Phase 0 Growth via
gamma-pattern.** The SPEC's URL + DOI stay as committed (the Zenodo record
itself is a real published companion artefact and the lock SHA over it
is meaningful); only its materialization in Phase 0 active runs is
deferred. Phase 0 Growth resurrects Flavell-2023 alongside WormID/SF
materialization. Original docstring kept below for audit:

----

The brain-wide-representations dataset (Atanas et al. 2023, *Cell*) is
deposited on Zenodo, **not** DANDI, contrary to what the placeholder lock
v1 implied. The canonical record is Zenodo `10.5281/zenodo.8150515`. The
canonical paper is DOI `10.1016/j.cell.2023.07.035`; the primary access
point is the project's web interface at `www.wormwideweb.org`.

(That original framing is now known to be wrong on two counts: the
Zenodo record is analysis-only, and the raw corpus is on DANDI 000776,
i.e. WormID-SF.)
"""

from wormjepa.data.sources.base import DatasetSource

SPEC = DatasetSource(
    name="flavell_2023",
    url="https://zenodo.org/api/records/8150515/files-archive",
    dest_filename="flavell_2023_atanas_cell.zip",
    sha256="sha256-pending-real-download-story-8-4",
    doi="10.5281/zenodo.8150515",
    license="CC-BY-4.0",
    citation="atanas_flavell_2023",
    redistribution_restrictions=(
        "Zenodo hosts the canonical archive; no project-side redistribution."
    ),
)
