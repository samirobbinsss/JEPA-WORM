"""WormBehavior Database source spec (per-experiment Zenodo subset).

The WormBehavior Database (Yemini / Brown, 2013) is a per-experiment
Zenodo archive — thousands of individual records, each with its own
DOI. The methodologically substantive Phase 0 commitment is therefore
a **pre-registered subset** of records, not a single corpus DOI.

**Stratification policy** (recorded in ``PRE-REGISTRATION.md`` §"Lock
History and Rollback (2026-05-14)"):

- 100 records total
- Stratified 20 records per strain across 5 strains (N2 wild-type +
  4 representative mutants)
- Strain selection complements OpenWormMovementDB (`openworm_movement.py`)
  so the union covers more genotype diversity

**Per-record enumeration is deferred to Story 8.5** (real loader wiring
for WormBehaviorDB). The minimum viable lock at v2 commit time includes
**two anchor records** (one N2 wild-type, one mutant) as the
substantive foundation; the remaining 98 records land via a
``### Frozen-artifact changes`` CHANGELOG entry when Story 8.5 fetches
the data. The stratification policy in PRE-REGISTRATION.md is the
binding methodological commitment that any future enumeration must
satisfy.
"""

from wormjepa.data.sources.base import ZenodoRecordPin, ZenodoSubsetSource

SPEC = ZenodoSubsetSource(
    name="wormbehavior_db",
    records=[
        # Anchor records — minimum viable lock at v2 commit.
        # Story 8.5 expands to 100 records under the stratification policy.
        ZenodoRecordPin(
            zenodo_record_id="1031550",
            doi="10.5281/zenodo.1031550",
            description="N2 Schafer Lab wild-type (Bristol, UK), 2010-01-26 — anchor",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029149",
            doi="10.5281/zenodo.1029149",
            description="MT1078 egl-13(n483)X mutant, 2010-07-16 — anchor",
        ),
    ],
    license="CC-BY-4.0",
    citation="yemini_brown_2013",
    redistribution_restrictions=(
        "Zenodo hosts per-experiment records; no project-side redistribution."
    ),
)
