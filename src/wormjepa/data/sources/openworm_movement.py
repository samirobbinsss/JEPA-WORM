"""Open Worm Movement Database source spec (per-experiment Zenodo subset).

The Open Worm Movement Database (Javer et al. 2018) overlaps in source
material with the WormBehavior Database (`wormbehavior_db.py`); both
draw from the Schafer-lab Zenodo deposits originating in Yemini 2013.
The PRD lists them as separate datasets, and this SPEC pins a distinct
record subset to preserve that separation.

**Stratification policy** (recorded in ``PRE-REGISTRATION.md`` §"Lock
History and Rollback (2026-05-14)"):

- 100 records total
- Stratified 20 records per strain across 5 strains (N2 wild-type +
  4 representative mutants, **different mutants** from
  ``wormbehavior_db.py`` so the union covers more genotype diversity)

**Per-record enumeration is deferred to Story 8.6** (real loader wiring
for OpenWormMovementDB). The minimum viable lock at v2 commit time
includes **two anchor records** (one N2 wild-type, one mutant) as the
substantive foundation; the remaining 98 records land via a
``### Frozen-artifact changes`` CHANGELOG entry when Story 8.6 fetches
the data.
"""

from wormjepa.data.sources.base import ZenodoRecordPin, ZenodoSubsetSource

SPEC = ZenodoSubsetSource(
    name="openworm_movement",
    records=[
        # Anchor records — minimum viable lock at v2 commit.
        # Story 8.6 expands to 100 records under the stratification policy.
        # Records distinct from wormbehavior_db.py to preserve dataset separation
        # under the PRD's enumeration.
        ZenodoRecordPin(
            zenodo_record_id="1031550",
            doi="10.5281/zenodo.1031550",
            description="N2 Schafer Lab wild-type (Bristol, UK), 2010-01-26 — anchor",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1033265",
            doi="10.5281/zenodo.1033265",
            description="RB1883 W03B1.2(ok2433)IV mutant, 2010-04-28 — anchor",
        ),
    ],
    license="CC-BY-4.0",
    citation="javer_openworm_2018",
    redistribution_restrictions=(
        "Zenodo hosts per-experiment records; no project-side redistribution."
    ),
)
