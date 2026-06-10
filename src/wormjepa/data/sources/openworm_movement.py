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

**Strain pick** (Story 9.6, project-lead signed off): N2 wild-type +
``CB120`` (unc-4) / ``ED3054`` (*C. briggsae* wild isolate) / ``RB1883``
(W03B1.2 KO) / ``MT1078`` (egl-13). The four mutants share no strain code
with the WBDB pick (``CB1112`` / ``MT2611`` / ``RB557`` / ``NL1137``), and
every record ID below is disjoint from the WBDB 100 and the two WBDB
anchors (``1031550``, ``1029149``).

**Record composition.**

- The ``N2`` anchor ``1031550`` is retained as a **separate anchor**
  outside the 100-record stratified subset. ``1031550`` is also pinned by
  ``wormbehavior_db.py`` as its N2 anchor; keeping it as a separate anchor
  here (rather than inside the OWMD 100) avoids a record-ID collision with
  the WBDB stratified set.
- 100 stratified records grouped into 5 contiguous blocks of 20, one block
  per strain, in ``sort=mostrecent`` order (Story 9.6 enumeration).

**Dropped video-less mutant anchor ``1033265``.** The prior live SPEC
pinned ``1033265`` as an RB1883 mutant anchor. The 2026-05-29 raw-video
probe (Story 9.6) confirmed that record is video-less (12 MB off-food,
skeleton/features only); ``_schafer_hdf5.py`` would silently skip it.
It is dropped entirely — RB1883 already contributes 20 video-bearing
records inside the 100, so no replacement anchor is added. The RB1883
block also swaps the two original video-less picks (``1033265`` +
``1033151``) for video-confirmed ``1020621`` + ``1005780``.

**SHA convention.** ``ZenodoRecordPin.sha256`` is omitted (defaults to
``""``); the canonical per-record file SHAs are filled by
``wormjepa preregister --force`` once the bytes-on-disk are fetched.
``description`` is excluded from the ``zenodo_subset`` canonicalization
(the hash is over ``(zenodo_record_id, doi)`` tuples).
"""

from wormjepa.data.sources.base import ZenodoRecordPin, ZenodoSubsetSource

SPEC = ZenodoSubsetSource(
    name="openworm_movement",
    records=[
        # ------------------------------------------------------------------
        # Anchor record — N2 wild-type 1031550, retained verbatim from the
        # prior live SPEC. Kept as a SEPARATE anchor OUTSIDE the 100
        # stratified records (avoids a record-ID collision with the WBDB
        # separate-anchor holding of the same ID). The prior mutant anchor
        # 1033265 is DROPPED (video-less, 2026-05-29 probe) with no
        # replacement.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1031550",
            doi="10.5281/zenodo.1031550",
            description="N2 Schafer Lab wild-type (Bristol, UK), 2010-01-26 — anchor",
        ),
        # ------------------------------------------------------------------
        # N2 (wild-type, Bristol) — 20 records, story 9-6 extension.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1033159",
            doi="10.5281/zenodo.1033159",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1033163",
            doi="10.5281/zenodo.1033163",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032123",
            doi="10.5281/zenodo.1032123",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031879",
            doi="10.5281/zenodo.1031879",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032500",
            doi="10.5281/zenodo.1032500",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032327",
            doi="10.5281/zenodo.1032327",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032915",
            doi="10.5281/zenodo.1032915",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032803",
            doi="10.5281/zenodo.1032803",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032935",
            doi="10.5281/zenodo.1032935",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032933",
            doi="10.5281/zenodo.1032933",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032645",
            doi="10.5281/zenodo.1032645",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032975",
            doi="10.5281/zenodo.1032975",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032773",
            doi="10.5281/zenodo.1032773",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032885",
            doi="10.5281/zenodo.1032885",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032795",
            doi="10.5281/zenodo.1032795",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032907",
            doi="10.5281/zenodo.1032907",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032851",
            doi="10.5281/zenodo.1032851",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032941",
            doi="10.5281/zenodo.1032941",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032133",
            doi="10.5281/zenodo.1032133",
            description="N2 wild-type — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032333",
            doi="10.5281/zenodo.1032333",
            description="N2 wild-type — story 9-6 extension",
        ),
        # ------------------------------------------------------------------
        # CB120 unc-4(e120) — cholinergic motor neuron fate — 20 records.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1019906",
            doi="10.5281/zenodo.1019906",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1023309",
            doi="10.5281/zenodo.1023309",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1019876",
            doi="10.5281/zenodo.1019876",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1015174",
            doi="10.5281/zenodo.1015174",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1014707",
            doi="10.5281/zenodo.1014707",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1014606",
            doi="10.5281/zenodo.1014606",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1023734",
            doi="10.5281/zenodo.1023734",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1026577",
            doi="10.5281/zenodo.1026577",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1019070",
            doi="10.5281/zenodo.1019070",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018872",
            doi="10.5281/zenodo.1018872",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1017633",
            doi="10.5281/zenodo.1017633",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031696",
            doi="10.5281/zenodo.1031696",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1030934",
            doi="10.5281/zenodo.1030934",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1026619",
            doi="10.5281/zenodo.1026619",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1023621",
            doi="10.5281/zenodo.1023621",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018964",
            doi="10.5281/zenodo.1018964",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1030056",
            doi="10.5281/zenodo.1030056",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1026310",
            doi="10.5281/zenodo.1026310",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1033763",
            doi="10.5281/zenodo.1033763",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1020583",
            doi="10.5281/zenodo.1020583",
            description="CB120 unc-4(e120) — story 9-6 extension",
        ),
        # ------------------------------------------------------------------
        # ED3054 — C. briggsae wild isolate (Nairobi, Kenya) — 20 records.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1016521",
            doi="10.5281/zenodo.1016521",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1014924",
            doi="10.5281/zenodo.1014924",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1011645",
            doi="10.5281/zenodo.1011645",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1014865",
            doi="10.5281/zenodo.1014865",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1017370",
            doi="10.5281/zenodo.1017370",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018420",
            doi="10.5281/zenodo.1018420",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018009",
            doi="10.5281/zenodo.1018009",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1019283",
            doi="10.5281/zenodo.1019283",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1016952",
            doi="10.5281/zenodo.1016952",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1016599",
            doi="10.5281/zenodo.1016599",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018207",
            doi="10.5281/zenodo.1018207",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1015857",
            doi="10.5281/zenodo.1015857",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018367",
            doi="10.5281/zenodo.1018367",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1016845",
            doi="10.5281/zenodo.1016845",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1017572",
            doi="10.5281/zenodo.1017572",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018129",
            doi="10.5281/zenodo.1018129",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1017645",
            doi="10.5281/zenodo.1017645",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018422",
            doi="10.5281/zenodo.1018422",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018199",
            doi="10.5281/zenodo.1018199",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1011989",
            doi="10.5281/zenodo.1011989",
            description="ED3054 C. briggsae wild isolate — story 9-6 extension",
        ),
        # ------------------------------------------------------------------
        # RB1883 W03B1.2(ok2433)IV — uncharacterised gene KO — 20 records.
        # Revised 2026-05-29: swapped video-less 1033265 + 1033151 for
        # video-bearing 1020621 + 1005780 (raw-video probe).
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1020621",
            doi="10.5281/zenodo.1020621",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1005780",
            doi="10.5281/zenodo.1005780",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1013246",
            doi="10.5281/zenodo.1013246",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1019793",
            doi="10.5281/zenodo.1019793",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1015466",
            doi="10.5281/zenodo.1015466",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1016984",
            doi="10.5281/zenodo.1016984",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018671",
            doi="10.5281/zenodo.1018671",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1020653",
            doi="10.5281/zenodo.1020653",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029930",
            doi="10.5281/zenodo.1029930",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1021495",
            doi="10.5281/zenodo.1021495",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1023173",
            doi="10.5281/zenodo.1023173",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025415",
            doi="10.5281/zenodo.1025415",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1027687",
            doi="10.5281/zenodo.1027687",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1023133",
            doi="10.5281/zenodo.1023133",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1013083",
            doi="10.5281/zenodo.1013083",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1011852",
            doi="10.5281/zenodo.1011852",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1011796",
            doi="10.5281/zenodo.1011796",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1014891",
            doi="10.5281/zenodo.1014891",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1011957",
            doi="10.5281/zenodo.1011957",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1014152",
            doi="10.5281/zenodo.1014152",
            description="RB1883 W03B1.2(ok2433)IV — story 9-6 extension",
        ),
        # ------------------------------------------------------------------
        # MT1078 egl-13(n483)X — Sox-family TF, egg-laying defective —
        # 20 records. Excludes the WBDB MT1078 anchor 1029149.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1025567",
            doi="10.5281/zenodo.1025567",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1021337",
            doi="10.5281/zenodo.1021337",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1028766",
            doi="10.5281/zenodo.1028766",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1021391",
            doi="10.5281/zenodo.1021391",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1027626",
            doi="10.5281/zenodo.1027626",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018595",
            doi="10.5281/zenodo.1018595",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1026589",
            doi="10.5281/zenodo.1026589",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1024562",
            doi="10.5281/zenodo.1024562",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025080",
            doi="10.5281/zenodo.1025080",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1027624",
            doi="10.5281/zenodo.1027624",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029252",
            doi="10.5281/zenodo.1029252",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031230",
            doi="10.5281/zenodo.1031230",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025827",
            doi="10.5281/zenodo.1025827",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1028719",
            doi="10.5281/zenodo.1028719",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1028172",
            doi="10.5281/zenodo.1028172",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029745",
            doi="10.5281/zenodo.1029745",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1027599",
            doi="10.5281/zenodo.1027599",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1013222",
            doi="10.5281/zenodo.1013222",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025465",
            doi="10.5281/zenodo.1025465",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1008463",
            doi="10.5281/zenodo.1008463",
            description="MT1078 egl-13(n483)X — story 9-6 extension",
        ),
    ],
    license="CC-BY-4.0",
    citation="javer-openworm-2018",
    redistribution_restrictions=(
        "Zenodo hosts per-experiment records; no project-side redistribution."
    ),
)
