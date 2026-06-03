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

**Record composition** (Story 9.5, 2026-05-21 enumeration). ``records``
holds 102 entries:

- **2 anchor records** retained verbatim from the v2 minimum-viable lock
  (one N2 wild-type ``1031550``, one MT1078 mutant ``1029149``). These
  are intentionally **not** part of the 100-record stratified subset —
  they remain separate anchors.
- **100 stratified records** grouped into 5 contiguous blocks of 20, one
  block per strain, in ``sort=mostrecent`` order: N2 (wild-type) +
  CB1112 ``cat-2(e1112)`` + MT2611 ``unc-8(n491n1192)`` + RB557
  ``asic-2(ok289)`` + NL1137 ``gpa-5(pk376)``. The 4 mutants are distinct
  from the OpenWormMovementDB pick (Story 9.6) to maximize genotype
  coverage across the WBDB / OWMD union. Record IDs are the
  pre-registration commitments enumerated in the Story 9.5 file
  (``_bmad-output/implementation-artifacts/
  9-5-wbdb-full-100-record-extension-original-story-8-5-ac.md``).

Per-record ``sha256`` is intentionally omitted (defaults to ``""``); it
is excluded from the ``zenodo_subset`` canonicalization and is populated
later from the canonical bytes-on-disk once records are fetched.
"""

from wormjepa.data.sources.base import ZenodoRecordPin, ZenodoSubsetSource

SPEC = ZenodoSubsetSource(
    name="wormbehavior_db",
    records=[
        # ------------------------------------------------------------------
        # Anchor records — retained verbatim from the v2 minimum-viable lock.
        # Intentionally NOT counted in the stratified 100 (per Story 9.5).
        # ------------------------------------------------------------------
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
        # ------------------------------------------------------------------
        # N2 (wild-type, Bristol) — 20 records, Story 9.5 extension.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1031915",
            doi="10.5281/zenodo.1031915",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032583",
            doi="10.5281/zenodo.1032583",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031845",
            doi="10.5281/zenodo.1031845",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032039",
            doi="10.5281/zenodo.1032039",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032351",
            doi="10.5281/zenodo.1032351",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032267",
            doi="10.5281/zenodo.1032267",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032187",
            doi="10.5281/zenodo.1032187",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1033043",
            doi="10.5281/zenodo.1033043",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032397",
            doi="10.5281/zenodo.1032397",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031859",
            doi="10.5281/zenodo.1031859",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031626",
            doi="10.5281/zenodo.1031626",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032667",
            doi="10.5281/zenodo.1032667",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032899",
            doi="10.5281/zenodo.1032899",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032911",
            doi="10.5281/zenodo.1032911",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032739",
            doi="10.5281/zenodo.1032739",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032733",
            doi="10.5281/zenodo.1032733",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032486",
            doi="10.5281/zenodo.1032486",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1011381",
            doi="10.5281/zenodo.1011381",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1015022",
            doi="10.5281/zenodo.1015022",
            description="N2 wild-type — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1033115",
            doi="10.5281/zenodo.1033115",
            description="N2 wild-type — story 9-5 extension",
        ),
        # ------------------------------------------------------------------
        # CB1112 cat-2(e1112) — dopamine-deficient — 20 records.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1192725",
            doi="10.5281/zenodo.1192725",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1201958",
            doi="10.5281/zenodo.1201958",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1201922",
            doi="10.5281/zenodo.1201922",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1192973",
            doi="10.5281/zenodo.1192973",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1201785",
            doi="10.5281/zenodo.1201785",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1201938",
            doi="10.5281/zenodo.1201938",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1192898",
            doi="10.5281/zenodo.1192898",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1200505",
            doi="10.5281/zenodo.1200505",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1189908",
            doi="10.5281/zenodo.1189908",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1192693",
            doi="10.5281/zenodo.1192693",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1192947",
            doi="10.5281/zenodo.1192947",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1201844",
            doi="10.5281/zenodo.1201844",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1203373",
            doi="10.5281/zenodo.1203373",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1192961",
            doi="10.5281/zenodo.1192961",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1202028",
            doi="10.5281/zenodo.1202028",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1201944",
            doi="10.5281/zenodo.1201944",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1191658",
            doi="10.5281/zenodo.1191658",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1201776",
            doi="10.5281/zenodo.1201776",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1192782",
            doi="10.5281/zenodo.1192782",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1190130",
            doi="10.5281/zenodo.1190130",
            description="CB1112 cat-2(e1112) — story 9-5 extension",
        ),
        # ------------------------------------------------------------------
        # MT2611 unc-8(n491n1192) — degenerin channel — 20 records.
        # (Thin pool — 27 total, zero margin.)
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1021469",
            doi="10.5281/zenodo.1021469",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029192",
            doi="10.5281/zenodo.1029192",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1027263",
            doi="10.5281/zenodo.1027263",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1023820",
            doi="10.5281/zenodo.1023820",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1028783",
            doi="10.5281/zenodo.1028783",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1030513",
            doi="10.5281/zenodo.1030513",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025364",
            doi="10.5281/zenodo.1025364",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025461",
            doi="10.5281/zenodo.1025461",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1022956",
            doi="10.5281/zenodo.1022956",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1028337",
            doi="10.5281/zenodo.1028337",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1030110",
            doi="10.5281/zenodo.1030110",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1027711",
            doi="10.5281/zenodo.1027711",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031586",
            doi="10.5281/zenodo.1031586",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029015",
            doi="10.5281/zenodo.1029015",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1015234",
            doi="10.5281/zenodo.1015234",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1013495",
            doi="10.5281/zenodo.1013495",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1020789",
            doi="10.5281/zenodo.1020789",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025877",
            doi="10.5281/zenodo.1025877",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1026445",
            doi="10.5281/zenodo.1026445",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025266",
            doi="10.5281/zenodo.1025266",
            description="MT2611 unc-8(n491n1192) — story 9-5 extension",
        ),
        # ------------------------------------------------------------------
        # RB557 asic-2(ok289) — acid-sensing ion channel — 20 records.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1033913",
            doi="10.5281/zenodo.1033913",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031552",
            doi="10.5281/zenodo.1031552",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031528",
            doi="10.5281/zenodo.1031528",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031632",
            doi="10.5281/zenodo.1031632",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1030361",
            doi="10.5281/zenodo.1030361",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018049",
            doi="10.5281/zenodo.1018049",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029876",
            doi="10.5281/zenodo.1029876",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031404",
            doi="10.5281/zenodo.1031404",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031300",
            doi="10.5281/zenodo.1031300",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031682",
            doi="10.5281/zenodo.1031682",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1022804",
            doi="10.5281/zenodo.1022804",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025437",
            doi="10.5281/zenodo.1025437",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1022179",
            doi="10.5281/zenodo.1022179",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1030699",
            doi="10.5281/zenodo.1030699",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031432",
            doi="10.5281/zenodo.1031432",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029250",
            doi="10.5281/zenodo.1029250",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029575",
            doi="10.5281/zenodo.1029575",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1029055",
            doi="10.5281/zenodo.1029055",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031056",
            doi="10.5281/zenodo.1031056",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1028485",
            doi="10.5281/zenodo.1028485",
            description="RB557 asic-2(ok289) — story 9-5 extension",
        ),
        # ------------------------------------------------------------------
        # NL1137 gpa-5(pk376) — G-protein alpha, chemosensory — 20 records.
        # ------------------------------------------------------------------
        ZenodoRecordPin(
            zenodo_record_id="1032005",
            doi="10.5281/zenodo.1032005",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1031000",
            doi="10.5281/zenodo.1031000",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032407",
            doi="10.5281/zenodo.1032407",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032233",
            doi="10.5281/zenodo.1032233",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032151",
            doi="10.5281/zenodo.1032151",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1032107",
            doi="10.5281/zenodo.1032107",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1021726",
            doi="10.5281/zenodo.1021726",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1022862",
            doi="10.5281/zenodo.1022862",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1012100",
            doi="10.5281/zenodo.1012100",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018201",
            doi="10.5281/zenodo.1018201",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1013164",
            doi="10.5281/zenodo.1013164",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1020469",
            doi="10.5281/zenodo.1020469",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1025290",
            doi="10.5281/zenodo.1025290",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018245",
            doi="10.5281/zenodo.1018245",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1022412",
            doi="10.5281/zenodo.1022412",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1019359",
            doi="10.5281/zenodo.1019359",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1015543",
            doi="10.5281/zenodo.1015543",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1026711",
            doi="10.5281/zenodo.1026711",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1024784",
            doi="10.5281/zenodo.1024784",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
        ZenodoRecordPin(
            zenodo_record_id="1018643",
            doi="10.5281/zenodo.1018643",
            description="NL1137 gpa-5(pk376) — story 9-5 extension",
        ),
    ],
    license="CC-BY-4.0",
    citation="yemini_brown_2013",
    redistribution_restrictions=(
        "Zenodo hosts per-experiment records; no project-side redistribution."
    ),
)
