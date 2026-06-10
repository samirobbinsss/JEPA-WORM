"""WormID corpus source spec (DANDI federation of 7 dandisets).

The WormID corpus per Sprague et al. (2025, Cell Reports Methods) is a
federation of seven DANDI dandisets contributed by five labs, harmonised
into a unified NWB-format corpus of 118 worms.

The lock canonicalization is :data:`dandi_federation`: SHA-256 over the
sorted-key JSON of the dandiset list (dandiset_id, version, doi).
Adding, removing, or version-bumping any dandiset is a substantive
frozen-artifact change requiring a CHANGELOG entry per FR48 / NFR14.

The per-lab train/eval cohort assignment is recorded separately in
``pre-registration/splits/wormid_train_eval.yaml``. SF (000776, the only
freely-moving lab) and HL (000714) are the eval cohort; the remaining
five dandisets across four labs (NP, EY, SK1+SK2, KK) are the train
cohort. See ``pre-registration/PRE-REGISTRATION.md`` §"Lock History
and Rollback (2026-05-14)" for rationale.
"""

from wormjepa.data.sources.base import DandiFederationSource, DandisetPin

SPEC = DandiFederationSource(
    name="wormid",
    dandisets=[
        DandisetPin(
            dandiset_id="000472",
            version="0.241009.1502",
            doi="10.48324/dandi.000472/0.241009.1502",
        ),  # SK2 (SK lab, immobilized microfluidic, 10 worms) — train
        DandisetPin(
            dandiset_id="000541",
            version="0.241009.1457",
            doi="10.48324/dandi.000541/0.241009.1457",
        ),  # EY (immobilized microfluidic, 21 worms) — train
        DandisetPin(
            dandiset_id="000565",
            version="0.241009.1504",
            doi="10.48324/dandi.000565/0.241009.1504",
        ),  # SK1 (SK lab, immobilized microfluidic, 21 worms) — train
        DandisetPin(
            dandiset_id="000692",
            version="0.240402.2118",
            doi="10.48324/dandi.000692/0.240402.2118",
        ),  # KK (semi-restricted microfluidic, 9 worms) — train
        DandisetPin(
            dandiset_id="000714",
            version="0.241009.1516",
            doi="10.48324/dandi.000714/0.241009.1516",
        ),  # HL (immobilized microfluidic, 9 worms) — eval
        DandisetPin(
            dandiset_id="000715",
            version="0.241009.1514",
            doi="10.48324/dandi.000715/0.241009.1514",
        ),  # NP / Original NeuroPAL (immobilized microfluidic, 10 worms) — train
        DandisetPin(
            dandiset_id="000776",
            version="0.241009.1509",
            doi="10.48324/dandi.000776/0.241009.1509",
        ),  # SF (freely moving, 38 worms) — eval
    ],
    license="CC-BY-4.0",
    citation="sprague-wormid-2025",
    redistribution_restrictions=(
        "DANDI archive provides canonical hosting; no project-side redistribution."
    ),
)
