# JEPA-WORM

[![CI](https://github.com/samirobbinsss/JEPA-WORM/actions/workflows/test.yml/badge.svg)](https://github.com/samirobbinsss/JEPA-WORM/actions/workflows/test.yml)

**Phase 0**: a self-contained software milestone that validates the JEPA-WORM thesis on public *C. elegans* data alone, before any hardware or biological procurement.

This codebase produces a numerical, pre-registered answer to one question:

> Can a self-supervised vision-only predictor (V-JEPA 2.1-style) yield a latent
> that is both behaviorally predictive (forecasts future body posture) and
> neurally meaningful (linearly decodes held-out head-neuron calcium activity)?

Either outcome — thresholds cleared or kill-criterion fired — is publishable. The full Phase 0 commitments are frozen in `pre-registration/PRE-REGISTRATION.md`.

## Quick orientation

- **Pre-registration commitments**: `pre-registration/PRE-REGISTRATION.md` — the frozen thresholds, datasets, probes, and kill criteria (SHA-locked in `pre-registration/MANIFEST.lock`)
- **Reproduction guide**: `REPRODUCE.md`
- **Frozen-artifact change log**: `CHANGELOG.md`
- **Citations**: `CITATIONS.bib`

## Install (development)

```bash
uv sync
uv run pre-commit install
```

## Run

```bash
uv run wormjepa --help
uv run wormjepa --version
```

Subcommands: `run` (train), `eval` (gate-evaluation orchestrator), `report` (CI-aware comparison against published results), `preregister` (verify the SHA-locked manifest), and `fetch` (download pre-registered Zenodo anchors). See `REPRODUCE.md` for the canonical reproduction sequence.

### Notes

On macOS, `wormjepa` prints an `objc[]: Class AVF... is implemented in both .../av/.../libavdevice.dylib and .../cv2/.../libavdevice.dylib` warning at startup. Both `av` and `opencv-python` ship their own copy of libavdevice; the warning is cosmetic and does not affect correctness.

## Tooling

- **Package management**: [uv](https://docs.astral.sh/uv/) (lockfile-pinned)
- **Linting / formatting**: [ruff](https://docs.astral.sh/ruff/) (strict)
- **Type checking**: [pyright](https://github.com/microsoft/pyright) (strict)
- **Tests**: pytest (unit + smoke + integration)
- **Pre-commit hooks**: ruff, pyright, MANIFEST.lock check (Epic 4), CHANGELOG-frozen enforcement (Epic 4)

## Reproducibility

Reproduction tolerance is the worm-level bootstrap CI of each reported number — not bit-exact (NFR7). The repo's `REPRODUCE.md` (after Epic 7) documents the canonical reproduction sequence; `wormjepa report --compare <other-results>` verifies a local reproduction.

## License

See `data/SOURCES.md` for per-dataset licensing. Code: MIT (see `LICENSE`). To cite this repository: see `CITATION.cff` (GitHub renders a "Cite this repository" button).
