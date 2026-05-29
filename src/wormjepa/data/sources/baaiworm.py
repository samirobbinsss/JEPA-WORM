"""BAAIWorm synthetic-generator source spec.

BAAIWorm (Zhao et al. 2024, *Nature Computational Science*) is a code-only
release on GitHub, **not** a payload with a DOI. The canonical pin is
the repository's `v1.0.0` tag at commit
``699d3ddb4c2d6a7dc2ac72c3574aee01673fc934``. The generator configuration
file is ``Metaworm/args/run_worm_swim_args.txt``; its canonical SHA-256
(via ``text_lf`` canonicalization) is recorded below.

The lock canonicalization is :data:`github_commit_pin`: SHA-256 over the
sorted-key JSON of ``{repo, commit_sha, config_path, config_sha256}``.
Bumping the commit, config path, or config content is a substantive
frozen-artifact change requiring a CHANGELOG entry per FR48 / NFR14.
"""

from wormjepa.data.sources.base import GithubGeneratorSource

SPEC = GithubGeneratorSource(
    name="baaiworm",
    repo="github.com/Jessie940611/BAAIWorm",
    commit_sha="699d3ddb4c2d6a7dc2ac72c3574aee01673fc934",  # v1.0.0
    config_path="Metaworm/args/run_worm_swim_args.txt",
    config_sha256="bbc9fa9bdd08982cd6506a7148fbb70e321eb04ad774b026fe2a0d877cc2845b",
    license="See repository LICENSE (MIT-style, per upstream).",
    citation="zhao_baaiworm_2024",
    redistribution_restrictions=(
        "GitHub is the canonical source; no project-side redistribution. "
        "Reproducers fetch by commit SHA via the standard git checkout flow."
    ),
)
