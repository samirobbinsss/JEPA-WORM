# Public Dataset Sources

This file is the canonical index of every public dataset JEPA-WORM Phase 0 consumes. Each row is populated by **Epic 2 Story 2.3** (`bmad-output/planning-artifacts/epics.md`). The Phase 0 codebase never redistributes dataset payloads (FR10); reproducers fetch from the canonical source via the download script.

| Dataset | Citation key | DOI | License | Redistribution restrictions | Used in |
|---|---|---|---|---|---|
| WormID / WormID-Bench (Sprague 2024–25) | TBD (Epic 2 Story 2.3) | TBD | TBD | TBD | Warm-start neural prior + headline-probe evaluation |
| Atanas/Flavell-2023 (Cell) | TBD | TBD | TBD | TBD | Behavioral-state labels for motif recovery |
| Randi 2023 functional connectivity atlas | TBD | TBD (OSF e2syt) | TBD | TBD | Phase D follow-up; not in Phase 0 MVP |
| Cook 2019 connectome (Nature) | TBD | TBD | TBD | TBD | Graph-prior auxiliary side-task |
| Stephens 2008 eigenworms | TBD | TBD (Princeton/Bialek GitHub) | TBD | TBD | Initialize 4D kinematic subspace |
| WormBehavior Database (Yemini/Brown 2013) | TBD | TBD | TBD | TBD | Posture pretraining augmentation |
| Open Worm Movement Database (Javer 2018) | TBD | TBD | TBD | TBD | Posture pretraining augmentation |
| BAAIWorm (Nat Comp Sci 2024) | TBD | TBD | TBD | TBD | Synthetic neural+body pretraining; ablated |

**Update protocol**: when a story adds or modifies a dataset row, the corresponding BibTeX entry must be added to `CITATIONS.bib` in the same commit.
