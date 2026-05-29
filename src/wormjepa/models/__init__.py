"""JEPA model components for JEPA-WORM Phase 0.

The encoder is **vision-only at test time** (FR17 / architectural pattern #9):
its ``forward()`` signature accepts only video. Neural data flows in only via
the warm-start auxiliary heads (FR16), which are composed by the training
loop and never invoked at eval time.
"""

from wormjepa.models.encoder import WormJEPAEncoder

__all__ = ["WormJEPAEncoder"]
