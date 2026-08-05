"""Autonomous edit pipeline.

Semantic decisions belong to the planner and the reviewer. The execution layer
carries them out, degrades on measured evidence when it must, and reports what
it did -- it never decides that a piece of material is not worth using.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
