"""How sliding windows are filled beyond the ends of a signal."""

from __future__ import annotations

from enum import Enum, auto


class Boundary(Enum):
    """
    Values assumed beyond the ends of a signal, shown for samples ``a b c d``.

    Attributes
    ----------
    REFLECT
        Mirror about the end sample: ``d c b | a b c d | c b a``. Keeps the
        local trend, the usual choice.
    NEAREST
        Repeat the end sample: ``a a a | a b c d | d d d``. Suits signals that
        settle at their ends.
    WRAP
        Continue periodically: ``b c d | a b c d | a b c``. Suits signals that
        cover whole periods.
    """

    REFLECT = auto()
    NEAREST = auto()
    WRAP = auto()
