"""DL-MoM Baseline Methods."""
from .methods import (
    BaselineResult,
    DirectBaseline,
    CoTBaseline,
    SelfConsistencyBaseline,
    TextMASBaseline,
    get_baseline,
)
from .advanced import (
    TreeOfThoughtBaseline,
    LatentMASSimple,
    get_advanced_baseline,
)
