# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "signal-filter",
#     "matplotlib-utilities",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# exclude-newer = "7 days"
#
# [tool.uv.sources]
# signal-filter = { path = "..", editable = true }
# matplotlib-utilities = { git = "https://github.com/ry-yoshida-dev/MatplotlibUtilities" }
# ///
"""
Draw the README figure that compares the filters.

Run ``uv run docs/draw_filters_figure.py`` from the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib_utilities import (
    GraphAxis,
    GraphLayout,
    GraphParameters,
    LegendLocation,
    LegendParameters,
    Linestyle,
    MatplotGraphMaker,
    PlotParameters,
    ScatterParameters,
)

from signal_filter import (
    Band,
    BesselFilter,
    ButterworthFilter,
    ChebyshevType1Filter,
    ChebyshevType2Filter,
    EllipticFilter,
    FloatArray,
    FrequencySelectiveFilter,
    HampelFilter,
    KaiserFIRFilter,
    MedianFilter,
    Phase,
    SavitzkyGolayFilter,
    SignalFilter,
    WindowFIRFilter,
)

INK: str = "#0b0b0b"
MUTED: str = "#b5b4ae"
PALETTE: tuple[str, ...] = ("#2a78d6", "#d6452a", "#2a9d5c", "#9b51c9", "#d69a2a")


@dataclass(frozen=True)
class FilterComparisonFigure:
    """
    A 2 x 2 grid: two panels of frequency responses and two of filtered signals.

    - IIR designs: gain of every IIR design for the same 20 Hz lowpass, showing
      ripple against steepness.
    - FIR designs: a windowed design and a Kaiser design, against Butterworth.
    - Phase: a noisy 2 Hz wave through one Butterworth lowpass run with
      ``Phase.ZERO`` and ``Phase.CAUSAL``; only the causal run lags.
    - Outliers: a slow wave with noise and spikes through median, Hampel, and
      Savitzky-Golay filters; only Hampel leaves the samples that are not
      outliers untouched.

    Attributes
    ----------
    sampling_rate : float
        The sampling rate of every filter and signal, in hertz.
    """

    sampling_rate: float = 200.0

    def draw(self, save_path: Path) -> None:
        """
        Draw the figure and save it.

        Parameters
        ----------
        save_path : Path
            Where to write the image.
        """
        maker = MatplotGraphMaker(
            layout=GraphLayout.from_row_column(row=2, column=2),
            parameters=GraphParameters(
                figsize=(13.0, 10.5), dpi=110, font_size=11, w_space=0.22, h_space=0.32
            ),
        )
        self._draw_responses(maker, 0, "IIR designs (order 4, lowpass 20 Hz)", self._iir_designs())
        self._draw_responses(maker, 1, "FIR designs (lowpass 20 Hz)", self._fir_designs())
        self._draw_phase(maker, 2)
        self._draw_outliers(maker, 3)
        maker.finalize(save_path=str(save_path), is_showing_result_enabled=False)

    def _iir_designs(self) -> list[tuple[str, FrequencySelectiveFilter]]:
        band = Band.lowpass(20.0)
        rate = self.sampling_rate
        causal = Phase.CAUSAL
        return [
            ("Butterworth", ButterworthFilter(band=band, sampling_rate=rate, phase=causal)),
            ("Chebyshev I", ChebyshevType1Filter(band=band, sampling_rate=rate, phase=causal)),
            ("Chebyshev II", ChebyshevType2Filter(band=band, sampling_rate=rate, phase=causal)),
            ("Elliptic", EllipticFilter(band=band, sampling_rate=rate, phase=causal)),
            ("Bessel", BesselFilter(band=band, sampling_rate=rate, phase=causal)),
        ]

    def _fir_designs(self) -> list[tuple[str, FrequencySelectiveFilter]]:
        band = Band.lowpass(20.0)
        rate = self.sampling_rate
        causal = Phase.CAUSAL
        kaiser = KaiserFIRFilter(
            band=band, transition_width=10.0, attenuation=60.0, sampling_rate=rate, phase=causal
        )
        return [
            ("Butterworth (IIR)", ButterworthFilter(band=band, sampling_rate=rate, phase=causal)),
            (
                "Window, Hamming, 51 taps",
                WindowFIRFilter(band=band, taps=51, sampling_rate=rate, phase=causal),
            ),
            (f"Kaiser, 60 dB in 10 Hz, {kaiser.taps} taps", kaiser),
        ]

    def _draw_responses(
        self,
        maker: MatplotGraphMaker,
        index: int,
        title: str,
        designs: list[tuple[str, FrequencySelectiveFilter]],
    ) -> None:
        for (name, design), color in zip(designs, PALETTE, strict=False):
            response = design.frequency_response()
            maker.plot(
                x=response.frequencies,
                y=np.maximum(response.gain_db, -120.0),
                index=index,
                subparams=PlotParameters(color=color, linewidth=1.8, label=name),
            )
        maker.set_lim(0.0, self.sampling_rate / 2, axis=GraphAxis.X, index=index)
        maker.set_lim(-90.0, 5.0, axis=GraphAxis.Y, index=index)
        maker.set_title(title=title, index=index)
        maker.set_label(label="Frequency [Hz]", axis=GraphAxis.X, index=index)
        maker.set_label(label="Gain [dB]", axis=GraphAxis.Y, index=index)
        maker.legend(
            index=index,
            subparams=LegendParameters(loc=LegendLocation.UPPER_RIGHT, frameon=False),
        )

    def _draw_phase(self, maker: MatplotGraphMaker, index: int) -> None:
        time = np.arange(400) / self.sampling_rate
        clean = np.sin(2.0 * np.pi * 2.0 * time)
        noisy = clean + 0.3 * np.random.default_rng(0).standard_normal(time.size)
        runs: list[tuple[str, SignalFilter]] = [
            (
                f"Butterworth 6 Hz, {phase.name.lower()} phase",
                ButterworthFilter(
                    band=Band.lowpass(6.0), sampling_rate=self.sampling_rate, phase=phase
                ),
            )
            for phase in Phase
        ]
        self._draw_signals(maker, index, time, noisy, clean, runs)
        maker.set_title(title="Zero phase against causal", index=index)
        maker.set_label(label="Time [s]", axis=GraphAxis.X, index=index)

    def _draw_outliers(self, maker: MatplotGraphMaker, index: int) -> None:
        time = np.arange(400) / self.sampling_rate
        clean = np.sin(2.0 * np.pi * 1.0 * time) * np.exp(-time / 2.0)
        noisy = clean + 0.03 * np.random.default_rng(1).standard_normal(time.size)
        spikes = np.random.default_rng(2).choice(time.size, size=12, replace=False)
        noisy[spikes] += np.random.default_rng(3).choice([-1.0, 1.0], size=12) * 1.2
        runs: list[tuple[str, SignalFilter]] = [
            ("Hampel, 15 samples", HampelFilter(window_length=15)),
            ("Median, 15 samples", MedianFilter(window_length=15)),
            (
                "Savitzky-Golay, 31 samples, cubic",
                SavitzkyGolayFilter(window_length=31, polynomial_order=3),
            ),
        ]
        self._draw_signals(maker, index, time, noisy, clean, runs)
        maker.set_title(title="Sliding windows on spiky data", index=index)
        maker.set_label(label="Time [s]", axis=GraphAxis.X, index=index)

    def _draw_signals(
        self,
        maker: MatplotGraphMaker,
        index: int,
        time: FloatArray,
        noisy: FloatArray,
        clean: FloatArray,
        runs: list[tuple[str, SignalFilter]],
    ) -> None:
        maker.scatter(
            x=time,
            y=noisy,
            index=index,
            subparams=ScatterParameters(c=MUTED, s=6, label="Input", zorder=1),
        )
        maker.plot(
            x=time,
            y=clean,
            index=index,
            subparams=PlotParameters(
                color=INK, linewidth=1.2, linestyle=Linestyle.DASHED, label="Clean", zorder=2
            ),
        )
        for (name, run), color in zip(runs, PALETTE, strict=False):
            maker.plot(
                x=time,
                y=run.apply(noisy),
                index=index,
                subparams=PlotParameters(color=color, linewidth=1.8, label=name, zorder=3),
            )
        maker.set_lim(time[0], time[-1], axis=GraphAxis.X, index=index)
        maker.legend(
            index=index,
            subparams=LegendParameters(
                loc=LegendLocation.UPPER_CENTER, bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False
            ),
        )


if __name__ == "__main__":
    output: Path = Path(__file__).resolve().parent / "filters.png"
    FilterComparisonFigure().draw(output)
