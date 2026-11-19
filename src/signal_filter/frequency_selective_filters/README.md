# frequency_selective_filters

## Overview

Implementations of `FrequencySelectiveFilter`, split into recursive (IIR) and non-recursive (FIR) designs. Each abstract family implements how its coefficients are run and evaluated; its implementations only design them.

## Components

| Component | Description |
| --------- | ----------- |
| [`iir_filter.py`](./iir_filter.py) | Abstract IIR filter run as second-order sections (`IIRFilter`) |
| [`iir_filters/`](./iir_filters/README.md) | Butterworth, Chebyshev type I and II, elliptic, Bessel, and notch designs |
| [`bessel_normalization.py`](./bessel_normalization.py) | What the cutoff of a Bessel design refers to (`BesselNormalization`) |
| [`fir_filter.py`](./fir_filter.py) | Abstract linear-phase FIR filter (`FIRFilter`) |
| [`fir_filters/`](./fir_filters/README.md) | Window method and Kaiser method designs |
| [`window_function.py`](./window_function.py) | Taper of a window method design (`WindowFunction`) |
