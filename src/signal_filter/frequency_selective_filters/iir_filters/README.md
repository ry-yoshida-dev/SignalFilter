# iir_filters

## Overview

Implementations of `IIRFilter`. The band designs trade passband flatness, steepness, and phase linearity against each other; with the same order, steeper designs ripple more.

## Components

| Component | Description |
| --------- | ----------- |
| [`butterworth.py`](./butterworth.py) | Maximally flat, no ripple (`ButterworthFilter`) |
| [`chebyshev_type1.py`](./chebyshev_type1.py) | Passband ripple for a steeper roll-off (`ChebyshevType1Filter`) |
| [`chebyshev_type2.py`](./chebyshev_type2.py) | Flat passband and a bounded stopband ripple; edges mark the stopband (`ChebyshevType2Filter`) |
| [`elliptic.py`](./elliptic.py) | Ripple in both bands for the steepest roll-off (`EllipticFilter`) |
| [`bessel.py`](./bessel.py) | Nearly constant group delay, little overshoot (`BesselFilter`) |
| [`notch.py`](./notch.py) | Removes a single frequency (`NotchFilter`) |
