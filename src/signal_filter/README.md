# signal_filter

## Overview

Package root. `SignalFilter` is the abstract filter every class implements; its two families, frequency-selective and sliding-window filters, are abstract subclasses whose implementations live in the sub-packages. Everything public is re-exported from [`__init__.py`](./__init__.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`array_types.py`](./array_types.py) | NumPy array type aliases (`FloatArray`, `ComplexArray`, `BoolArray`) |
| [`_samples.py`](./_samples.py) | Private conversion and validation of input signals |
| [`_parameters.py`](./_parameters.py) | Private validation of filter parameters |
| [`signal_filter.py`](./signal_filter.py) | Abstract filter (`SignalFilter`) |
| [`frequency_selective_filter.py`](./frequency_selective_filter.py) | Abstract filter designed in the frequency domain (`FrequencySelectiveFilter`) |
| [`band.py`](./band.py) | Frequency band to pass or reject (`Band`) |
| [`band_type.py`](./band_type.py) | Lowpass, highpass, bandpass, or bandstop (`BandType`) |
| [`phase.py`](./phase.py) | Zero-phase or causal application (`Phase`) |
| [`frequency_response.py`](./frequency_response.py) | Complex gain over frequency (`FrequencyResponse`) |
| [`frequency_selective_filters/`](./frequency_selective_filters/README.md) | IIR and FIR filters |
| [`sliding_window_filter.py`](./sliding_window_filter.py) | Abstract filter over a centered window (`SlidingWindowFilter`) |
| [`boundary.py`](./boundary.py) | Values assumed beyond the ends of a signal (`Boundary`) |
| [`sliding_window_filters/`](./sliding_window_filters/README.md) | Moving average, median, Savitzky-Golay, and Hampel filters |
