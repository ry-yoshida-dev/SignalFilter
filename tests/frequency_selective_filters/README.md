# frequency_selective_filters

## Overview

Tests for the filters in `signal_filter.frequency_selective_filters`.

## Components

| Component | Description |
| --------- | ----------- |
| [`test_frequency_selective_filter.py`](./test_frequency_selective_filter.py) | Phases, frequency responses against filtered sinusoids, and validation shared by IIR and FIR filters |
| [`test_iir_filters.py`](./test_iir_filters.py) | Edge gains, ripple, attenuation, step response, and notch bandwidth of each IIR design |
| [`test_fir_filters.py`](./test_fir_filters.py) | Linear phase, delay, window attenuation, and Kaiser specifications |
