# tests

## Overview

Pytest suite, laid out like `src/signal_filter`. Besides input handling it checks the signal processing: measured gains of sinusoids against `frequency_response`, the ripple and attenuation each design promises, delays, and boundary handling against references built with NumPy alone.

## Components

| Component | Description |
| --------- | ----------- |
| [`conftest.py`](./conftest.py) | Every concrete filter as a parametrized fixture, and test sinusoids |
| [`test_signal_filter.py`](./test_signal_filter.py) | Shapes, channels, dtypes, and input validation shared by every filter |
| [`test_band.py`](./test_band.py) | Band construction and validation |
| [`test_frequency_response.py`](./test_frequency_response.py) | Magnitude, gain, and phase of a response |
| [`frequency_selective_filters/`](./frequency_selective_filters/README.md) | IIR and FIR filters |
| [`sliding_window_filters/`](./sliding_window_filters/README.md) | Moving average, median, Savitzky-Golay, and Hampel filters |

## Examples

```bash
uv run pytest
uv run mypy && uv run basedpyright
```
