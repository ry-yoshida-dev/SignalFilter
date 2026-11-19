# sliding_window_filters

## Overview

Implementations of `SlidingWindowFilter`. They need no sampling rate and act on the samples around each sample, so they suit smoothing and outlier removal where no frequency band is defined.

## Components

| Component | Description |
| --------- | ----------- |
| [`moving_average.py`](./moving_average.py) | Mean of the window (`MovingAverageFilter`) |
| [`median.py`](./median.py) | Median of the window; removes impulses, keeps steps (`MedianFilter`) |
| [`savitzky_golay.py`](./savitzky_golay.py) | Local polynomial fit; keeps peaks, gives smoothed derivatives (`SavitzkyGolayFilter`) |
| [`hampel.py`](./hampel.py) | Replaces only the outliers with the window median (`HampelFilter`) |

## Examples

```python
smoothed = SavitzkyGolayFilter(window_length=11, polynomial_order=3).apply(signal)
is_outlier = HampelFilter(window_length=11, threshold=3.0).detect(signal)
```
