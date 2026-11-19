# fir_filters

## Overview

Implementations of `FIRFilter`. Both taper an ideal response with a window; they differ in whether the length is given or derived from a specification.

## Components

| Component | Description |
| --------- | ----------- |
| [`window.py`](./window.py) | Given length and window function (`WindowFIRFilter`) |
| [`kaiser.py`](./kaiser.py) | Length and Kaiser window derived from a transition width and an attenuation (`KaiserFIRFilter`) |
