# SKAB Data for ANSR-DT

This directory contains the subset of the Skoltech Anomaly Benchmark (SKAB) used by the dedicated ANSR-DT SKAB pipeline. Only the raw CSV sensor streams required by the loader are kept locally: `anomaly-free/`, `valve1/`, `valve2/`, and `other/`.

SKAB is a multivariate industrial anomaly benchmark collected from a water-circulation testbed. The ANSR-DT SKAB configuration uses eight sensor channels: `Accelerometer1RMS`, `Accelerometer2RMS`, `Current`, `Pressure`, `Temperature`, `Thermocouple`, `Voltage`, and `Volume Flow RateRMS`, together with the SKAB `anomaly` labels.

External sources:

- GitHub: https://github.com/waico/SKAB
- DOI / dataset record: https://doi.org/10.34740/KAGGLE/DSV/1693952

The original benchmark repository includes notebooks, reference baselines, documentation, and additional assets that are intentionally not vendored here.
