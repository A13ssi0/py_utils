# py_utils

General-purpose Python utilities for EEG data loading, event handling, signal processing, covariance extraction, online filtering, and plotting.

This repository is intentionally small and can be used independently from any parent project. It is useful as a helper package for offline EEG analysis, BCI pipelines, and scripts that need consistent handling of `.mat`, `.gdf`, `.fif`, and event-annotated EEG recordings.

## Modules

```text
.
|-- data_managment.py     # File discovery, MAT/GDF/FIF loading, event parsing, save/load helpers
|-- eeg_managment.py      # Channel selection, event-to-vector conversion, train/validation indices
|-- signal_processing.py  # Filters, covariance matrices, spectrograms, Fisher score, online processing
|-- plots_prints.py       # Rich terminal tables and Matplotlib plotting helpers
`-- LICENSE
```

## Installation

The package is usually used by adding this repository to your Python path or by including it as a Git submodule in another project.

Example:

```bash
git clone https://github.com/A13ssi0/py_utils.git
```

Then import it from a script that can see the repository folder:

```python
from py_utils.data_managment import get_files
from py_utils.signal_processing import get_bandranges
```

If you are using it as a submodule, make sure the parent project adds the parent directory to `sys.path`, or run scripts from a location where `py_utils` is importable.

## Dependencies

The utilities rely on common scientific Python packages:

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn mne joblib tqdm rich pyriemann
```

Depending on the functions you use, not every dependency is always required. For example, plotting utilities require `matplotlib` and `rich`, while covariance helpers require `pyriemann`.

## Data Loading

`data_managment.py` contains helpers for loading and saving EEG-related files.

Supported inputs include:

- MATLAB `.mat` files containing signal `s` and header `h`.
- GDF files loaded through MNE.
- FIF files with external CSV event files.
- `.joblib` and `.mat` objects through the generic `load()` and `save()` helpers.

Example:

```python
from py_utils.data_managment import get_files

signal, events, header, filenames = get_files(
    path="path/to/recordings",
    ask_user=True
)
```

Expected output shape:

- `signal`: samples by channels.
- `events`: a pandas DataFrame with event metadata.
- `header`: dictionary-like metadata.
- `filenames`: list of loaded files.

For MATLAB recordings, the expected structure is:

```text
s           # EEG signal, samples x channels
h           # header struct
h.EVENT     # event struct with TYP, POS, and DUR
```

## Event Utilities

`eeg_managment.py` provides helpers to convert event tables into sample/window vectors and select channels.

Example:

```python
from py_utils.eeg_managment import get_EventsVector_onFeedback

labels = get_EventsVector_onFeedback(
    events=events,
    lengthVector=n_windows,
    events_typ=[769, 770]
)
```

Useful functions:

- `get_channelsMask()`: build a boolean mask from desired channel names.
- `select_channels()`: select channels from a signal matrix.
- `proc_pos2win()`: convert event positions from samples to window indices.
- `get_EventsVector_onFeedback()`: label feedback windows from cue/feedback events.
- `get_indices_train_validation()`: split feedback samples into train/validation indices.
- `apply_ROI_over_channels()`: average or group channels by region of interest.

## Signal Processing

`signal_processing.py` contains offline and online signal-processing helpers.

Common offline workflow:

```python
from py_utils.signal_processing import (
    get_bandranges,
    get_covariance_matrix_normalized,
)

filtered = get_bandranges(
    signal=signal,
    bandranges=[[6, 24]],
    fs=250,
    filter_order=2,
    filtType="bandpass"
)

covs, cov_events = get_covariance_matrix_normalized(
    data=filtered,
    events=events,
    windowsLength=1,
    windowsShift=0.04,
    fs=250,
    normalizationMethod="lwf"
)
```

Useful functions/classes:

- `get_bandranges()`: apply Butterworth filters to one or more frequency bands.
- `get_data_windowed()`: split continuous data into sliding windows.
- `get_covariance_matrix_normalized()`: compute covariance matrices with trace or Ledoit-Wolf normalization.
- `get_covariance_matrix_traceNorm_online()`: online trace-normalized covariance.
- `get_covariance_matrix_lwfNorm_online()`: online Ledoit-Wolf covariance.
- `logbandpower()`: compute log-bandpower features.
- `proc_spectrogram()`: compute spectrogram features.
- `compute_fisher_score()`: estimate feature discriminability across runs.
- `RealTimeButterFilter`: stateful online Butterworth filter.
- `RealTimeLogBandPower`: stateful online log-bandpower processor.

## Plotting And Printing

`plots_prints.py` contains small helpers for formatted terminal output and Matplotlib visualizations.

Example:

```python
from py_utils.plots_prints import plot_confusion_matrix

plot_confusion_matrix(matrix, labels=["left", "right"])
```

Useful functions:

- `plot_confusion_matrix()`: print a colored confusion matrix in the terminal using Rich.
- `fmt()`: format scalar/vector/matrix values for compact logs.
- `plot_array_runs_grid()`: plot per-run/per-band metrics.
- `plot_similarity_matrices()`: plot angle and distance similarity matrices with optional performance traces.

## Notes

- Many functions assume EEG matrices are shaped as samples by channels.
- Some signal-processing functions expect banded data shaped as bands by samples by channels.
- Event DataFrames generally use lowercase column names such as `typ`, `pos`, and `dur`.
- GDF/FIF loading depends on MNE and on the event annotations available in the source files.
- This is a utility repository rather than a fully packaged Python distribution; import paths may need to be configured by the calling project.

## License

This project is released under the MIT License. See `LICENSE` for details.
