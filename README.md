<p align="center">
  <a href="https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA.jpg">
    <img
      src="https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/main/media/logo_oct_inclusive.png"
      alt="aivus-caa logo"
    >
  </a>
</p>

> [!IMPORTANT]  
> The original codebase has been renamed to **[HolOrama](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama)** and continues to be actively developed there.
> This repository (**AIVUS-CAA**) is a frozen-in-time fork, kept solely to mirror the link cited in the original publication (Computer Methods and Programs in Biomedicine). It is **not maintained**, for the current, actively developed version, please use [HolOrama](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama).

# AIVUS (Automated IntraVascular UltraSound Image Processing) <!-- omit in toc -->
- AIVUS-CAA (Quantification of Coronary Artery Anomalies)
- AIVUS-OCT (Quantification of Optical Coherence Tomography)

[![version](https://img.shields.io/github/v/tag/AI-in-Cardiovascular-Medicine/AIVUS-CAA?label=version&color=yellow)](https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/tags)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/readthedocs/aivus-caa)](https://aivus-caa.readthedocs.io)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cmpb.2025.109065-blue)](https://doi.org/10.1016/j.cmpb.2025.109065)

## Table of contents <!-- omit in toc -->

- [Installation](#installation)
- [Basic](#basic)
- [Functionalities](#functionalities)
- [Configuration](#configuration)
- [Usage](#usage)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Acknowledgements](#acknowledgements)

## Installation

The easiest way to install is via the provided scripts — they handle all platform-specific steps automatically. If you run into problems, follow the step-by-step instructions further below.

### Linux / macOS — script

```bash
bash install.sh
```

The default install is GPU-ready (CUDA 11.8). Optional flags:

| Flag | Effect |
|------|--------|
| `--dev` | Also install dev dependencies |
| `--nnuzoo` | Install nnUZoo from GitHub |
| `--cpu` | Switch to CPU-only torch (no GPU) |
| `--cuda 121` | Switch to CUDA 12.1 build instead of the default cu118 |

### Windows — script

First install the [Visual C++ Redistributable 2022 (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) if not already present, then run in PowerShell:

```powershell
.\install.ps1
```

The script automatically applies all Windows-specific fixes (missing OpenMP DLL, `optree` version pin). The default install is GPU-ready (CUDA 11.8). Optional flags:

| Flag | Effect |
|------|--------|
| `-Dev` | Also install dev dependencies |
| `-NnUZoo` | Install nnUZoo from GitHub |
| `-Cpu` | Switch to CPU-only torch (no GPU) |
| `-Cuda 121` | Switch to CUDA 12.1 build instead of the default cu118 |

---

### Step-by-step installation (if the scripts fail)

#### Linux / macOS

```bash
pip install uv
uv sync
source .venv/bin/activate
```

For developers:
```bash
uv sync --group dev
```

For nnUZoo (automatic segmentation):
```bash
uv pip install git+https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo@main
```

For GPU (CUDA 11.8):
```bash
uv pip install --reinstall "torch==2.4.0+cu118" "torchvision==0.19.0+cu118" \
    --index-url https://download.pytorch.org/whl/cu118
```

If you plan on using GPU acceleration, install the required NVIDIA drivers and CUDA toolkit beforehand:
```bash
sudo apt update && sudo apt upgrade
sudo apt install build-essential dkms
sudo ubuntu-drivers autoinstall
sudo reboot
nvidia-smi  # verify driver installation
sudo apt install nvidia-cuda-toolkit
```

#### Windows — step by step

##### 1. Install Visual C++ Redistributable

Download and install the [Visual C++ Redistributable 2022 (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) if not already present.

##### 2. Base install

```powershell
pip install uv
uv sync
.\.venv\Scripts\Activate.ps1
```

##### 3. Fix missing LLVM OpenMP runtime (`libomp140.x86_64.dll`)

PyTorch 2.4.0 on Windows depends on `libomp140.x86_64.dll` which is not bundled in the pip wheel. Run this once after installation:

```python
import urllib.request, tarfile, io, os, sys

url = 'https://conda.anaconda.org/conda-forge/win-64/llvm-openmp-14.0.0-h2d74725_0.tar.bz2'
data = urllib.request.urlopen(url).read()
dest = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib', 'libomp140.x86_64.dll')

with tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2') as t:
    f = t.extractfile('Library/bin/libomp.dll')
    with open(dest, 'wb') as out:
        out.write(f.read())
print('Done:', dest)
```

> **Note:** This file will be lost if torch is reinstalled — re-run the script afterwards.

##### 4. Fix `optree` version incompatibility

`optree >= 0.14` is incompatible with `torch 2.4.0` and causes a C-level access violation. Downgrade it:

```bash
uv pip install "optree==0.13.1"
```

##### 5. GPU acceleration (CUDA)

Install the CUDA-enabled torch build matching your driver. With CUDA driver ≤ 12.0 (check with `nvidia-smi`), use the CUDA 11.8 build:

```bash
uv pip install --reinstall "torch==2.4.0+cu118" "torchvision==0.19.0+cu118" \
    --index-url https://download.pytorch.org/whl/cu118
```

After installing the CUDA build, re-run the `libomp140.x86_64.dll` script from step 3.

### Precompiled version
Find a precompiled version pinned to release (compiled with `nuitka`).
If you want to compile the project on your own, use the following command:
```bash
python -m nuitka --standalone --plugin-enable=pyqt6 --include-package=pydicom --include-package=scipy --include-package=numpy --follow-imports --show-progress main.py
```

## Functionalities

This application is designed for IVUS and OCT images in DICOM or NIfTi format and offers the following functionalities:

- Inspect IVUS/OCT images frame-by-frame and display DICOM metadata
- Manually **draw one or several contours** (lumen, eem, calcium, side branch, macrophage, lipid) with automatic calculation of several measurements
- Either draw closed spline, open spline or a closed spline with an uncertain region indicated by start- and end point
- **Automatic segmentation** of (currently only IVUS) lumen for all frames
- **Automatic gating** with extraction of diastolic/systolic frames if in IVUS mode
- Manually tag diastolic/systolic frames
- Ability to measure up to two distances per frame which will be stored in the report
- Indicate the wire shadow using an angle
- Create automatic masks from contour with predefined rulesets
- **Auto-save** of contours and tags enabled by default with user-definable interval
- Generation of report file containing detailed metrics for each frame
- Save coordinate data as csv files
- Ability to save images and segmentations as **NIfTi files**, e.g. to train a machine learning model

## Configuration

Make sure to quickly check the **config.yaml** file and configure everything to your needs.

**Display**:
- image_size: In Pixel creates quadratic box displaying the IVUS images. Default 800x800 px.
- gating_display_stretch: input parameter for .setStretchFactor in class RightHalf
- lview_display_stretch: input parameter for .setStretchFactor in class RightHalf
- windowing_sensitivity: Defines how much windowing changes with <kbd>RMB<kbd> draging
- n_interactive_points: The dragable points on the contour, default 10 equally spaced points, however new one can also be added interactively by clicking on the contour
- alpha_contour: Used as input parameter for .setAlpha in class Display. Default 128 for 50% transparency, higher values more opaque.

**Gating**:
- normalize_step: If step=0 compute one global z-score over the entire data. If step > 0 split data into non-overlapping windows of length normalize_step and apply z-score to each window seperately.
- lowcut: lower frequency for Butterworth filter. Default 1.33Hz which is ~80bpm (since detecting systole and diastole this is equivalent to 40bpm).
- highcut: higher frequency for Butterworth filter. Default is 6.0Hz which is 360bpm (since detecting systole and diastole this is equivalent to 180bpm).
- order: Order for the Butterworth filter. Default 6 based on experiments with our data.
- extrema_y_lim: Setting for finding local extrema, next extrema most be >50th percentile of previous as default
- extrema_x_lim: Distance in frames for next local extrema. Default set to 6 frames.

## Usage

After the config file is set up properly, you can run the application using:

```bash
python3 src/main.py
```

This will open a graphical user interface (GUI) in which you have access to the above-mentioned functionalities.

## Keyboard shortcuts

For ease-of-use, this application contains several keyboard shortcuts.\
In the current state, these cannot be changed by the user (at least not without changing the source code).
### v1.0.0 (Base module and state of publication)
- Press <kbd>Ctrl</kbd> + <kbd>O</kbd> to open a DICOM/NIfTi file
- Use the <kbd>A</kbd> and <kbd>D</kbd> keys to move through the IVUS images frame-by-frame
- If gated (diastolic/systolic) frames are available, you can move through those using <kbd>S</kbd> and <kbd>W</kbd>\
  Make sure to select which gated frames you want to traverse using the corresponding button (blue for diastolic, red for systolic)
- Press <kbd>E</kbd> to manually draw a new lumen contour\
  In case you accidentally delete a contour, you can use <kbd>Ctrl</kbd> + <kbd>Z</kbd> to undo
- Use <kbd>1</kbd>, <kbd>2</kbd> to draw measurements 1 and 2, respectively
- Drag <kbd>RMB</kbd> left/righ up/down for windowing (can be reset by pressing <kbd>R</kbd>)
- Press <kbd>C</kbd> to toggle color mode
- Press <kbd>H</kbd> to hide all contours
- Press <kbd>J</kbd> to jiggle around the current frame
- Press <kbd>Ctrl</kbd> + <kbd>S</kbd> to manually save contours (auto-save is enabled by default)
- Press <kbd>Ctrl</kbd> + <kbd>R</kbd> to generate report file
- Press <kbd>Ctrl</kbd> + <kbd>Q</kbd> to close the program
- Press <kbd>Alt</kbd> + <kbd>P</kbd> to plot the results for gated frames (difference area systole and diastole, by distance)
- Press <kbd>Alt</kbd> + <kbd>Delete</kbd> to define a range of frames to remove gating
- Press <kbd>Alt</kbd> + <kbd>S</kbd> to define a range of frames to switch systole and diastole in gated frames

### v1.1.0 and higher
Additionally:
- Press <kbd>Esc</kbd> to exit drawing mode and return to a neutral state
- Press <kbd>RMB</kbd> on an existing knot point to remove it
- Scroll <kbd>MW</kbd> to scroll through frames (forward/backward)
- Drag <kbd>LMB</kbd> up/down for zooming (can be reset by pressing <kbd>F</kbd>)
- Drag <kbd>Ctrl</kbd> + <kbd>LMB</kbd> to move the image inside it's widget
- Press <kbd>Ctrl</kbd> + <kbd>MW</kbd> to shrink or expand the currently selected contour (moves all knot points toward/away from their centroid)
- Press <kbd>Q</kbd> to manually draw an ``external elastic membrane`` (EEM) contour
- Press <kbd>Shift</kbd> + <kbd>Q</kbd> to spawn an ``EEM`` contour from an existing ``lumen`` contour (20 % radial expansion from lumen centroid); does nothing if EEM already exists on that frame
- Press <kbd>Shift</kbd> + <kbd>A</kbd> to copy the active contour from the previous frame to the current frame
- Press <kbd>Shift</kbd> + <kbd>D</kbd> to copy the active contour from the next frame to the current frame
- Press <kbd>Shift</kbd> + <kbd>W</kbd> to copy the active contour from the next gated/tagged frame (only works when the current frame is itself gated/tagged)
- Press <kbd>Shift</kbd> + <kbd>S</kbd> to copy the active contour from the previous gated/tagged frame (only works when the current frame is itself gated/tagged)
- Press <kbd>7</kbd> to manually draw a ``calcification`` contour
- Press <kbd>Ctrl</kbd> + <kbd>7</kbd> to draw an additional ``calcification`` contour in the current active spline tool (open or closed)
- Press <kbd>8</kbd> to manually draw a ``side branch`` contour
- Press <kbd>Ctrl</kbd> + <kbd>8</kbd> to draw an additional ``side branch`` contour in the current active spline tool (open or closed)
- Press <kbd>9</kbd> to manually draw a ``lipid`` contour (only open spline)
- Press <kbd>Ctrl</kbd> + <kbd>9</kbd> to draw an additional ``lipid`` contour
- Press <kbd>0</kbd> to manually draw a ``macrophage`` contour (only open spline)
- Press <kbd>Ctrl</kbd> + <kbd>0</kbd> to draw an additional ``macrophage`` contour

## Tutorial (v1.0.0 - Base module)
An example case is provided under "/test_cases/patient_example", allowing to follow along.

### Window manipulation:
![Demo](media/explanation_software_part1.gif)
### Contour manipulation:
![Demo](media/explanation_software_part2.gif)
### Gating module:

This module implements gating by analyzing both image-derived metrics (e.g., pixel-wise correlation and blurriness) and vector-based contour measurements (e.g., distance and direction from the image center to each contour centroid). Changes in these metrics are displayed over the sequence of frames during a pullback.

The resting phases of the cardiac cycle—diastole and systole—are characterized by minimal vessel motion for several consecutive frames. We visualize these phases using two curves: the image-based curve (green) represents metrics such as correlation peaks and minimal blurriness, while the contour-based curve (yellow) reflects extrema in the vector measurements (i.e., alternating peaks and valleys corresponding to systolic and diastolic positions).

- **Image-Based Metrics**: Select local maxima corresponding to frames with the highest pixel correlation and lowest blurriness.

- **Contour-Based Metrics**: Select extrema in the distance vector, capturing the transition between diastole and systole.

Movement patterns may vary between datasets; consequently, the final frame selection is left to the user.

**Peak Assignment**: Detected peaks in each curve are matched by intersecting their frame indices. We apply a Butterworth filter (passband: 45–180 bpm) to smooth each curve; the unfiltered signal is displayed as a dotted line beneath the filtered curve.

**Interactive Gating Interface**:
- Range Selection: Specify the frame interval for gating.
- Zoom & Pan: Zoom into the plot and drag lines to adjust gating thresholds or remove unwanted markers by dragging them downward.
- Compare Frames: Click "Compare Frames" to open the nearest proximal frame for the selected phase (systole or diastole).

![Demo](media/explanation_software_part3.gif)

## Tutorial (v1.1.x - Full segmentation)
Version 1.1.0 and higher offer the additional possibility to segment the EEM, calcification and side branches. This works in the same style as for the base contours. Clicking on any contour in the image automatically sets it as the active contour.

> [!NOTE]
> The segmentation models are currently only trained for lumen contours. In the future, we will implement additional models for all contour types.

Since version 1.2.0 additionally it is possible to read in OCT images and performe additional contouring functionalities.

### Example v1.2.0 with OCT
Here first adding a catheter angle from the tools above, then adding a lumen contour (closed spline) and lastly adding a EEM contour (closed spline) but then setting an uncertain region between start point (yellow) and end point (red) by double clicking. Since this version also zoom in can be performed by mouse scroll on the current mouse position. Measurements are currently hidden, see Checkbox bottom.

![Demo](media/explanation_software_part4.gif)

Here in a next step an open spline is created for calcium (if the spline is open it automatically calculates two an angle from the lumen center to start and end point of the open spline). Points are then removed using <kbd>RMB</kbd> and using <kbd>Ctrl</kbd> + <kbd>7</kbd> and choosing closed spline a second calcium contour is drawn. In a last step a side branch contour is drawn.

![Demo](media/explanation_software_part5.gif)

The display can change between mask mode (with pre-applied logic of contour layering). Additionally can the contours also be hidden.

![Demo](media/explanation_software_part6.gif)

# Citation
Please kindly cite the following paper if you use this repository.

```
@article{stark2025automated,
  title={Automated intravascular ultrasound image processing and quantification of coronary artery anomalies: the AIVUS-CAA software},
  author={Stark, Anselm W and Kazaj, Pooya Mohammadi and Balzer, Sebastian and Ilic, Marc and Bergamin, Manuel and Kakizaki, Ryota and Giannopoulos, Andreas and Haeberlin, Andreas and R{\"a}ber, Lorenz and Shiri, Isaac and others},
  journal={Computer Methods and Programs in Biomedicine},
  pages={109065},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.cmpb.2025.109065},
  url={https://doi.org/10.1016/j.cmpb.2025.109065}
}
```
```
Stark, A. W., Kazaj, P. M., Balzer, S., Ilic, M., Bergamin, M., Kakizaki, R., Giannopoulos A., Haeberlin A., Räber L., Gräni, C. (2025). Automated intravascular ultrasound image processing and quantification of coronary artery anomalies: the AIVUS-CAA software. Computer Methods and Programs in Biomedicine, 109065.
```

