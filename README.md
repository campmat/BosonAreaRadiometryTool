# Boson Area Radiometry Tool - B.A.R.T.

A PyQt-based graphical tool for live region-of-interest temperature analysis using a FLIR Boson thermal camera.

The program displays live thermal images, allows interactive ROI selection, calculates temperature statistics for selected regions, and supports recording/playback of measurement sessions.

## Features

* Live FLIR Boson thermal image display
* Rectangle, ellipse, and polygon ROI selection
* Transparent ROI overlays with editable colors
* ROI area calculation in pixels
* Live ROI table
* Delete selected ROI using the `Delete` key
* Double-click ROI inspection window
* Live temperature statistics per ROI:
  * minimum temperature
  * maximum temperature
  * mean temperature
  * standard deviation
* Recording directory selection
* Save current annotated frame and raw temperature frame
* Record measurement sessions with:
  * raw thermal frames as NumPy arrays
  * annotated frames as PNG images
  * ROI geometry as JSON
  * ROI statistics as CSV
  * session metadata as JSON
* Playback of saved raw-frame sessions
* Designed with Qt Designer

## Supported Camera

Tested/developed for:

* FLIR Boson 320 IR camera

The program assumes Boson TLinear output where:

```text
temperature [°C] = raw_value / 100 - 273.15
```

Check your camera configuration before using the program for quantitative measurements.

## Requirements

* Windows
* Python 3.10
* FLIR Boson camera
* Virtual environment recommended

The program can also run in simulated-frame mode when the FLIR camera or `flirpy` is not available. This is useful for GUI development and testing.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv bart
.\bart\Scripts\activate
```

Install dependencies:

```powershell
pip install numpy opencv-python flirpy PyQt5 pyqtgraph pyinstaller
```

Optional export/plotting packages:

```powershell
pip install pandas openpyxl matplotlib
```

Or install from `requirements.txt` if available:

```powershell
pip install -r requirements.txt
```

## Running

From the project directory:

```powershell
python main.py
```

Expected project structure:

```text
BosonAreaRadiometryTool/
├── main.py
├── ui/
│   └── window.ui
├── README.md
├── requirements.txt
└── LICENSE
```

## Recording and Playback

Recording sessions are stored in timestamped folders inside the selected recording directory.

Example session structure:

```text
recordings/
└── 2026-06-03_15-20-45/
    ├── frames_raw/
    │   ├── frame_000001.npy
    │   ├── frame_000002.npy
    │   └── ...
    ├── frames_annotated/
    │   ├── frame_000001.png
    │   ├── frame_000002.png
    │   └── ...
    ├── rois.json
    ├── statistics.csv
    └── session.json
```

### File Types

| File | Description |
|---|---|
| `frames_raw/*.npy` | Raw temperature arrays in °C |
| `frames_annotated/*.png` | Display image with ROI overlays |
| `rois.json` | ROI type, points, color, and area |
| `statistics.csv` | Timestamped ROI statistics over time |
| `session.json` | Session metadata |

Raw `.npy` frames are preferred for scientific work because conventional video codecs usually do not preserve radiometric 16-bit or floating-point temperature information.

## Building an EXE

Install PyInstaller:

```powershell
pip install pyinstaller
```

Build:

```powershell
pyinstaller --onefile --windowed main.py
```

The executable will be generated in:

```text
dist/
```

If the UI file is not automatically included, use a `.spec` file or add the UI folder manually during packaging.

## Scientific Use Notes

For reproducible measurements, annotated images alone are not sufficient. Recommended exported data should include:

* raw radiometric frame data
* ROI geometry
* temperature conversion assumptions
* timestamps
* ROI statistics
* camera configuration metadata

Radiometric accuracy depends on the camera configuration, calibration state, emissivity assumptions, reflected apparent temperature, optics, and environmental conditions.

## Planned Features

* Export ROI statistics plots, for example mean ± standard deviation over time
* Record annotated video in addition to frame sequences
* Reopen full saved measurement sessions including ROI overlays
* ROI naming, visibility toggling, and locking
* Camera/emissivity settings dialog
* Temperature colorbar and configurable colormap
* Export ROI masks as image or NumPy arrays

## License

This project is intended to be released under the MIT License.

## Acknowledgments

This project uses:

* PyQt5 for the graphical interface
* OpenCV for image processing
* NumPy for numerical processing
* flirpy for FLIR Boson camera access
