# Boson Area Radiometry Tool - B.A.R.T.

A PyQt-based graphical tool for live region-of-interest temperature analysis using a FLIR Boson thermal camera.

The program allows users to display live thermal images, draw measurement regions, and calculate temperature statistics for selected areas.

## Features

* Live FLIR Boson thermal image display
* Rectangle, ellipse, and polygon ROI selection
* Transparent ROI overlays with editable colors
* ROI area calculation in pixels
* Temperature statistics per ROI:

  * minimum temperature
  * maximum temperature
  * mean temperature
  * standard deviation
* Live ROI table
* Double-click ROI inspection window
* Designed with Qt Designer
* Intended for future frame/video recording and data export

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

## Planned Features

* Save annotated thermal frames
* Save raw radiometric frames
* Record annotated video
* Export ROI definitions as JSON
* Export ROI statistics as CSV
* Plot ROI mean temperature ± standard deviation over time
* Reopen saved measurement sessions
* ROI naming, visibility toggling, and locking

## Scientific Use Notes

For reproducible measurements, annotated images alone are not sufficient. Recommended exported data should include:

* raw radiometric frame data
* ROI geometry
* temperature conversion assumptions
* timestamps
* ROI statistics
* camera configuration metadata

## License

This project is intended to be released under the MIT License.

## Acknowledgments

This project uses:

* PyQt5 for the graphical interface
* OpenCV for image processing
* NumPy for numerical processing
* flirpy for FLIR Boson camera access
