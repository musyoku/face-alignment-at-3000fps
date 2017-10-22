# Face Alignment at 3000 FPS via Regressing Local Binary Features

## Requirements

- Python 3
- OpenCV 3
- Boost
- Dlib

## Installation

### macOS

Python 3

`brew install python3`

Boost

`brew install boost-python --with-python3`

### Ubuntu

Boost

```
./bootstrap.sh --with-python=python3 --with-python-version=3.5
./b2 python=3.5 -d2 -j4 --prefix YOUR_BOOST_DIR install
```

Dlib

```
sudo apt install libomp-dev
pip install dlib --user
```

## Datasets

###  300 Faces in the Wild

Download `300w.zip.001`, `300w.zip.002`, `300w.zip.003`, `300w.zip.004` from [ibug.doc.ic.ac.uk/resources/facial-point-annotations/](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

### IBUG

Download `afw.zip`, `ibug.zip`, `lfpw.zip`, `helen.zip` from [ibug.doc.ic.ac.uk/resources/facial-point-annotations/](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

or from Google Drive.

- [afw.zip](https://drive.google.com/open?id=0ByQaxyG1S5JRMUdtNGYzNWJJUmc)
- [ibug.zip](https://drive.google.com/open?id=0ByQaxyG1S5JRR2dMd29Lakt0UDg)
- [lfpw.zip](https://drive.google.com/open?id=0ByQaxyG1S5JRTUhuMnExdDlBRFk)
- [helen.zip](https://drive.google.com/open?id=0ByQaxyG1S5JRazF3MGU0enZkSVk)