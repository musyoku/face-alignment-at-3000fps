# [WIP]Face Alignment at 3000 FPS via Regressing Local Binary Features

# Work in Progress

- [x] Training local binary features
- [x] Training global linear regression
- [ ] Validation 
- [ ] Model serialization 

## Referrences

- [Face Alignment at 3000 FPS via Regressing Local Binary Features](https://pdfs.semanticscholar.org/d59f/b96a60168f2baec6f5c61b82393576c33fb7.pdf)
- [One Millisecond Face Alignment with an Ensemble of Regression Trees](https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf)
- [Face Alignment by Explicit Shape Regression](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/01/Face-Alignment-by-Explicit-Shape-Regression.pdf)
- [Robust Face Landmark Estimation under Occlusion](http://www.vision.caltech.edu/~xpburgos/papers/ICCV13%20Burgos-Artizzu.pdf)
- [freesouls/face-alignment-at-3000fps](https://github.com/freesouls/face-alignment-at-3000fps)

## Requirements

- Python 3
- OpenCV 3
- Boost 1.64+
- Dlib
- OpenMP

## Installation

### macOS

Python 3

`brew install python3`

Boost

`brew install boost-python --with-python3`

OpenCV 3

```
brew tap homebrew/science
brew install opencv3 --with-python3　--without-python
```

Open `/usr/local/Cellar/opencv/3.3.1/lib/pkgconfig/opencv.pc` and replace `-llibopencv_xxxxxx.3.3.1.dylib` with `-lopencv_xxxxxx`.

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