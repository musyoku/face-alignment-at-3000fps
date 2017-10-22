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

- [HELEN](http://www.ifp.illinois.edu/~vuongle2/helen/)
- [300-W](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)