#pragma once
#include <boost/python/numpy.hpp>
#include <opencv/opencv.hpp>
#include "corpus.h"

namespace lbf {
	namespace python {
		class Dataset {
		public:
			Corpus* _corpus;
			cv::Mat_<double> _mean_shape;
			int _augmentation_size;
			std::vector<std::vector<int>> _initial_shape_indices_of_data;
			Dataset(Corpus* corpus, boost::python::numpy::ndarray mean_shape_ndarray, int augmentation_size);
		};
	}
}