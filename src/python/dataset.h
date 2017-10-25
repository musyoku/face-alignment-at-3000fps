#pragma once
#include <boost/python/numpy.hpp>
#include <opencv/opencv.hpp>
#include "corpus.h"

namespace lbf {
	namespace python {
		class Dataset {
		public:
			Corpus* _training_corpus;
			Corpus* _validation_corpus;
			cv::Mat1d _mean_shape;
			int _augmentation_size;
			std::vector<std::vector<int>> _augmented_initial_shape_indices_of_data;
			Dataset(Corpus* training_corpus, Corpus* validation_corpus, boost::python::numpy::ndarray mean_shape_ndarray, int augmentation_size);
			int get_num_training_images();
			int get_num_validation_images();
		};
	}
}