#pragma once
#include <boost/python/numpy.hpp>
#include <opencv/opencv.hpp>

namespace lbf {
	namespace python {
		class Corpus {
		public:
			std::vector<cv::Mat_<uint8_t>> _training_images;
			std::vector<cv::Mat_<uint8_t>> _test_images;
			std::vector<std::vector<cv::Point2d>> _training_landmarks;
			std::vector<std::vector<cv::Point2d>> _test_landmarks;
			void add_training_data(boost::python::numpy::ndarray _image, boost::python::numpy::ndarray _landmarks);
			void add_test_data(boost::python::numpy::ndarray _image, boost::python::numpy::ndarray _landmarks);
		};
	}
}