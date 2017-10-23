#pragma once
#include <boost/python/numpy.hpp>
#include <opencv/opencv.hpp>
#include <vector>

namespace lbf {
	namespace python {
		class Corpus {
		private:
			void _add_image_to(boost::python::numpy::ndarray &image_ndarray, std::vector<cv::Mat_<uint8_t>> &image_vec);
			void _add_shape_to(boost::python::numpy::ndarray &shape_ndarray, std::vector<cv::Mat_<double>> &shape_vec);
		public:
			std::vector<cv::Mat_<uint8_t>> _images_train;
			std::vector<cv::Mat_<uint8_t>> _images_test;
			std::vector<cv::Mat_<double>> _shapes_train;
			std::vector<cv::Mat_<double>> _normalized_shapes_train;
			std::vector<cv::Mat_<double>> _shapes_test;
			void add_training_data(boost::python::numpy::ndarray image_ndarray, boost::python::numpy::ndarray shape_ndarray, boost::python::numpy::ndarray normalized_shape_ndarray);
			void add_test_data(boost::python::numpy::ndarray image_ndarray, boost::python::numpy::ndarray shape_ndarray);
			int get_num_training_images();
			int get_num_test_images();
		};
	}
}