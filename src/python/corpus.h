#pragma once
#include <boost/python/numpy.hpp>
#include <opencv/opencv.hpp>
#include <vector>

namespace lbf {
	namespace python {
		class Corpus {
		private:
			template <typename T>
			void _add_ndarray_matrix_to(boost::python::numpy::ndarray &array, std::vector<cv::Mat_<T>> &corpus);
			void _add_ndarray_point_to(boost::python::numpy::ndarray &array, std::vector<cv::Point_<double>> &corpus);
		public:
			std::vector<cv::Mat_<uint8_t>> _images_train;
			std::vector<cv::Mat_<uint8_t>> _images_test;
			std::vector<cv::Mat_<double>> _shapes_train;
			std::vector<cv::Mat_<double>> _normalized_shapes_train;
			std::vector<cv::Mat_<double>> _shapes_test;
			std::vector<cv::Mat_<double>> _rotation_train;
			std::vector<cv::Mat_<double>> _rotation_test;
			std::vector<cv::Point_<double>> _shift_train;
			std::vector<cv::Point_<double>> _shift_test;
			void add_training_data(boost::python::numpy::ndarray image_ndarray, 
								   boost::python::numpy::ndarray shape_ndarray, 
								   boost::python::numpy::ndarray normalized_shape_ndarray,
								   boost::python::numpy::ndarray rotation,
								   boost::python::numpy::ndarray shift);
			void add_test_data(boost::python::numpy::ndarray image_ndarray, boost::python::numpy::ndarray shape_ndarray);
			int get_num_training_images();
			int get_num_test_images();
			cv::Mat_<double> & get_training_shape_of(int data_index);
			cv::Mat_<double> & get_training_normalized_shape_of(int data_index);
			cv::Mat_<uint8_t> & get_training_image_of(int data_index);
			boost::python::numpy::ndarray get_training_image(int data_index);
		};
	}
}