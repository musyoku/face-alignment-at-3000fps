#pragma once
#include <boost/python/numpy.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace lbf {
	namespace python {
		class Corpus {
		private:
			template <typename T>
			void _add_ndarray_matrix_to(boost::python::numpy::ndarray &array, std::vector<cv::Mat_<T>> &corpus);
			void _add_ndarray_point_to(boost::python::numpy::ndarray &array, std::vector<cv::Point2d> &corpus);
		public:
			std::vector<cv::Mat1b> _images;
			std::vector<cv::Mat1d> _shapes;
			std::vector<cv::Mat1d> _normalized_shapes;
			std::vector<cv::Mat1d> _rotation;
			std::vector<cv::Mat1d> _rotation_inv;
			std::vector<cv::Point2d> _shift;
			std::vector<cv::Point2d> _shift_inv;
			void add(boost::python::numpy::ndarray image_ndarray, 
					 boost::python::numpy::ndarray shape_ndarray, 
					 boost::python::numpy::ndarray normalized_shape_ndarray,
					 boost::python::numpy::ndarray rotation,
					 boost::python::numpy::ndarray rotation_inv,
					 boost::python::numpy::ndarray shift,
					 boost::python::numpy::ndarray shift_inv);
			int get_num_images();
			cv::Mat1d & get_original_shape(int data_index);
			cv::Mat1d & get_normalized_shape(int data_index);
			cv::Mat1b & get_image(int data_index);
			cv::Mat1d & get_rotation(int data_index);
			cv::Mat1d & get_rotation_inv(int data_index);
			cv::Point2d & get_shift(int data_index);
			cv::Point2d & get_shift_inv(int data_index);
			boost::python::numpy::ndarray python_get_image(int data_index);
		};
	}
}