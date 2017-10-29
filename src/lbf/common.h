#pragma once
#include <boost/python/numpy.hpp>
#include <opencv2/opencv.hpp>

namespace cv {
	cv::Mat1d point_to_mat(cv::Point2d point);
}

namespace boost {
	namespace python {
		template <typename T>
		boost::python::list vector_to_list(std::vector<T> &vector);
	}
}

namespace lbf {
	class FeatureLocation {
	public:
		cv::Point2d a;
		cv::Point2d b;
		FeatureLocation(cv::Point2d _a, cv::Point2d _b);
		FeatureLocation();
	};
	namespace utils {
		template <typename T>
		cv::Mat_<T> ndarray_matrix_to_cv_matrix(boost::python::numpy::ndarray &array);
		template <typename T>
		cv::Mat_<T> ndarray_vector_to_cv_matrix(boost::python::numpy::ndarray &array);
		cv::Mat1d project_shape(cv::Mat1d &shape, cv::Mat1d &rotation, cv::Mat1d &shift);
		cv::Mat1d project_shape(cv::Mat1d &shape, cv::Mat1d &rotation, cv::Point2d &shift_point);
		boost::python::numpy::ndarray cv_matrix_to_ndarray_matrix(cv::Mat1d &cv_matrix);
	}
}