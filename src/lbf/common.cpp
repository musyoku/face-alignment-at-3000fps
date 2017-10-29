#include "common.h"

namespace cv {
	cv::Mat1d point_to_mat(cv::Point2d point){
		cv::Mat1d mat(2, 1);
		mat(0, 0) = point.x;
		mat(1, 0) = point.y;
		return mat;
	}
}

namespace boost {
	namespace python {
		template <typename T>
		boost::python::list vector_to_list(std::vector<T> &vector) {
			boost::python::list list;
			for (auto iter = vector.begin(); iter != vector.end(); ++iter) {
				list.append(*iter);
			}
			return list;
		}
		template boost::python::list vector_to_list(std::vector<double> &vector);
	}
}

namespace lbf {

	FeatureLocation::FeatureLocation(cv::Point2d _a, cv::Point2d _b){
		a = _a;
		b = _b;
	}
	FeatureLocation::FeatureLocation(){
		a = cv::Point2d(0.0, 0.0);
		b = cv::Point2d(0.0, 0.0);
	};

	namespace utils {
		template <typename T>
		cv::Mat_<T> ndarray_matrix_to_cv_matrix(boost::python::numpy::ndarray &array){
			auto size = array.get_shape();
			auto stride = array.get_strides();
			cv::Mat_<T> mat(size[0], size[1]);
			for (int h = 0; h < size[0]; ++h) {
				for (int w = 0; w < size[1]; ++w) {
					T value = *reinterpret_cast<T*>(array.get_data() + h * stride[0] + w * stride[1]);
					mat(h, w) = value;
				}
			}
			return mat;
		}
		template cv::Mat1b ndarray_matrix_to_cv_matrix(boost::python::numpy::ndarray &array);
		template cv::Mat1d ndarray_matrix_to_cv_matrix(boost::python::numpy::ndarray &array);

		template <typename T>
		cv::Mat_<T> ndarray_vector_to_cv_matrix(boost::python::numpy::ndarray &array){
			auto size = array.get_shape();
			auto stride = array.get_strides();
			cv::Mat_<T> mat(size[0], 1);
			for (int h = 0; h < size[0]; ++h) {
				T value = *reinterpret_cast<T*>(array.get_data() + h * stride[0]);
				mat(h, 0) = value;
			}
			return mat;
		}
		template cv::Mat1b ndarray_vector_to_cv_matrix(boost::python::numpy::ndarray &array);
		template cv::Mat1d ndarray_vector_to_cv_matrix(boost::python::numpy::ndarray &array);

		cv::Mat1d project_shape(cv::Mat1d shape, cv::Mat1d &rotation, cv::Mat1d &shift){
			assert(shape.cols == 2);
			assert(rotation.rows == 2 && rotation.cols == 2);
			assert(shift.rows == 2 && shift.cols == 1);
			cv::Mat1d shape_T(shape.cols, shape.rows);
			cv::transpose(shape, shape_T);
			shape = rotation * shape_T;
			for (int w = 0; w < shape.cols; w++) {
				shape.col(w) += shift;
			}
			cv::transpose(shape, shape_T);
			return shape_T;
		}
		cv::Mat1d project_shape(cv::Mat1d &shape, cv::Mat1d &rotation, cv::Point2d &shift_point){
			cv::Mat1d shift = cv::point_to_mat(shift_point);
			return project_shape(shape, rotation, shift);
		}
		boost::python::numpy::ndarray cv_matrix_to_ndarray_matrix(cv::Mat1d &cv_matrix){
			boost::python::tuple size = boost::python::make_tuple(cv_matrix.rows, cv_matrix.cols);
			boost::python::numpy::ndarray ndarray = boost::python::numpy::zeros(size, boost::python::numpy::dtype::get_builtin<double>());
			for(int h = 0;h < cv_matrix.rows;h++) {
				for(int w = 0;w < cv_matrix.cols;w++) {
					ndarray[h][w] = cv_matrix(h, w);
				}
			}
			return ndarray;
		}
	}
}