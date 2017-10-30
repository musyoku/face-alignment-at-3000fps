#include "../lbf/common.h"
#include "corpus.h"

using std::cout;
using std::endl;
namespace np = boost::python::numpy;

namespace lbf {
	namespace python {
		void Corpus::add(np::ndarray image_ndarray, 
						 np::ndarray shape_ndarray, 
						 np::ndarray normalized_shape_ndarray,
						 np::ndarray rotation,
						 np::ndarray rotation_inv,
						 np::ndarray shift,
						 np::ndarray shift_inv,
					 	 double normalized_pupil_distance)
		{
			_add_ndarray_matrix_to(image_ndarray, _images);
			_add_ndarray_matrix_to(shape_ndarray, _shapes);
			_add_ndarray_matrix_to(normalized_shape_ndarray, _normalized_shapes);
			_add_ndarray_matrix_to(rotation, _rotation);
			_add_ndarray_matrix_to(rotation_inv, _rotation_inv);
			_add_ndarray_point_to(shift, _shift);
			_add_ndarray_point_to(shift_inv, _shift_inv);

			_normalized_pupil_distances.push_back(normalized_pupil_distance);

			assert(_images.size() == _shift.size());
			assert(_images.size() == _normalized_shapes.size());
			assert(_images.size() == _rotation.size());
			assert(_images.size() == _rotation_inv.size());
			assert(_images.size() == _normalized_pupil_distances.size());
		}
		template <typename T>
		void Corpus::_add_ndarray_matrix_to(np::ndarray &array, std::vector<cv::Mat_<T>> &corpus){
			auto size = array.get_shape();
			auto stride = array.get_strides();
			cv::Mat_<T> mat(size[0], size[1]);
			for (int h = 0; h < size[0]; ++h) {
				for (int w = 0; w < size[1]; ++w) {
					T value = *reinterpret_cast<T*>(array.get_data() + h * stride[0] + w * stride[1]);
					mat(h, w) = value;
				}
			}
			corpus.push_back(mat);
		}
		void Corpus::_add_ndarray_point_to(np::ndarray &array, std::vector<cv::Point2d> &corpus){
			auto size = array.get_shape();
			auto stride = array.get_strides();
			cv::Point2d point;
			point.x = *reinterpret_cast<double*>(array.get_data());
			point.y = *reinterpret_cast<double*>(array.get_data() + stride[0]);
			corpus.push_back(point);
		}
		int Corpus::get_num_images(){
			return _images.size();
		}
		cv::Mat1d & Corpus::get_original_shape(int data_index){
			assert(data_index < _shapes.size());
			return _shapes[data_index];
		}
		cv::Mat1d & Corpus::get_normalized_shape(int data_index){
			assert(data_index < _normalized_shapes.size());
			return _normalized_shapes[data_index];
		}
		cv::Mat1b & Corpus::get_image(int data_index){
			assert(data_index < _images.size());
			return _images[data_index];
		}
		np::ndarray Corpus::python_get_image(int data_index){
			assert(data_index < _images.size());
			cv::Mat1b &image = _images[data_index];
			return utils::cv_matrix_to_ndarray_matrix(image);
		}
		np::ndarray Corpus::python_get_normalized_shape(int data_index){
			assert(data_index < _normalized_shapes.size());
			cv::Mat1d &normalized_shape = _normalized_shapes[data_index];
			return utils::cv_matrix_to_ndarray_matrix(normalized_shape);
		}
		cv::Mat1d & Corpus::get_rotation(int data_index){
			assert(data_index < _rotation.size());
			return _rotation[data_index];
		}
		cv::Mat1d & Corpus::get_rotation_inv(int data_index){
			assert(data_index < _rotation_inv.size());
			return _rotation_inv[data_index];
		}
		np::ndarray Corpus::python_get_rotation_inv(int data_index){
			assert(data_index < _images.size());
			cv::Mat1d &rotation_inv = _rotation_inv[data_index];
			return utils::cv_matrix_to_ndarray_matrix(rotation_inv);
		}
		cv::Point2d & Corpus::get_shift(int data_index){
			assert(data_index < _shift.size());
			return _shift[data_index];
		}
		cv::Point2d & Corpus::get_shift_inv(int data_index){
			assert(data_index < _shift_inv.size());
			return _shift_inv[data_index];
		}
		double Corpus::get_normalized_pupil_distance(int data_index){
			assert(data_index < _normalized_pupil_distances.size());
			return _normalized_pupil_distances[data_index];
		}
		np::ndarray Corpus::python_get_shift_inv(int data_index){
			assert(data_index < _images.size());
			cv::Point2d &shift_inv = _shift_inv[data_index];

			boost::python::tuple size = boost::python::make_tuple(2);
			np::ndarray shift_inv_ndarray = np::zeros(size, np::dtype::get_builtin<uchar>());
			shift_inv_ndarray[0] = shift_inv.x;
			shift_inv_ndarray[1] = shift_inv.y;
			return shift_inv_ndarray;
		}
	}
}