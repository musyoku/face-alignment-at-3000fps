#include "corpus.h"

using std::cout;
using std::endl;
namespace np = boost::python::numpy;

namespace lbf {
	namespace python {
		void Corpus::add_training_data(np::ndarray image_ndarray, np::ndarray shape_ndarray, boost::python::numpy::ndarray normalized_shape_ndarray){
			_add_image_to(image_ndarray, _image_vec_train);
			_add_shape_to(shape_ndarray, _shape_vec_train);
			_add_shape_to(normalized_shape_ndarray, _normalized_shape_vec_train);
		}
		void Corpus::add_test_data(np::ndarray image_ndarray, np::ndarray shape_ndarray){
			_add_image_to(image_ndarray, _image_vec_test);
			_add_shape_to(shape_ndarray, _shape_vec_test);
		}
		void Corpus::_add_image_to(boost::python::numpy::ndarray &image_ndarray, std::vector<cv::Mat_<uint8_t>> &image_vec){
			auto size = image_ndarray.get_shape();
			auto stride = image_ndarray.get_strides();
			cv::Mat_<uint8_t> image(size[0], size[1]);
			for (int h = 0; h < size[0]; ++h) {
				for (int w = 0; w < size[1]; ++w) {
					uint8_t pixel_value = *reinterpret_cast<uint8_t*>(image_ndarray.get_data() + h * stride[0] + w * stride[1]);
					image(h, w) = pixel_value;
				}
			}
			image_vec.push_back(image);
		}
		void Corpus::_add_shape_to(boost::python::numpy::ndarray &shape_ndarray, std::vector<cv::Mat_<double>> &shape_vec){
			auto size = shape_ndarray.get_shape();
			auto stride = shape_ndarray.get_strides();
			cv::Mat_<double> shape(size[0], size[1]);
			for (int h = 0; h < size[0]; ++h) {
				for (int w = 0; w < size[1]; ++w) {
					double coord = *reinterpret_cast<double*>(shape_ndarray.get_data() + h * stride[0] + w * stride[1]);
					shape(h, w) = coord;
				}
			}
			shape_vec.push_back(shape);
		}
		int Corpus::get_num_training_images(){
			return _image_vec_train.size();
		}
		int Corpus::get_num_test_images(){
			return _image_vec_test.size();
		}
	}
}