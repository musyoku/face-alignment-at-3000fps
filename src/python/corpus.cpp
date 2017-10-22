#include <iostream>
#include "corpus.h"

using std::cout;
using std::endl;
namespace np = boost::python::numpy;

namespace lbf {
	namespace python {
		void Corpus::add_training_data(np::ndarray _image, np::ndarray _landmarks){
			auto shape = _image.get_shape();
			auto strides = _image.get_strides();
			cout << shape[0] << ", " << shape[1] << endl;
			cout << "strides[0] == " << strides[0] << endl;
			cout << "strides[1] == " << strides[1] << endl;
			uint8_t *p = reinterpret_cast<uint8_t*>(_image.get_data());
			for (int h = 0; h < 5; ++h) {
				for (int w = 0; w < 5; ++w) {
					uint8_t pixel_value = *(p + h * strides[0] + w * strides[1]);
					cout << h << ", " << w << " = " << (int)pixel_value << endl;
				}
			}
		}
		void Corpus::add_test_data(np::ndarray _image, np::ndarray _landmarks){
			cout << _image.shape(0) << ", " << _image.shape(1) << endl;
			uint8_t *p = reinterpret_cast<uint8_t*>(_image.get_data());
		}
	}
}