#include "dataset.h"
#include "../lbf/sampler.h"

using std::cout;
using std::endl;

namespace lbf {
	namespace python {
		Dataset::Dataset(Corpus* corpus, boost::python::numpy::ndarray mean_shape_ndarray, int augmentation_size){
			_augmentation_size = augmentation_size;
			_corpus = corpus;

			// convert mean shape to cv::Mat
			auto size = mean_shape_ndarray.get_shape();
			auto stride = mean_shape_ndarray.get_strides();
			cv::Mat_<double> mean_shape(size[0], size[1]);
			for (int h = 0; h < size[0]; ++h) {
				for (int w = 0; w < size[1]; ++w) {
					double coord = *reinterpret_cast<double*>(mean_shape_ndarray.get_data() + h * stride[0] + w * stride[1]);
					mean_shape(h, w) = coord;
				}
			}
			_mean_shape = mean_shape;

			// sample shapes to augment data
			int num_training_images = corpus->get_num_training_images();
			_augmented_initial_shape_indices_of_data.reserve(num_training_images);

			for(int data_index = 0;data_index < num_training_images;data_index++){
				std::vector<int> initial_shape_indices;
				initial_shape_indices.reserve(augmentation_size);

				for(int m = 0;m < augmentation_size;m++){
					int index = 0;
					do {
						index = sampler::uniform_int(0, num_training_images - 1);
					} while(index == data_index);	// refect same shape
					initial_shape_indices.push_back(index);
				}

				_augmented_initial_shape_indices_of_data.push_back(initial_shape_indices);
			}
		}
	}
}