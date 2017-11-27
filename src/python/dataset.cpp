#include "dataset.h"
#include "../lbf/sampler.h"

using std::cout;
using std::endl;

namespace lbf {
	namespace python {
		Dataset::Dataset(Corpus* corpus, int augmentation_size){
			_augmentation_size = augmentation_size;
			_corpus = corpus;

			// sample shapes to augment data
			int num_training_images = corpus->get_num_images();
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
		int Dataset::get_num_images(){
			return _corpus->get_num_images();
		}
	}
}