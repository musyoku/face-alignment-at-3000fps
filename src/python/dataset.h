#pragma once
#include <opencv2/opencv.hpp>
#include "corpus.h"

namespace lbf {
	namespace python {
		class Dataset {
		public:
			Corpus* _corpus;
			int _augmentation_size;
			std::vector<std::vector<int>> _augmented_initial_shape_indices_of_data;
			Dataset(Corpus* corpus, int augmentation_size);
			int get_num_images();
		};
	}
}