#pragma once
#include "../lbf/common.h"
#include "dataset.h"
#include "model.h"

namespace lbf {
	namespace python {
		class Trainer {
		private:
			Dataset* _dataset;
			Model* _model;
			int _num_features_to_sample;
		public:
			std::vector<std::vector<FeatureLocation>> _sampled_feature_locations_of_stage;
			Trainer(Dataset* dataset, Model* model, int num_features_to_sample);
			void train_local_binary_features();
		};
	}
}