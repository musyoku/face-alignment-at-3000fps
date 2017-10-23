#pragma once
#include <vector>
#include "../common.h"
#include "tree.h"

namespace lbf {
	namespace randomforest {
		class Forest {
		public:
			int _stage;
			int _num_features_to_sample;
			double _radius;
			std::vector<Tree*> _trees;
			std::vector<FeatureLocation*> _sampled_local_positions;
			Forest(){};
		};
	}
}