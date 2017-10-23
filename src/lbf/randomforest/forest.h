#pragma once
#include <vector>
#include "../common.h"
#include "tree.h"

namespace lbf {
	namespace randomforest {
		class Forest {
		public:
			int _stage;
			int _landmark_index;
			int _num_trees;
			int _num_features_to_sample;
			double _radius;
			std::vector<Tree*> _trees;
			Forest(){};
			Forest(int stage, int landmark_index, int num_trees, double radius, int tree_depth);
		};
	}
}