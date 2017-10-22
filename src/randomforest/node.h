#pragma once
#include "../common.h"

namespace lbf {
	namespace randomforest {
		class Node {
		public:
			int _depth;
			int _is_leaf;
			double _pixel_difference_threshold;
			FeatureLocation* _feature_location;
			Node(){};
			Node(Node* left, Node* right);
		};
	}
}