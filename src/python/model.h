#pragma once
#include <boost/python.hpp>
#include <vector>
#include "../lbf/randomforest/forest.h"

namespace lbf {
	namespace python {
		class Model {
		public:
			int _num_stages;
			int _num_trees_per_forest;
			int _num_landmarks;
			int _tree_depth;
			std::vector<double> _local_radius_at_stage;
			std::vector<std::vector<randomforest::Forest*>> _forest_at_stage;
			Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, boost::python::list feature_radius);
			Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, std::vector<double> &feature_radius);
			randomforest::Forest* get_forest_of(int stage, int landmark_index);
		};
	}
}