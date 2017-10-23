#include "forest.h"

namespace lbf {
	namespace randomforest {
		Forest::Forest(int stage, int landmark_index, int num_trees, double radius, int tree_depth){
			_stage = stage;
			_num_trees = num_trees;
			_radius = radius;
			_landmark_index = landmark_index;

			_trees.reserve(num_trees);
			for(int n = 0;n < num_trees;n++){
				Tree* tree = new Tree(tree_depth);
				_trees.push_back(tree);
			}
		}
		void Forest::train(std::vector<FeatureLocation> &sampled_feature_locations, cv::Mat_<int> &pixel_differences, std::vector<cv::Mat_<double>> &target_shapes){
			for(int tree_index = 0;tree_index < _num_trees;tree_index++){

			}
		}
	}
}