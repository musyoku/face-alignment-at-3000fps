#include "forest.h"

namespace lbf {
	namespace randomforest {
		Tree::Tree(int max_depth){
			_max_depth = max_depth;
		}
		void Tree::train(std::set<int> &data_indices,
			 std::vector<FeatureLocation> &sampled_feature_locations, 
			 cv::Mat_<int> &pixel_differences, 
			 std::vector<cv::Mat_<double>> &target_shapes)
		{

		}
	}
}