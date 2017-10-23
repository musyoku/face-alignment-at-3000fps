#include "forest.h"

namespace lbf {
	namespace randomforest {
		Tree::Tree(int max_depth){
			_max_depth = max_depth;
			_root = NULL;
		}
		void Tree::train(std::set<int> &data_indices,
			 std::vector<FeatureLocation> &sampled_feature_locations, 
			 cv::Mat_<int> &pixel_differences, 
			 std::vector<cv::Mat_<double>> &target_shapes)
		{
			_root = new Node(1);
			split_node(_root, data_indices, sampled_feature_locations, pixel_differences, target_shapes);
		}
		void Tree::split_node(Node* node, 
							  std::set<int> &data_indices,
							  std::vector<FeatureLocation> &sampled_feature_locations, 
							  cv::Mat_<int> &pixel_differences, 
							  std::vector<cv::Mat_<double>> &target_shapes)
		{
			if(node->_depth > _max_depth){
				return;
			}
			bool need_to_split = _root->split(data_indices, sampled_feature_locations, pixel_differences, target_shapes);
			if(need_to_split == false){
				return;
			}
			node->_left = new Node(node->_depth + 1);
			node->_right = new Node(node->_depth + 1);
			// split_node(node->_left, node->_left_indices, sampled_feature_locations, pixel_differences, target_shapes);
			// split_node(node->_right, node->_right_indices, sampled_feature_locations, pixel_differences, target_shapes);
		}
	}
}