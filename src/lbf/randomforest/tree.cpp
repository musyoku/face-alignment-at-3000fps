#include "forest.h"

namespace lbf {
	namespace randomforest {
		Tree::Tree(int max_depth){
			_max_depth = max_depth;
			_autoincrement_leaf_index = 0;
			_root = new Node(1, _autoincrement_leaf_index);
			_autoincrement_leaf_index++;
			_num_leafs = 0;
		}
		void Tree::train(std::set<int> &data_indices,
			 std::vector<FeatureLocation> &sampled_feature_locations, 
			 cv::Mat_<int> &pixel_differences, 
			 std::vector<cv::Mat_<double>> &target_shapes)
		{
			assert(data_indices.size() > 0);
			split_node(_root, data_indices, sampled_feature_locations, pixel_differences, target_shapes);
		}
		void Tree::split_node(Node* node, 
							  std::set<int> &data_indices,
							  std::vector<FeatureLocation> &sampled_feature_locations, 
							  cv::Mat_<int> &pixel_differences, 
							  std::vector<cv::Mat_<double>> &target_shapes)
		{
			assert(data_indices.size() > 0);
			if(node->_depth > _max_depth){
				return;
			}
			bool need_to_split = node->split(data_indices, sampled_feature_locations, pixel_differences, target_shapes);
			if(need_to_split == false){
				_num_leafs++;
				return;
			}
			assert(node->_left_indices.size() > 0);
			assert(node->_right_indices.size() > 0);

			node->_left = new Node(node->_depth + 1, _autoincrement_leaf_index);
			_autoincrement_leaf_index++;
			
			node->_right = new Node(node->_depth + 1, _autoincrement_leaf_index);
			_autoincrement_leaf_index++;

			split_node(node->_left, node->_left_indices, sampled_feature_locations, pixel_differences, target_shapes);
			split_node(node->_right, node->_right_indices, sampled_feature_locations, pixel_differences, target_shapes);
		}
		int Tree::get_num_leafs(){
			return _num_leafs;
		}
	}
}