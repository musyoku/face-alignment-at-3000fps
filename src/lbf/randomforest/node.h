#pragma once
#include <opencv/opencv.hpp>
#include <vector>
#include <set>
#include "../common.h"

namespace lbf {
	namespace randomforest {
		class Node {
		public:
			int _depth;
			int _is_leaf;
			int _leaf_index;
			std::set<int> _left_indices;
			std::set<int> _right_indices;
			double _pixel_difference_threshold;
			FeatureLocation _feature_location;
			Node* _left;
			Node* _right;
			Node(int depth, int leaf_index){
				_left = NULL;
				_right = NULL;
				_depth = depth;
				_is_leaf = false;
				_leaf_index = leaf_index;
			};
			bool split(std::set<int> &data_indices,
					   std::vector<FeatureLocation> &sampled_feature_locations, 
					   cv::Mat_<int> &pixel_differences, 
					   std::vector<cv::Mat_<double>> &target_shapes_of_data);
		};
	}
}