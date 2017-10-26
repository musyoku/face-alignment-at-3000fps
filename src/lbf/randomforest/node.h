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
			int _leaf_identifier;
			int _landmark_index;
			std::set<int> _left_indices;
			std::set<int> _right_indices;
			std::set<int> _assigned_data_indices;
			double _pixel_difference_threshold;
			FeatureLocation _feature_location;
			cv::Point2d _delta_shape;
			Node* _left;
			Node* _right;
			Node(int depth, int landmark_index){
				_left = NULL;
				_right = NULL;
				_depth = depth;
				_is_leaf = false;
				_leaf_identifier = -1;
				_landmark_index = landmark_index;
			};
			bool split(std::set<int> &data_indices,
					   std::vector<FeatureLocation> &sampled_feature_locations, 
					   cv::Mat_<int> &pixel_differences, 
					   std::vector<cv::Mat1d> &regression_targets);
			int identifier();
			bool is_leaf();
			void _update_delta_shape(std::vector<cv::Mat1d> &regression_targets);
			void mark_as_leaf(int leaf_identifier, std::set<int> &data_indices, std::vector<cv::Mat1d> &regression_targets);
		};
	}
}