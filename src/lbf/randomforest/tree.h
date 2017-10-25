#pragma once
#include <opencv/opencv.hpp>
#include <vector>
#include <set>
#include "node.h"

namespace lbf {
	namespace randomforest {
		class Tree {
		private:
			Node* _root;
			int _autoincrement_leaf_index;
			int _num_leaves;
		public:
			int _max_depth;
			Tree(){};
			Tree(int max_depth);
			void train(std::set<int> &data_indices,
					   std::vector<FeatureLocation> &sampled_feature_locations, 
					   cv::Mat_<int> &pixel_differences, 
					   std::vector<cv::Mat1d> &target_shapes);
			void split_node(Node* node,
							std::set<int> &data_indices,
							std::vector<FeatureLocation> &sampled_feature_locations, 
							cv::Mat_<int> &pixel_differences, 
							std::vector<cv::Mat1d> &target_shapes);
			int get_num_leaves();
			Node* predict(cv::Mat1d &shape, cv::Mat1b &image, int landmark_index);
		};
	}
}