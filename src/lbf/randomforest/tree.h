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
					   std::vector<cv::Mat_<double>> &target_shapes);
			void split_node(Node* node,
							std::set<int> &data_indices,
							std::vector<FeatureLocation> &sampled_feature_locations, 
							cv::Mat_<int> &pixel_differences, 
							std::vector<cv::Mat_<double>> &target_shapes);
			int get_num_leaves();
			Node* predict(cv::Mat_<double> &shape, cv::Mat_<uint8_t> &image, int landmark_index);
		};
	}
}