#pragma once
#include <boost/serialization/serialization.hpp>
#include <opencv2/opencv.hpp>
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
			int _landmark_index;
			friend class boost::serialization::access;
			template <class Archive>
			void serialize(Archive &ar, unsigned int version);
		public:
			int _max_depth;
			Tree(){};
			Tree(int max_depth, int landmark_index);
			void train(std::set<int> &data_indices,
					   std::vector<FeatureLocation> &sampled_feature_locations, 
					   cv::Mat_<int> &pixel_differences, 
					   std::vector<cv::Mat1d> &regression_targets);
			void split_node(Node* node,
							std::set<int> &data_indices,
							std::vector<FeatureLocation> &sampled_feature_locations, 
							cv::Mat_<int> &pixel_differences, 
							std::vector<cv::Mat1d> &regression_targets);
			int get_num_leaves();
			Node* predict(cv::Mat1d &shape, cv::Mat1b &image);
		};
	}
}