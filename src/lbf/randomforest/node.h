#pragma once
#include <boost/serialization/serialization.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include "../common.h"

namespace lbf {
	namespace randomforest {
		class Tree;
		class Node {
		private:
			friend class boost::serialization::access;
			template <class Archive>
			void serialize(Archive &ar, unsigned int version);
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
			Tree* _tree;
			Node(){};
			Node(int depth, int landmark_index, Tree* tree){
				_left = NULL;
				_right = NULL;
				_depth = depth;
				_is_leaf = false;
				_leaf_identifier = -1;
				_landmark_index = landmark_index;
				_pixel_difference_threshold = 0;
				_tree = tree;
			};
			bool split(std::set<int> &data_indices,
					   std::vector<FeatureLocation> &sampled_feature_locations, 
					   cv::Mat_<int> &pixel_differences, 
					   std::vector<cv::Mat1d> &regression_targets,
					   std::set<int> &_selected_feature_indices_of_all_nodes);
			int identifier();
			bool is_leaf();
			void _update_delta_shape(std::vector<cv::Mat1d> &regression_targets);
			void mark_as_leaf(int leaf_identifier, std::set<int> &data_indices, std::vector<cv::Mat1d> &regression_targets);
		};
	}
}