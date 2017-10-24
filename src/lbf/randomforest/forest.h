#pragma once
#include <vector>
#include "../common.h"
#include "tree.h"

namespace lbf {
	namespace randomforest {
		class Forest {
		public:
			int _stage;
			int _landmark_index;
			int _num_trees;
			int _num_features_to_sample;
			int _num_total_leaves;
			double _radius;
			std::vector<Tree*> _trees;
			Forest(){};
			Forest(int stage, int landmark_index, int num_trees, double radius, int tree_depth);
			void train(std::vector<FeatureLocation> &sampled_feature_locations, 
					   cv::Mat_<int> &pixel_differences, 
					   std::vector<cv::Mat_<double>> &target_shapes);
			void predict(cv::Mat_<double> &shape, cv::Mat_<uint8_t> &image, std::vector<Node*> &leaves);
			Tree* get_tree_at(int tree_index);
			int get_num_trees();
			int get_num_total_leaves();
		};
	}
}