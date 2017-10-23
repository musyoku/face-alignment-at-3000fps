#pragma once
#include <opencv/opencv.hpp>
#include <vector>
#include <set>

namespace lbf {
	namespace randomforest {
		class Tree {
		public:
			int _max_depth;
			Tree(){};
			Tree(int max_depth);
			void train(std::set<int> &data_indices,
					   std::vector<FeatureLocation> &sampled_feature_locations, 
					   cv::Mat_<int> &pixel_differences, 
					   std::vector<cv::Mat_<double>> &target_shapes);
		};
	}
}