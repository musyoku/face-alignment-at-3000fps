#pragma once
#include <boost/serialization/serialization.hpp>
#include <vector>
#include "../common.h"
#include "tree.h"

namespace lbf {
	namespace randomforest {
		class Forest {
		private:
			friend class boost::serialization::access;
			template <class Archive>
			void serialize(Archive &ar, unsigned int version);
		public:
			int _stage;
			int _landmark_index;
			int _num_trees;
			int _num_total_leaves;
			double _radius;
			std::vector<Tree*> _trees;
			Forest(){};
			~Forest();
			Forest(int stage, int landmark_index, int num_trees, double radius, int tree_depth);
			void train(std::vector<FeatureLocation> &sampled_feature_locations, 
					   cv::Mat_<int> &pixel_differences, 
					   std::vector<cv::Mat1d> &regression_targets);
			void predict(cv::Mat1d &shape, cv::Mat1b &image, std::vector<Node*> &leaves);
			Tree* get_tree_at(int tree_index);
			int get_num_trees();
			int get_num_total_leaves();
			int enumerate_num_total_leaves();
		};
	}
}