#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include "../../../src/lbf/randomforest/forest.h"
#include "../../../src/lbf/randomforest/tree.h"
#include "../../../src/lbf/randomforest/node.h"

int main(){
	Forest(int stage, int landmark_index, int num_trees, double radius, int tree_depth);
	void train(std::vector<FeatureLocation> &sampled_feature_locations, 
			   cv::Mat_<int> &pixel_differences, 
			   std::vector<cv::Mat1d> &target_shapes);
}