#include <iostream>
#include "node.h"

using std::cout;
using std::endl;

namespace lbf {
	namespace randomforest {
		bool Node::split(std::set<int> &data_indices,
			 std::vector<FeatureLocation> &sampled_feature_locations, 
			 cv::Mat_<int> &pixel_differences, 
			 std::vector<cv::Mat_<double>> &target_shapes_of_data)
		{
			int num_features = pixel_differences.rows;
			double minimum_score = 9999999999;
			double selected_feature_index = -1;

			for(int feature_index = 0;feature_index < num_features;feature_index++){
				// select threshold
				std::vector<int> pixel_differences_of_data;
				pixel_differences_of_data.reserve(data_indices.size());

				for(int data_index: data_indices){
					pixel_differences_of_data.push_back(pixel_differences(feature_index, data_index));
				}
				std::sort(pixel_differences_of_data.begin(), pixel_differences_of_data.end());
				int tmp_threshold = pixel_differences_of_data[pixel_differences_of_data.size() / 2];

				// calculate variance of target landmark positons
				std::set<int> tmp_left_indices;
				std::set<int> tmp_right_indices;
				cv::Point2d squared_mean_left(0, 0);
				cv::Point2d squared_mean_right(0, 0);
				cv::Point2d mean_left(0, 0);
				cv::Point2d mean_right(0, 0);

				for(int data_index: data_indices){
					int pixel_difference = pixel_differences(feature_index, data_index);
					cv::Mat_<double> &target_shape = target_shapes_of_data[data_index];
					double target_x = target_shape(feature_index, 0);
					double target_y = target_shape(feature_index, 1);

					// left
					if(pixel_difference < tmp_threshold){
						squared_mean_left.x += target_x * target_x;
						mean_left.x += target_x;
						squared_mean_left.y += target_y * target_y;
						mean_left.y += target_y;
						tmp_left_indices.insert(data_index);
						continue;
					}

					// right
					squared_mean_right.x += target_x * target_x;
					mean_right.x += target_x;
					squared_mean_right.y += target_y * target_y;
					mean_right.y += target_y;
					tmp_right_indices.insert(data_index);
				}

				// compute variance
				double var_left = 0;
				if(tmp_left_indices.size() > 0){
					squared_mean_left /= (double)tmp_left_indices.size();
					mean_left /= (double)tmp_left_indices.size();
					var_left = squared_mean_left.x - mean_left.x * mean_left.x + squared_mean_left.y - mean_left.y * mean_left.y;
				}
				double var_right = 0;
				if(tmp_right_indices.size() > 0){
					squared_mean_right /= (double)tmp_right_indices.size();
					mean_right /= (double)tmp_right_indices.size();
					var_right = squared_mean_right.x - mean_right.x * mean_right.x + squared_mean_right.y - mean_right.y * mean_right.y;
				}

				// compute score
				double sum_squared_error_left = var_left * tmp_left_indices.size();
				double sum_squared_error_right = var_right * tmp_right_indices.size();
				double score = sum_squared_error_left + sum_squared_error_right;

				cout << "var_left = " << var_left << ", " << "var_right = " << var_right << endl;
				cout << "score = " << score << endl;
				if(score < minimum_score){
					minimum_score = score;
					selected_feature_index = feature_index;
					_left_indices = tmp_left_indices;
					_right_indices = tmp_right_indices;
					_pixel_difference_threshold = tmp_threshold;
					_feature_location = sampled_feature_locations[selected_feature_index];
				}
			}
			cout << "minimum_score = " << minimum_score << endl;
			cout << "selected_feature_index = " << selected_feature_index << endl;
			cout << _left_indices.size() << " : " << _right_indices.size() << endl;
			assert(selected_feature_index != -1);

			if(_left_indices.size() == 0 || _right_indices.size() == 0){
				_is_leaf = true;
				return false;
			}
			return true;
		}
	}
}