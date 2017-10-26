#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/set.hpp>
#include <iostream>
#include "../sampler.h"
#include "node.h"

using std::cout;
using std::endl;

namespace lbf {
	namespace randomforest {
		bool Node::split(std::set<int> &data_indices,
						 std::vector<FeatureLocation> &sampled_feature_locations, 
						 cv::Mat_<int> &pixel_differences, 
						 std::vector<cv::Mat1d> &regression_targets_of_data)
		{
			assert(data_indices.size() > 0);
			int num_features = pixel_differences.rows;
			assert(num_features > 0);
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
				int random_index = sampler::uniform_int(0, pixel_differences_of_data.size() * 0.9) + pixel_differences_of_data.size() * 0.05;
				int tmp_threshold = pixel_differences_of_data[random_index];

				// calculate variance of target landmark positons
				std::set<int> tmp_left_indices;
				std::set<int> tmp_right_indices;
				cv::Point2d squared_mean_left(0, 0);
				cv::Point2d squared_mean_right(0, 0);
				cv::Point2d mean_left(0, 0);
				cv::Point2d mean_right(0, 0);

				for(int data_index: data_indices){
					int pixel_difference = pixel_differences(feature_index, data_index);
					cv::Mat1d &regression_target = regression_targets_of_data[data_index];
					double target_x = regression_target(_landmark_index, 0);
					double target_y = regression_target(_landmark_index, 1);

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

				// cout << "var_left = " << var_left << ", " << "var_right = " << var_right << endl;
				// cout << "score = " << score << endl;
				if(score < minimum_score){
					minimum_score = score;
					selected_feature_index = feature_index;
					_left_indices = tmp_left_indices;
					_right_indices = tmp_right_indices;
					_pixel_difference_threshold = tmp_threshold;
					_feature_location = sampled_feature_locations[selected_feature_index];
				}
			}
			// cout << "minimum_score = " << minimum_score << endl;
			// cout << "selected_feature_index = " << selected_feature_index << endl;
			// cout << _left_indices.size() << " : " << _right_indices.size() << endl;
			assert(selected_feature_index != -1);

			if(_left_indices.size() == 0 || _right_indices.size() == 0){
				return false;
			}
			return true;
		}
		int Node::identifier(){
			return _leaf_identifier;		
		}
		bool Node::is_leaf(){
			return _is_leaf;
		}
		void Node::_update_delta_shape(std::vector<cv::Mat1d> &regression_targets_of_data){
			assert(_assigned_data_indices.size() > 0);
			_delta_shape.x = 0;
			_delta_shape.y = 0;
			for(int data_index: _assigned_data_indices){
				cv::Mat1d &regression_target = regression_targets_of_data[data_index];
				_delta_shape.x += regression_target(_landmark_index, 0);
				_delta_shape.y += regression_target(_landmark_index, 1);
			}
			_delta_shape.x /= _assigned_data_indices.size();
			_delta_shape.y /= _assigned_data_indices.size();
		}
		void Node::mark_as_leaf(int leaf_identifier, std::set<int> &data_indices, std::vector<cv::Mat1d> &regression_targets){
			_is_leaf = true;
			_leaf_identifier = leaf_identifier;
			_assigned_data_indices = data_indices;
			_update_delta_shape(regression_targets);
		}
		template <class Archive>
		void Node::serialize(Archive &ar, unsigned int version){
			boost::serialization::split_member(ar, *this, version);
		}
		template void Node::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
		template void Node::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
		void Node::save(boost::archive::binary_oarchive &ar, unsigned int version) const{
			ar & _depth;
			ar & _is_leaf;
			ar & _leaf_identifier;
			ar & _landmark_index;
			ar & _left_indices;
			ar & _right_indices;
			ar & _assigned_data_indices;
			ar & _pixel_difference_threshold;
			ar & _feature_location;
			ar & _delta_shape;
			ar & _left;
			ar & _right;
		}
		void Node::load(boost::archive::binary_iarchive &ar, unsigned int version){
			ar & _depth;
			ar & _is_leaf;
			ar & _leaf_identifier;
			ar & _landmark_index;
			ar & _left_indices;
			ar & _right_indices;
			ar & _assigned_data_indices;
			ar & _pixel_difference_threshold;
			ar & _feature_location;
			ar & _delta_shape;
			ar & _left;
			ar & _right;
		}
	}
}