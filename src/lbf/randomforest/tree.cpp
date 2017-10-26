#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "forest.h"

namespace lbf {
	namespace randomforest {
		Tree::Tree(int max_depth, int landmark_index){
			_max_depth = max_depth;
			_autoincrement_leaf_index = 0;
			_root = new Node(1, landmark_index);
			_num_leaves = 0;
			_landmark_index = landmark_index;
		}
		void Tree::train(std::set<int> &data_indices,
						 std::vector<FeatureLocation> &sampled_feature_locations, 
						 cv::Mat_<int> &pixel_differences, 
						 std::vector<cv::Mat1d> &regression_targets)
		{
			assert(data_indices.size() > 0);
			split_node(_root, data_indices, sampled_feature_locations, pixel_differences, regression_targets);
		}
		void Tree::split_node(Node* node, 
							  std::set<int> &data_indices,
							  std::vector<FeatureLocation> &sampled_feature_locations, 
							  cv::Mat_<int> &pixel_differences, 
							  std::vector<cv::Mat1d> &regression_targets)
		{
			assert(data_indices.size() > 0);
			if(node->_depth > _max_depth){
				node->mark_as_leaf(_autoincrement_leaf_index, data_indices, regression_targets);
				_autoincrement_leaf_index++;
				_num_leaves++;
				return;
			}
			bool need_to_split = node->split(data_indices, sampled_feature_locations, pixel_differences, regression_targets);
			if(need_to_split == false){
				node->mark_as_leaf(_autoincrement_leaf_index, data_indices, regression_targets);
				_autoincrement_leaf_index++;
				_num_leaves++;
				return;
			}
			node->_is_leaf = false;
			assert(node->_left_indices.size() > 0);
			assert(node->_right_indices.size() > 0);

			node->_left = new Node(node->_depth + 1, _landmark_index);
			node->_right = new Node(node->_depth + 1, _landmark_index);

			split_node(node->_left, node->_left_indices, sampled_feature_locations, pixel_differences, regression_targets);
			split_node(node->_right, node->_right_indices, sampled_feature_locations, pixel_differences, regression_targets);
		}
		int Tree::get_num_leaves(){
			return _num_leaves;
		}
		Node* Tree::predict(cv::Mat1d &shape, cv::Mat1b &image){
			int image_width = image.rows;
			int image_height = image.cols;
			assert(_landmark_index < shape.rows);
			double landmark_x = shape(_landmark_index, 0);	// [-1, 1] : origin is the center of the image
			double landmark_y = shape(_landmark_index, 1);	// [-1, 1] : origin is the center of the image

			Node* node = _root;

			while(node->_is_leaf == false){
				FeatureLocation &local_location = node->_feature_location; // [-1, 1] : origin is the landmark position

				// a
				double local_x_a = local_location.a.x + landmark_x;	// [-1, 1] : origin is the center of the image
				double local_y_a = local_location.a.y + landmark_y;
				int pixel_x_a = (image_width / 2.0) + local_x_a * (image_width / 2.0);	// [0, image_width]
				int pixel_y_a = (image_height / 2.0) + local_y_a * (image_height / 2.0);

				// b
				double local_x_b = local_location.b.x + landmark_x;
				double local_y_b = local_location.b.y + landmark_y;
				int pixel_x_b = (image_width / 2.0) + local_x_b * (image_width / 2.0);
				int pixel_y_b = (image_height / 2.0) + local_y_b * (image_height / 2.0);

				// clip bounds
				pixel_x_a = std::max(0, std::min(pixel_x_a, image_width));
				pixel_y_a = std::max(0, std::min(pixel_y_a, image_height));
				pixel_x_b = std::max(0, std::min(pixel_x_b, image_width));
				pixel_y_b = std::max(0, std::min(pixel_y_b, image_height));

				// get pixel value
				int luminosity_a = image(pixel_x_a, pixel_y_a);
				int luminosity_b = image(pixel_x_b, pixel_y_b);

				// pixel difference feature
				int diff = luminosity_a - luminosity_b;

				// select child
				if(diff < node->_pixel_difference_threshold){
					assert(node->_left != NULL);
					node = node->_left;
					continue;
				}
				assert(node->_right != NULL);
				node = node->_right;
			}
			return node;
		}
		template <class Archive>
		void Tree::serialize(Archive &ar, unsigned int version){
			ar & _autoincrement_leaf_index;
			ar & _num_leaves;
			ar & _landmark_index;
			ar & _root;
		}
		template void Tree::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
		template void Tree::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
	}
}