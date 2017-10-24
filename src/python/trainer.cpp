#include <boost/python.hpp>
#include <cmath>
#include <iostream>
#include "../lbf/liblinear/linear.h"
#include "../lbf/sampler.h"
#include "../lbf/randomforest/forest.h"
#include "trainer.h"

using std::cout;
using std::endl;

namespace lbf {
	using namespace randomforest;
	namespace python {
		Trainer::Trainer(Dataset* dataset, Model* model, int num_features_to_sample){
			_dataset = dataset;
			_model = model;
			_num_features_to_sample = num_features_to_sample;

			Corpus* corpus = _dataset->_corpus;
			int num_data = corpus->_images_train.size();
			_num_augmented_data = (dataset->_augmentation_size + 1) * num_data;

			// sample feature locations
			for(int stage = 0;stage < model->_num_stages;stage++){
				double localized_radius = model->_local_radius_at_stage[stage];

				std::vector<FeatureLocation> sampled_feature_locations;
				sampled_feature_locations.reserve(num_features_to_sample);

				for(int feature_index = 0;feature_index < num_features_to_sample;feature_index++){
					double r, theta;
					
					r = localized_radius * sampler::uniform(0, 1);
					theta = M_PI * 2.0 * sampler::uniform(0, 1);
					cv::Point2d a(r * std::cos(theta), r * std::cos(theta));
					
					r = localized_radius * sampler::uniform(0, 1);
					theta = M_PI * 2.0 * sampler::uniform(0, 1);
					cv::Point2d b(r * std::cos(theta), r * std::cos(theta));

					FeatureLocation location(a, b);
					sampled_feature_locations.push_back(location);
				}

				_sampled_feature_locations_at_stage.push_back(sampled_feature_locations);
			}

			// set initial shape
			int num_landmarks = model->_num_landmarks;
			int augmentation_size = _dataset->_augmentation_size;

			_augmented_predicted_shapes.resize(_num_augmented_data);
			_augmented_target_shapes.resize(_num_augmented_data);
			_augmented_indices_to_data_index.resize(_num_augmented_data);

			// normalized shape
			for(int data_index = 0;data_index < num_data;data_index++){
				_augmented_predicted_shapes[data_index] = corpus->get_training_normalized_shape_of(data_index);
				_augmented_target_shapes[data_index] = corpus->get_training_shape_of(data_index);
				_augmented_indices_to_data_index[data_index] = data_index;
			}

			// augmented shapes
			for(int data_index = 0;data_index < num_data;data_index++){
				std::vector<int> &initial_shape_indices = _dataset->_augmented_initial_shape_indices_of_data[data_index];
				assert(initial_shape_indices.size() == augmentation_size);
				for(int n = 0;n < augmentation_size;n++){
					int augmented_data_index = (n + 1) * num_data + data_index;
					int shape_index = initial_shape_indices[n];
					_augmented_predicted_shapes[augmented_data_index] = corpus->get_training_normalized_shape_of(shape_index);
					_augmented_target_shapes[augmented_data_index] = corpus->get_training_shape_of(data_index);
					_augmented_indices_to_data_index[augmented_data_index] = data_index;
				}
			}

		}
		void Trainer::train(){
			for(int stage = 0;stage < _model->_num_stages;stage++){
				if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
					return;		
				}
				cout << "training stage: " << stage << " of " << _model->_num_stages << endl;
				// train local binary feature
				for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
					if (PyErr_CheckSignals() != 0) {	// ctrl+cが押されたかチェック
						return;		
					}
					_train_forest(stage, landmark_index);
				}
				// train global linear regression
				int num_trees_in_forest = 0;
				for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
					Forest* forest = _model->get_forest_of(stage, landmark_index);
					num_trees_in_forest += forest->_num_trees;
				}
				cout << "num_trees_in_forest = " << num_trees_in_forest << endl;
				struct liblinear::feature_node** binary_features = new struct liblinear::feature_node*[_num_augmented_data];
				for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
					binary_features[augmented_data_index] = new liblinear::feature_node[num_trees_in_forest + 1]; // with bias
				}

				// predict shape
			}
		}
		cv::Mat_<uint8_t> & Trainer::get_image_by_augmented_index(int augmented_data_index){
			assert(augmented_data_index < _augmented_indices_to_data_index.size());
			int data_index = _augmented_indices_to_data_index[augmented_data_index];
			return _dataset->_corpus->_images_train[data_index];
		}
		void Trainer::_train_forest(int stage, int landmark_index){
			Corpus* corpus = _dataset->_corpus;
			Forest* forest = _model->get_forest_of(stage, landmark_index);

			std::vector<FeatureLocation> &sampled_feature_locations = _sampled_feature_locations_at_stage[stage];
			assert(sampled_feature_locations.size() == _num_features_to_sample);

			int num_data = corpus->_images_train.size();
			int augmentation_size = _dataset->_augmentation_size;

			// pixel differece features
			cv::Mat_<int> pixel_differences(_num_features_to_sample, _num_augmented_data);

			// get pixel differences with normalized shape
			for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
				cv::Mat_<double> &prev_predicted_shape = _augmented_predicted_shapes[augmented_data_index];
				cv::Mat_<uint8_t> &image = get_image_by_augmented_index(augmented_data_index);
				_compute_pixel_differences(prev_predicted_shape, image, pixel_differences, sampled_feature_locations, augmented_data_index, landmark_index);
			}

			forest->train(sampled_feature_locations, pixel_differences, _augmented_target_shapes);
		}
		void Trainer::_compute_pixel_differences(cv::Mat_<double> &shape, 
												 cv::Mat_<uint8_t> &image, 
												 cv::Mat_<int> &pixel_differences, 
												 std::vector<FeatureLocation> &sampled_feature_locations,
												 int data_index, 
												 int landmark_index){
			int image_width = image.rows;
			int image_height = image.cols;
			double landmark_x = shape(landmark_index, 0);	// [-1, 1] : origin is the center of the image
			double landmark_y = shape(landmark_index, 1);	// [-1, 1] : origin is the center of the image

			for(int feature_index = 0;feature_index < _num_features_to_sample;feature_index++){
				FeatureLocation &local_location = sampled_feature_locations[feature_index]; // origin is the landmark position

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

				pixel_differences(feature_index, data_index) = diff;
			}
		}
	}
}