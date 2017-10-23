#include <cmath>
#include <iostream>
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

			// sample feature locations
			_sampled_feature_locations_of_stage.resize(model->_num_stages);
			for(int stage = 0;stage < model->_num_stages;stage++){
				double localized_radius = model->_local_radius_of_stage[stage];

				std::vector<FeatureLocation> &sampled_feature_locations = _sampled_feature_locations_of_stage[stage];
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
			}
		}
		void Trainer::train_local_binary_features(){
			Corpus* corpus = _dataset->_corpus;
			int num_data = corpus->_images_train.size();

			for(int stage = 0;stage < _model->_num_stages;stage++){
				std::vector<FeatureLocation> &sampled_feature_locations = _sampled_feature_locations_of_stage[stage];
				assert(sampled_feature_locations.size() == _num_features_to_sample);

				for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
					Forest* forest = _model->get_forest_of(stage, landmark_index);

					// calculate pixel differece features
					cv::Mat_<int> pixel_differences(_num_features_to_sample, num_data);

					for(int data_index = 0;data_index < num_data;data_index++){
						cv::Mat_<double> &shape = corpus->_normalized_shapes_train[data_index];
						cv::Mat_<uint8_t> &image = corpus->_images_train[data_index];

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

					forest->train(pixel_differences);
				}

			}
		}
	}
}