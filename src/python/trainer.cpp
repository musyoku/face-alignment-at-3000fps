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

				for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
					Forest* forest = _model->get_forest_of(stage, landmark_index);

					// calculate pixel differece features
					cv::Mat_<int> pixel_differences(_num_features_to_sample, num_data);

					for(int data_index = 0;data_index < num_data;data_index++){
						for(int feature_index = 0;feature_index < _num_features_to_sample;feature_index++){
							FeatureLocation &local_location = sampled_feature_locations[feature_index];

							// a
							double local_x_a = local_location.a.x;

						}
					}

				}

			}
		}
	}
}