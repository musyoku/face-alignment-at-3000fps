#pragma once
#include <boost/python/numpy.hpp>
#include "../lbf/common.h"
#include "dataset.h"
#include "model.h"

namespace lbf {
	namespace python {
		class Trainer {
		private:
			Dataset* _dataset;
			Model* _model;
			int _num_features_to_sample;
			int _num_augmented_data;
			std::vector<cv::Mat1d> _augmented_predicted_shapes;
			std::vector<cv::Mat1d> _augmented_target_shapes;
			std::vector<int> _augmented_indices_to_data_index;
			std::vector<std::vector<FeatureLocation>> _sampled_feature_locations_at_stage;
			void _train_forest(int stage, int landmark_index);
			void _compute_pixel_differences(cv::Mat1d &shape,
											cv::Mat_<uint8_t> &image,
											cv::Mat_<int> &pixel_differences,
											std::vector<FeatureLocation> &sampled_feature_locations,
											int data_index, 
											int landmark_index);
			cv::Mat_<uint8_t> & get_image_by_augmented_index(int augmented_data_index);
		public:
			Trainer(Dataset* dataset, Model* model, int num_features_to_sample);
			void train();
			void train_stage(int stage);
			boost::python::numpy::ndarray get_predicted_shape(int augmented_data_index, bool transform);
		};
	}
}