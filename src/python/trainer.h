#pragma once
#include <boost/python/numpy.hpp>
#include "../lbf/common.h"
#include "dataset.h"
#include "model.h"

namespace lbf {
	namespace python {
		class Trainer {
		private:
			int _num_features_to_sample;
			int _num_augmented_data;
			int _augmentation_size;
			std::vector<cv::Mat1d> _augmented_estimated_shapes;		// contains normalized shape
			std::vector<cv::Mat1d> _augmented_target_shapes;		// contains normalized shape
			std::vector<int> _augmented_indices_to_data_index;
			std::vector<std::vector<FeatureLocation>> _sampled_feature_locations_at_stage;
			void _train_forest(int stage, int landmark_index);
			void _compute_pixel_differences(cv::Mat1d &shape,
											cv::Mat1b &image,
											cv::Mat_<int> &pixel_differences,
											std::vector<FeatureLocation> &sampled_feature_locations,
											int data_index, 
											int landmark_index);
			cv::Mat1b & get_image_by_augmented_index(int augmented_data_index);
			int get_data_index_by_augmented_index(int augmented_data_index);
		public:
			Corpus* _training_corpus;
			Corpus* _validation_corpus;
			Model* _model;
			Trainer(Corpus* training_dataset, Corpus* validation_dataset, Model* model, int augmentation_size, int num_features_to_sample);
			void train();
			void train_stage(int stage);
			void train_local_feature_mapping_functions(int stage);
			void train_global_linear_regression_at_stage(int stage, struct liblinear::feature_node** binary_features);
			void evaluate_stage(int stage);
			cv::Mat1d project_current_estimated_shape(int augmented_data_index);
			boost::python::numpy::ndarray python_get_current_estimated_shape(int augmented_data_index, bool transform);
			boost::python::numpy::ndarray python_get_target_shape(int augmented_data_index, bool transform);
			boost::python::numpy::ndarray python_get_validation_estimated_shape(int data_index, bool transform);
			boost::python::numpy::ndarray python_estimate_shape_only_using_local_binary_features(int stage, int augmented_data_index, bool transform);
		};
	}
}