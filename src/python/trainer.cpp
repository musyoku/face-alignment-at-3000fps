#include <boost/python.hpp>
#include <omp.h>
#include <cmath>
#include <iostream>
#include "../lbf/liblinear/linear.h"
#include "../lbf/sampler.h"
#include "../lbf/randomforest/forest.h"
#include "trainer.h"

using std::cout;
using std::endl;
using std::flush;
using namespace lbf::randomforest;
namespace liblinear = lbf::liblinear;
namespace np = boost::python::numpy;

namespace lbf {
	namespace python {
		Trainer::Trainer(Dataset* dataset, Model* model, int num_features_to_sample){
			_dataset = dataset;
			_model = model;
			_num_features_to_sample = num_features_to_sample;

			Corpus* corpus = _dataset->_training_corpus;
			int num_data = corpus->_images.size();
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

			_augmented_estimated_shapes.resize(_num_augmented_data);
			_augmented_target_shapes.resize(_num_augmented_data);
			_augmented_indices_to_data_index.resize(_num_augmented_data);

			// normalized shape
			for(int data_index = 0;data_index < num_data;data_index++){
				_augmented_estimated_shapes[data_index] = _dataset->_mean_shape.clone();	// make a copy
				_augmented_target_shapes[data_index] = corpus->get_normalized_shape(data_index);
				_augmented_indices_to_data_index[data_index] = data_index;
			}

			// augmented shapes
			for(int data_index = 0;data_index < num_data;data_index++){
				std::vector<int> &initial_shape_indices = _dataset->_augmented_initial_shape_indices_of_data[data_index];
				assert(initial_shape_indices.size() == augmentation_size);
				for(int n = 0;n < augmentation_size;n++){
					int augmented_data_index = (n + 1) * num_data + data_index;
					int shape_index = initial_shape_indices[n];
					_augmented_estimated_shapes[augmented_data_index] = corpus->get_normalized_shape(shape_index).clone();	// make a copy
					_augmented_target_shapes[augmented_data_index] = corpus->get_normalized_shape(data_index);
					_augmented_indices_to_data_index[augmented_data_index] = data_index;
				}
			}

		}
		void Trainer::train(){
			for(int stage = 0;stage < _model->_num_stages;stage++){
				train_stage(stage);
			}
		}
		void Trainer::train_local_binary_features_at_stage(int stage){
			cout << "training local binary features ..." << endl;
			#pragma omp parallel for
			for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
				_train_forest(stage, landmark_index);
			}
		}
		void Trainer::train_stage(int stage){
			cout << "training stage: " << (stage + 1) << " of " << _model->_num_stages << endl;
			
			// train local binary feature
			train_local_binary_features_at_stage(stage);
			if (PyErr_CheckSignals() != 0) {
				return;		
			}
			
			// train global linear regression
			cout << "training global linear regression ..." << endl;
			//// setup liblinear
			int num_total_trees = 0;
			int num_total_leaves = 0;
			for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
				Forest* forest = _model->get_forest(stage, landmark_index);
				num_total_trees += forest->get_num_trees();
				num_total_leaves += forest->get_num_total_leaves();
			}
			cout << "#trees = " << num_total_trees << endl;
			cout << "#features = " << num_total_leaves << endl;
			struct liblinear::feature_node** binary_features = new struct liblinear::feature_node*[_num_augmented_data];
			for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
				binary_features[augmented_data_index] = new struct liblinear::feature_node[num_total_trees + 1];
			}
			//// compute binary features
			for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){

				cv::Mat1d estimated_shape = project_current_estimated_shape(augmented_data_index);
				cv::Mat1b &image = get_image_by_augmented_index(augmented_data_index);
				int feature_offset = 1;		// start with 1
				int feature_pointer = 0;

				for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
					// find leaves
					Forest* forest = _model->get_forest(stage, landmark_index);
					std::vector<Node*> leaves;
					forest->predict(estimated_shape, image, leaves);
					assert(leaves.size() == forest->get_num_trees());
					// delta_shape
					for(int tree_index = 0;tree_index < forest->get_num_trees();tree_index++){
						Tree* tree = forest->get_tree_at(tree_index);
						int num_leaves = tree->get_num_leaves();
						Node* leaf = leaves[tree_index];
						assert(feature_pointer < num_total_trees + 1);
						liblinear::feature_node &feature = binary_features[augmented_data_index][feature_pointer];
						feature.index = feature_offset + leaf->identifier();
						feature.value = 1.0;	// binary feature
						feature_pointer++;
						feature_offset += tree->get_num_leaves();
					}
				}
				liblinear::feature_node &feature = binary_features[augmented_data_index][feature_pointer];
				feature.index = -1;
				feature.value = -1;
			}

			if (PyErr_CheckSignals() != 0) {
				return;		
			}

			struct liblinear::problem* problem = new struct liblinear::problem;
			problem->l = _num_augmented_data;
			problem->n = num_total_leaves;
			problem->x = binary_features;
			problem->bias = -1;

			struct liblinear::parameter* parameter = new struct liblinear::parameter;
			parameter->solver_type = liblinear::L2R_L2LOSS_SVR_DUAL;
			parameter->C = 1.0;
			parameter->p = 0;

		    double** targets = new double*[_model->_num_landmarks];
			for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
				targets[landmark_index] = new double[_num_augmented_data];
			}

			// train regressor
			#pragma omp parallel for
			for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
				cout << "." << flush;

				// train x
				for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
					cv::Mat1d &target_shape = _augmented_target_shapes[augmented_data_index];
					cv::Mat1d &estimated_shape = _augmented_estimated_shapes[augmented_data_index];

					double delta_x = target_shape(landmark_index, 0) - estimated_shape(landmark_index, 0);

					targets[landmark_index][augmented_data_index] = delta_x;
				}
				problem->y = targets[landmark_index];
				liblinear::check_parameter(problem, parameter);
		        struct liblinear::model* model_x = liblinear::train(problem, parameter);

				// train y
				for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
					cv::Mat1d &target_shape = _augmented_target_shapes[augmented_data_index];
					cv::Mat1d &estimated_shape = _augmented_estimated_shapes[augmented_data_index];

					double delta_y = target_shape(landmark_index, 1) - estimated_shape(landmark_index, 1);

					targets[landmark_index][augmented_data_index] = delta_y;
				}
				problem->y = targets[landmark_index];
				liblinear::check_parameter(problem, parameter);
		        struct liblinear::model* model_y = liblinear::train(problem, parameter);

		        _model->set_linear_models(model_x, model_y, stage, landmark_index);
			}

			cout << endl;

			// predict shape
			double average_error = 0;
			for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){

				struct liblinear::model* model_x = _model->get_linear_model_x_at(stage, landmark_index);
				struct liblinear::model* model_y = _model->get_linear_model_y_at(stage, landmark_index);

				for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
					cv::Mat1d &target_shape = _augmented_target_shapes[augmented_data_index];
					cv::Mat1d &estimated_shape = _augmented_estimated_shapes[augmented_data_index];
					double delta_x = liblinear::predict(model_x, binary_features[augmented_data_index]);
					double delta_y = liblinear::predict(model_y, binary_features[augmented_data_index]);

					// update shape
					estimated_shape(landmark_index, 0) += delta_x;
					estimated_shape(landmark_index, 1) += delta_y;

					// compute error
					double error_x = target_shape(landmark_index, 0) - estimated_shape(landmark_index, 0);
					double error_y = target_shape(landmark_index, 1) - estimated_shape(landmark_index, 1);
					double error = std::sqrt(error_x * error_x + error_y * error_y);
					average_error += error;
				}
			}
			average_error /= _model->_num_landmarks * _num_augmented_data;
			cout << "Error: " << average_error << endl;

			for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
				delete[] targets[landmark_index];
			}
			delete[] targets;
			for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
				delete[] binary_features[augmented_data_index];
			}
			delete[] binary_features;
		}
		cv::Mat1b & Trainer::get_image_by_augmented_index(int augmented_data_index){
			assert(augmented_data_index < _augmented_indices_to_data_index.size());
			int data_index = _augmented_indices_to_data_index[augmented_data_index];
			return _dataset->_training_corpus->_images[data_index];
		}
		void Trainer::_train_forest(int stage, int landmark_index){
			Corpus* corpus = _dataset->_training_corpus;
			Forest* forest = _model->get_forest(stage, landmark_index);

			std::vector<FeatureLocation> &sampled_feature_locations = _sampled_feature_locations_at_stage[stage];
			assert(sampled_feature_locations.size() == _num_features_to_sample);

			int num_data = corpus->_images.size();
			int augmentation_size = _dataset->_augmentation_size;

			// pixel differece features
			cv::Mat_<int> pixel_differences(_num_features_to_sample, _num_augmented_data);

			// get pixel differences
			for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
				cv::Mat1b &image = get_image_by_augmented_index(augmented_data_index);
				cv::Mat1d projected_shape = project_current_estimated_shape(augmented_data_index);
				_compute_pixel_differences(projected_shape, image, pixel_differences, sampled_feature_locations, augmented_data_index, landmark_index);
			}

			// compute ground truth shape increment	
			std::vector<cv::Mat1d> regression_targets_of_data(_num_augmented_data);	
			for(int augmented_data_index = 0;augmented_data_index < _num_augmented_data;augmented_data_index++){
				cv::Mat1d &target_shape = _augmented_target_shapes[augmented_data_index];
				cv::Mat1d &estimated_shape = _augmented_estimated_shapes[augmented_data_index];
				cv::Mat1d regression_targets = target_shape - estimated_shape;
				regression_targets_of_data[augmented_data_index] = regression_targets;
			}
			forest->train(sampled_feature_locations, pixel_differences, regression_targets_of_data);
		}
		void Trainer::_compute_pixel_differences(cv::Mat1d &shape, 
												 cv::Mat1b &image, 
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
		cv::Mat1d Trainer::project_current_estimated_shape(int augmented_data_index){
			assert(augmented_data_index < _augmented_estimated_shapes.size());
			cv::Mat1d shape = _augmented_estimated_shapes[augmented_data_index].clone();	// make a copy
			Corpus* corpus = _dataset->_training_corpus;
			int data_index = _augmented_indices_to_data_index[augmented_data_index];

			cv::Mat1d &rotation_inv = corpus->get_rotation_inv(data_index);
			cv::Point2d &_shift_inv = corpus->get_shift_inv(data_index);
			cv::Mat1d shift_inv(2, 1);
			shift_inv(0, 0) = _shift_inv.x;
			shift_inv(1, 0) = _shift_inv.y;

			// inverse
			cv::Mat1d shape_T(shape.cols, shape.rows);
			cv::transpose(shape, shape_T);
			shape = rotation_inv * shape_T;
			for (int w = 0; w < shape.cols; ++w) {
				shape.col(w) += shift_inv;
			}
			cv::transpose(shape, shape_T);
			shape = shape_T;

			return shape;
		}
		np::ndarray Trainer::python_get_target_shape(int augmented_data_index, bool transform){
			assert(augmented_data_index < _augmented_target_shapes.size());
			cv::Mat1d shape = _augmented_target_shapes[augmented_data_index];

			if(transform){
				Corpus* corpus = _dataset->_training_corpus;
				int data_index = _augmented_indices_to_data_index[augmented_data_index];

				cv::Mat1d &rotation_inv = corpus->get_rotation_inv(data_index);
				cv::Point2d &_shift_inv = corpus->get_shift_inv(data_index);
				cv::Mat1d shift_inv(2, 1);
				shift_inv(0, 0) = _shift_inv.x;
				shift_inv(1, 0) = _shift_inv.y;

				// inverse
				cv::Mat1d shape_T(shape.cols, shape.rows);
				cv::transpose(shape, shape_T);
				shape = rotation_inv * shape_T;
				for (int w = 0; w < shape.cols; ++w) {
					shape.col(w) += shift_inv;
				}
				cv::transpose(shape, shape_T);
				shape = shape_T;
			}

			boost::python::tuple size = boost::python::make_tuple(shape.rows, shape.cols);
			np::ndarray shape_ndarray = np::zeros(size, np::dtype::get_builtin<double>());
			for(int h = 0;h < shape.rows;h++) {
				for(int w = 0;w < shape.cols;w++) {
					shape_ndarray[h][w] = shape(h, w);
				}
			}
			return shape_ndarray;
		}
		np::ndarray Trainer::python_get_current_estimated_shape(int augmented_data_index, bool transform){
			assert(augmented_data_index < _augmented_estimated_shapes.size());
			cv::Mat1d shape = _augmented_estimated_shapes[augmented_data_index];

			if(transform){
				shape = project_current_estimated_shape(augmented_data_index);
			}

			boost::python::tuple size = boost::python::make_tuple(shape.rows, shape.cols);
			np::ndarray shape_ndarray = np::zeros(size, np::dtype::get_builtin<double>());
			for(int h = 0;h < shape.rows;h++) {
				for(int w = 0;w < shape.cols;w++) {
					shape_ndarray[h][w] = shape(h, w);
				}
			}
			return shape_ndarray;
		}
		np::ndarray Trainer::python_estimate_shape_with_only_local_binary_features(int stage, int augmented_data_index, bool transform){
			assert(augmented_data_index < _augmented_estimated_shapes.size());
			cv::Mat1d shape = _augmented_estimated_shapes[augmented_data_index].clone();


			cv::Mat1d projected_shape = project_current_estimated_shape(augmented_data_index);
			cv::Mat1b &image = get_image_by_augmented_index(augmented_data_index);

			for(int landmark_index = 0;landmark_index < _model->_num_landmarks;landmark_index++){
				// find leaves
				Forest* forest = _model->get_forest(stage, landmark_index);
				std::vector<Node*> leaves;
				forest->predict(projected_shape, image, leaves);
				assert(leaves.size() == forest->get_num_trees());
				cv::Point2d mean_delta;
				mean_delta.x = 0;
				mean_delta.y = 0;
				// delta_shape
				for(int tree_index = 0;tree_index < forest->get_num_trees();tree_index++){
					Node* leaf = leaves[tree_index];
					mean_delta.x += leaf->_delta_shape.x;
					mean_delta.y += leaf->_delta_shape.y;
				}
				mean_delta.x /= forest->get_num_trees();
				mean_delta.y /= forest->get_num_trees();

				shape(landmark_index, 0) += mean_delta.x;
				shape(landmark_index, 1) += mean_delta.y;
			}

			if(transform){
				Corpus* corpus = _dataset->_training_corpus;
				int data_index = _augmented_indices_to_data_index[augmented_data_index];

				cv::Mat1d &rotation_inv = corpus->get_rotation_inv(data_index);
				cv::Point2d &_shift_inv = corpus->get_shift_inv(data_index);
				cv::Mat1d shift_inv(2, 1);
				shift_inv(0, 0) = _shift_inv.x;
				shift_inv(1, 0) = _shift_inv.y;

				// inverse
				cv::Mat1d shape_T(shape.cols, shape.rows);
				cv::transpose(shape, shape_T);
				shape = rotation_inv * shape_T;
				for (int w = 0; w < shape.cols; ++w) {
					shape.col(w) += shift_inv;
				}
				cv::transpose(shape, shape_T);
				shape = shape_T;
			}

			boost::python::tuple size = boost::python::make_tuple(shape.rows, shape.cols);
			np::ndarray shape_ndarray = np::zeros(size, np::dtype::get_builtin<double>());
			for(int h = 0;h < shape.rows;h++) {
				for(int w = 0;w < shape.cols;w++) {
					shape_ndarray[h][w] = shape(h, w);
				}
			}
			return shape_ndarray;
		}
	}
}