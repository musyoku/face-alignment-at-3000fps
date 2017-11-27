#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <cassert>
#include <iostream>
#include "model.h"

using namespace lbf::randomforest;
namespace np = boost::python::numpy;

namespace lbf {
	namespace python {
		Model::Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, np::ndarray mean_shape_ndarray, boost::python::list feature_radius){
			std::vector<double> feature_radius_vector;
			int num_words = boost::python::len(feature_radius);
			for(int i = 0;i < num_words;i++){
				double radius = boost::python::extract<double>(feature_radius[i]);
				feature_radius_vector.push_back(radius);
			}
			_init(num_stages, num_trees_per_forest, tree_depth, num_landmarks, mean_shape_ndarray, feature_radius_vector);
		}
		Model::Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, np::ndarray mean_shape_ndarray, std::vector<double> &feature_radius){
			_init(num_stages, num_trees_per_forest, tree_depth, num_landmarks, mean_shape_ndarray, feature_radius);
		}
		Model::~Model(){
			for(auto &forests: _forest_at_stage){
				for(auto forest: forests){
					delete forest;
				}
			}
			for(auto &linear_models: _linear_models_x_at_stage){
				for(auto model: linear_models){
					delete model;
				}
			}
			for(auto &linear_models: _linear_models_y_at_stage){
				for(auto model: linear_models){
					delete model;
				}
			}
		}
		void Model::_init(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, np::ndarray &mean_shape_ndarray, std::vector<double> &feature_radius){
			_num_stages = num_stages;
			_num_trees_per_forest = num_trees_per_forest;
			_num_landmarks = num_landmarks;
			_tree_depth = tree_depth;
			_local_radius_at_stage = feature_radius;

			// convert mean shape to cv::Mat
			auto size = mean_shape_ndarray.get_shape();
			auto stride = mean_shape_ndarray.get_strides();
			cv::Mat1d mean_shape(size[0], size[1]);
			for (int h = 0; h < size[0]; ++h) {
				for (int w = 0; w < size[1]; ++w) {
					double coord = *reinterpret_cast<double*>(mean_shape_ndarray.get_data() + h * stride[0] + w * stride[1]);
					mean_shape(h, w) = coord;
				}
			}
			_mean_shape = mean_shape;

			// build forests
			_forest_at_stage.resize(num_stages);
			for(int stage = 0;stage < _num_stages;stage++){
				std::vector<Forest*> &forest_of_landmark = _forest_at_stage[stage];
				forest_of_landmark.resize(num_landmarks);
				for(int landmark_index = 0;landmark_index < num_landmarks;landmark_index++){
					Forest* forest = new Forest(stage, landmark_index, _num_trees_per_forest, _local_radius_at_stage[stage], _tree_depth);
					forest_of_landmark[landmark_index] = forest;
				}
			}

			// liblinear
			_linear_models_x_at_stage.resize(num_stages);
			_linear_models_y_at_stage.resize(num_stages);
			for(int stage = 0;stage < _num_stages;stage++){
				std::vector<lbf::liblinear::model*> &linear_models_x = _linear_models_x_at_stage[stage];
				std::vector<lbf::liblinear::model*> &linear_models_y = _linear_models_y_at_stage[stage];
				linear_models_x.resize(num_landmarks);
				linear_models_y.resize(num_landmarks);
				for(int landmark_index = 0;landmark_index < num_landmarks;landmark_index++){
					linear_models_x[landmark_index] = NULL;
					linear_models_y[landmark_index] = NULL;
				}
			}

			_training_finished_at_stage.resize(num_stages);
			for(int stage = 0;stage < num_stages;stage++){
				_training_finished_at_stage[stage] = false;
			}
		}
		Model::Model(std::string filename){
			if(python_load(filename) == false){
				std::cout << filename << " not found." << std::endl;
				exit(0);
			}
		}
		void Model::set_num_stages(int num_stages){
			_num_stages = num_stages;
		}
		void Model::finish_training_at_stage(int stage){
			assert(stage < _num_stages);
			_training_finished_at_stage[stage] = true;
		}
		Forest* Model::get_forest(int stage, int landmark_index){
			assert(stage < _num_stages);
			assert(landmark_index < _num_landmarks);
			std::vector<Forest*> &forest_of_landmark = _forest_at_stage[stage];
			assert(forest_of_landmark.size() == _num_landmarks);
			return forest_of_landmark[landmark_index];
		}
		void Model::set_linear_models(lbf::liblinear::model* model_x, lbf::liblinear::model* model_y, int stage, int landmark_index){
			assert(stage < _linear_models_x_at_stage.size());
			std::vector<lbf::liblinear::model*> &linear_models_x = _linear_models_x_at_stage[stage];
			std::vector<lbf::liblinear::model*> &linear_models_y = _linear_models_y_at_stage[stage];
			assert(landmark_index < linear_models_x.size());
			linear_models_x[landmark_index] = model_x;
			linear_models_y[landmark_index] = model_y;
		}
		lbf::liblinear::model* Model::get_linear_model_x_at(int stage, int landmark_index){
			assert(stage < _linear_models_x_at_stage.size());
			std::vector<lbf::liblinear::model*> &linear_models_x = _linear_models_x_at_stage[stage];
			assert(landmark_index < linear_models_x.size());
			return linear_models_x[landmark_index];
		}
		lbf::liblinear::model* Model::get_linear_model_y_at(int stage, int landmark_index){
			assert(stage < _linear_models_y_at_stage.size());
			std::vector<lbf::liblinear::model*> &linear_models_y = _linear_models_y_at_stage[stage];
			assert(landmark_index < linear_models_y.size());
			return linear_models_y[landmark_index];
		}
		template <class Archive>
		void Model::serialize(Archive &ar, unsigned int version){
			boost::serialization::split_member(ar, *this, version);
		}
		template void Model::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
		template void Model::serialize(boost::archive::binary_oarchive &ar, unsigned int version);

		void Model::save(boost::archive::binary_oarchive &ar, unsigned int version) const {
			ar & _num_stages;
			ar & _num_trees_per_forest;
			ar & _num_landmarks;
			ar & _tree_depth;
			ar & _local_radius_at_stage;
			ar & _forest_at_stage;
			ar & _training_finished_at_stage;
			ar & _mean_shape.rows;
			ar & _mean_shape.cols;
			for(int h = 0;h < _mean_shape.rows;h++){
				for(int w = 0;w < _mean_shape.cols;w++){
					ar & _mean_shape(h, w);
				}
			}
			save_liblinear_models(ar, _linear_models_x_at_stage);
			save_liblinear_models(ar, _linear_models_y_at_stage);
		}
		void Model::save_liblinear_models(boost::archive::binary_oarchive &ar, const std::vector<std::vector<lbf::liblinear::model*>> &linear_models_at_stage) const {
			for(int stage = 0;stage < _num_stages;stage++){
				const std::vector<lbf::liblinear::model*> &linear_models = linear_models_at_stage[stage];
				for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){
					lbf::liblinear::model const* model = linear_models[landmark_index];
					bool skip_flag = true ? model == NULL : false;
					ar & skip_flag;
					if(skip_flag){
						continue;
					}
					const lbf::liblinear::parameter &param = model->param;
					int nr_feature = model->nr_feature;

					ar & param.solver_type;
					ar & model->nr_class;
					ar & model->bias;

					int w_size = nr_feature;
					if(model->bias >= 0){
						w_size = nr_feature + 1;
					}
					int nr_w = model->nr_class;
					if(model->nr_class == 2 && param.solver_type != lbf::liblinear::MCSVM_CS){
						nr_w = 1;
					}
					ar & nr_feature;
					ar & nr_w;
					ar & w_size;
					for(int i = 0;i < w_size;i++){
						for(int j = 0;j < nr_w;j++){
							ar & model->w[i * nr_w + j];
						}
					}
				}
			}
		}
		void Model::load(boost::archive::binary_iarchive &ar, unsigned int version){
			ar & _num_stages;
			ar & _num_trees_per_forest;
			ar & _num_landmarks;
			ar & _tree_depth;
			ar & _local_radius_at_stage;
			ar & _forest_at_stage;
			ar & _training_finished_at_stage;

			for(auto forests: _forest_at_stage){
				for(auto forest: forests){
					// std::cout << forest->get_num_total_leaves() << std::endl;
				}
			}

			int rows = 0;
			int cols = 0;
			ar & rows;
			ar & cols;
			_mean_shape = cv::Mat1d(rows, cols);
			for(int h = 0;h < rows;h++){
				for(int w = 0;w < cols;w++){
					double value;
					ar & value;
					_mean_shape(h, w) = value;
				}
			}
			load_liblinear_models(ar, _linear_models_x_at_stage);
			load_liblinear_models(ar, _linear_models_y_at_stage);
		}
		void Model::load_liblinear_models(boost::archive::binary_iarchive &ar, std::vector<std::vector<lbf::liblinear::model*>> &linear_models_at_stage){
			linear_models_at_stage.clear();
			linear_models_at_stage.resize(_num_stages);

			for(int stage = 0;stage < _num_stages;stage++){

				std::vector<lbf::liblinear::model*> &linear_models = linear_models_at_stage[stage];
				linear_models.clear();
				linear_models.resize(_num_landmarks);

				for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){
					bool skip_flag = true;
					ar & skip_flag;
					if(skip_flag){
						continue;
					}

					lbf::liblinear::model* model = new lbf::liblinear::model;
					ar & model->param.solver_type;
					ar & model->nr_class;
					ar & model->bias;
					
					int nr_w = 0;
					int w_size = 0;
					ar & model->nr_feature;
					ar & nr_w;
					ar & w_size;
					model->w = new double[w_size * nr_w];
					for(int i = 0;i < w_size;i++){
						for(int j = 0;j < nr_w;j++){
							ar & model->w[i * nr_w + j];
						}
					}
					linear_models[landmark_index] = model;
				}
			}
		}
		bool Model::python_save(std::string filename){
			bool success = false;
			std::ofstream ofs(filename);
			if(ofs.good()){
				boost::archive::binary_oarchive oarchive(ofs);
				oarchive << *this;
				success = true;
			}
			ofs.close();
			return success;
		}
		bool Model::python_load(std::string filename){
			bool success = false;
			std::ifstream ifs(filename);
			if(ifs.good()){
				boost::archive::binary_iarchive iarchive(ifs);
				iarchive >> *this;
				success = true;
			}
			ifs.close();
			return success;
		}
		np::ndarray Model::python_estimate_shape(np::ndarray image_ndarray){
			cv::Mat1b image = utils::ndarray_matrix_to_cv_matrix<uchar>(image_ndarray);
			cv::Mat1d estimated_shape = _mean_shape.clone();

			for(int stage = 0;stage < _num_stages;stage++){
				if(_training_finished_at_stage[stage] == false){
					continue;
				}

				struct liblinear::feature_node* binary_features = compute_binary_features_at_stage(image, estimated_shape, stage);

				for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){

					struct liblinear::model* model_x = get_linear_model_x_at(stage, landmark_index);
					struct liblinear::model* model_y = get_linear_model_y_at(stage, landmark_index);

					assert(model_x != NULL);
					assert(model_y != NULL);

					double delta_x = liblinear::predict(model_x, binary_features);
					double delta_y = liblinear::predict(model_y, binary_features);

					estimated_shape(landmark_index, 0) += delta_x;
					estimated_shape(landmark_index, 1) += delta_y;
				}
				delete[] binary_features;
			}

			return utils::cv_matrix_to_ndarray_matrix(estimated_shape);
		}
		boost::python::numpy::ndarray Model::python_estimate_shape_using_initial_shape(
			boost::python::numpy::ndarray image_ndarray,
			boost::python::numpy::ndarray initial_shape_ndarray)
		{
			cv::Mat1b image = utils::ndarray_matrix_to_cv_matrix<uchar>(image_ndarray);
			cv::Mat1d estimated_shape = utils::ndarray_matrix_to_cv_matrix<double>(initial_shape_ndarray);

			for(int stage = 0;stage < _num_stages;stage++){
				if(_training_finished_at_stage[stage] == false){
					continue;
				}

				struct liblinear::feature_node* binary_features = compute_binary_features_at_stage(image, estimated_shape, stage);

				for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){

					struct liblinear::model* model_x = get_linear_model_x_at(stage, landmark_index);
					struct liblinear::model* model_y = get_linear_model_y_at(stage, landmark_index);

					assert(model_x != NULL);
					assert(model_y != NULL);

					double delta_x = liblinear::predict(model_x, binary_features);
					double delta_y = liblinear::predict(model_y, binary_features);

					estimated_shape(landmark_index, 0) += delta_x;
					estimated_shape(landmark_index, 1) += delta_y;
				}
				delete[] binary_features;
			}

			return utils::cv_matrix_to_ndarray_matrix(estimated_shape);
		}
		np::ndarray Model::python_estimate_shape_by_translation(
			np::ndarray image_ndarray, 
			np::ndarray rotation_inv_ndarray, 
			np::ndarray shift_inv_ndarray)
		{
			using namespace std;
			cv::Mat1b image = utils::ndarray_matrix_to_cv_matrix<uchar>(image_ndarray);
			cv::Mat1d rotation_inv = utils::ndarray_matrix_to_cv_matrix<double>(rotation_inv_ndarray);
			cv::Mat1d shift_inv = utils::ndarray_vector_to_cv_matrix<double>(shift_inv_ndarray);
			cv::Mat1d estimated_shape = _mean_shape.clone();
			
			for(int stage = 0;stage < _num_stages;stage++){
				if(_training_finished_at_stage[stage] == false){
					continue;
				}

				cv::Mat1d projected_estimated_shape = utils::project_shape(estimated_shape, rotation_inv, shift_inv);
				struct liblinear::feature_node* binary_features = compute_binary_features_at_stage(image, projected_estimated_shape, stage);

				for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){

					struct liblinear::model* model_x = get_linear_model_x_at(stage, landmark_index);
					struct liblinear::model* model_y = get_linear_model_y_at(stage, landmark_index);

					assert(model_x != NULL);
					assert(model_y != NULL);

					double delta_x = liblinear::predict(model_x, binary_features);
					double delta_y = liblinear::predict(model_y, binary_features);

					estimated_shape(landmark_index, 0) += delta_x;
					estimated_shape(landmark_index, 1) += delta_y;
				}
				delete[] binary_features;
			}

			return utils::cv_matrix_to_ndarray_matrix(estimated_shape);
		}
		np::ndarray Model::python_get_mean_shape(){
			return utils::cv_matrix_to_ndarray_matrix(_mean_shape);
		}
		struct liblinear::feature_node* Model::compute_binary_features_at_stage(cv::Mat1b &image, cv::Mat1d &shape, int stage){
			assert(shape.rows == _num_landmarks && shape.cols == 2);
			
			int num_total_trees = 0;
			int num_total_leaves = 0;
			for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){
				Forest* forest = get_forest(stage, landmark_index);
				num_total_trees += forest->get_num_trees();
				num_total_leaves += forest->get_num_total_leaves();
			}

			struct liblinear::feature_node* binary_features = new liblinear::feature_node[num_total_trees + 1];
			int feature_offset = 1;		// start with 1
			int feature_pointer = 0;

			for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){
				// find leaves
				Forest* forest = get_forest(stage, landmark_index);
				std::vector<Node*> leaves;
				forest->predict(shape, image, leaves);
				assert(leaves.size() == forest->get_num_trees());
				// delta_shape
				for(int tree_index = 0;tree_index < forest->get_num_trees();tree_index++){
					Tree* tree = forest->get_tree_at(tree_index);
					int num_leaves = tree->get_num_leaves();
					Node* leaf = leaves[tree_index];
					assert(feature_pointer < num_total_trees + 1);
					liblinear::feature_node &feature = binary_features[feature_pointer];
					feature.index = feature_offset + leaf->identifier();
					feature.value = 1.0;	// binary feature
					feature_pointer++;
					feature_offset += tree->get_num_leaves();
				}
			}
			liblinear::feature_node &feature = binary_features[feature_pointer];
			feature.index = -1;
			feature.value = -1;
			return binary_features;
		}
		boost::python::list Model::python_compute_error(np::ndarray image_ndarray, 
													    np::ndarray normalized_target_shape_ndarray, 
													    np::ndarray rotation_inv_ndarray, 
													    np::ndarray shift_inv_ndarray,
													    double normalized_pupil_distance)
		{
			cv::Mat1b image = utils::ndarray_matrix_to_cv_matrix<uchar>(image_ndarray);
			cv::Mat1d target_shape = utils::ndarray_matrix_to_cv_matrix<double>(normalized_target_shape_ndarray);
			cv::Mat1d rotation_inv = utils::ndarray_matrix_to_cv_matrix<double>(rotation_inv_ndarray);
			cv::Mat1d shift_inv = utils::ndarray_vector_to_cv_matrix<double>(shift_inv_ndarray);
			std::vector<double> error_at_stage = compute_error(image, target_shape, rotation_inv, shift_inv, normalized_pupil_distance);
			return boost::python::vector_to_list(error_at_stage);
		}
		std::vector<double> Model::compute_error(cv::Mat1b &image, 
												 cv::Mat1d &target_shape, 
												 cv::Mat1d &rotation_inv, 
												 cv::Mat1d &shift_inv,
												 double normalized_pupil_distance)
		{
			assert(target_shape.rows == _num_landmarks && target_shape.cols == 2);
			assert(rotation_inv.rows == 2 && rotation_inv.cols == 2);
			assert(shift_inv.rows == 2 && shift_inv.cols == 1);

			cv::Mat1d estimated_shape = _mean_shape.clone();
			std::vector<double> error_at_stage;

			for(int stage = 0;stage < _num_stages;stage++){
				if(_training_finished_at_stage[stage] == false){
					continue;
				}

				cv::Mat1d projected_estimated_shape = utils::project_shape(estimated_shape, rotation_inv, shift_inv);
				struct liblinear::feature_node* binary_features = compute_binary_features_at_stage(image, projected_estimated_shape, stage);
				double error = 0;

				for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){

					struct liblinear::model* model_x = get_linear_model_x_at(stage, landmark_index);
					struct liblinear::model* model_y = get_linear_model_y_at(stage, landmark_index);

					assert(model_x != NULL);
					assert(model_y != NULL);

					double delta_x = liblinear::predict(model_x, binary_features);
					double delta_y = liblinear::predict(model_y, binary_features);

					// update shape
					estimated_shape(landmark_index, 0) += delta_x;
					estimated_shape(landmark_index, 1) += delta_y;

					// compute error
					double error_x = target_shape(landmark_index, 0) - estimated_shape(landmark_index, 0);
					double error_y = target_shape(landmark_index, 1) - estimated_shape(landmark_index, 1);
					error += std::sqrt(error_x * error_x + error_y * error_y);
				}
				error_at_stage.push_back(error / _num_landmarks / normalized_pupil_distance * 100);
				delete[] binary_features;
			}
			return error_at_stage;
		}
	}
}