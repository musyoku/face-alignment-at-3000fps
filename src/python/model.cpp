#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <cassert>
#include "model.h"

namespace lbf {
	using namespace randomforest;
	namespace python {
		Model::Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, boost::python::list feature_radius){
			_num_stages = num_stages;
			_num_trees_per_forest = num_trees_per_forest;
			_num_landmarks = num_landmarks;
			_tree_depth = tree_depth;

			assert(num_stages == boost::python::len(feature_radius));
			for(int stage = 0;stage < num_stages;stage++){
				double radius = boost::python::extract<double>(feature_radius[stage]);
				_local_radius_at_stage.push_back(radius);
			}

			_forest_at_stage.resize(num_stages);
			for(int stage = 0;stage < _num_stages;stage++){
				std::vector<Forest*> &forest_of_landmark = _forest_at_stage[stage];
				forest_of_landmark.resize(num_landmarks);
				for(int landmark_index = 0;landmark_index < num_landmarks;landmark_index++){
					Forest* forest = new Forest(stage, landmark_index, _num_trees_per_forest, _local_radius_at_stage[stage], _tree_depth);
					forest_of_landmark[landmark_index] = forest;
				}
			}

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
		}
		Model::Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, std::vector<double> &feature_radius){
			_num_stages = num_stages;
			_num_trees_per_forest = num_trees_per_forest;
			_num_landmarks = num_landmarks;
			_tree_depth = tree_depth;
			_local_radius_at_stage = feature_radius;

			_forest_at_stage.resize(num_stages);
			for(int stage = 0;stage < _num_stages;stage++){
				std::vector<Forest*> &forest_of_landmark = _forest_at_stage[stage];
				forest_of_landmark.resize(num_landmarks);
				for(int landmark_index = 0;landmark_index < num_landmarks;landmark_index++){
					Forest* forest = new Forest(stage, landmark_index, _num_trees_per_forest, _local_radius_at_stage[stage], _tree_depth);
					forest_of_landmark[landmark_index] = forest;
				}
			}

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
			save_liblinear_models(ar, _linear_models_x_at_stage);
			save_liblinear_models(ar, _linear_models_y_at_stage);
		}
		void Model::save_liblinear_models(boost::archive::binary_oarchive &ar, const std::vector<std::vector<lbf::liblinear::model*>> &linear_models_at_stage) const {
			for(int stage = 0;stage < _num_stages;stage++){
				const std::vector<lbf::liblinear::model*> &linear_models = linear_models_at_stage[stage];
				for(int landmark_index = 0;landmark_index < _num_landmarks;landmark_index++){
					lbf::liblinear::model const* model = linear_models[landmark_index];
					bool skip_flag = false ? model == NULL : true;
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
					
					int nr_feature = 0;
					int nr_w = 0;
					int w_size = 0;
					ar & nr_feature;
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
	}
}