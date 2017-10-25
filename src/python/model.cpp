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
	}
}