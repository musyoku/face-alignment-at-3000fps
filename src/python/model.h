#pragma once
#include <boost/python.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <vector>
#include "../lbf/liblinear/linear.h"
#include "../lbf/randomforest/forest.h"

namespace lbf {
	namespace python {
		class Model {
		private:
			friend class boost::serialization::access;
			template <class Archive>
			void serialize(Archive &archive, unsigned int version);
			void save(boost::archive::binary_oarchive &archive, unsigned int version) const;
			void save_liblinear_models(boost::archive::binary_oarchive &ar, const std::vector<std::vector<lbf::liblinear::model*>> &linear_models_at_stage) const;
			void load(boost::archive::binary_iarchive &archive, unsigned int version);
			void load_liblinear_models(boost::archive::binary_iarchive &ar, std::vector<std::vector<lbf::liblinear::model*>> &linear_models_at_stage);
		public:
			int _num_stages;
			int _num_trees_per_forest;
			int _num_landmarks;
			int _tree_depth;
			std::vector<double> _local_radius_at_stage;
			std::vector<std::vector<randomforest::Forest*>> _forest_at_stage;
			std::vector<std::vector<lbf::liblinear::model*>> _linear_models_x_at_stage;
			std::vector<std::vector<lbf::liblinear::model*>> _linear_models_y_at_stage;
			Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, boost::python::list feature_radius);
			Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, std::vector<double> &feature_radius);
			randomforest::Forest* get_forest(int stage, int landmark_index);
			void set_linear_models(lbf::liblinear::model* model_x, lbf::liblinear::model* model_y, int stage, int landmark_index);
			lbf::liblinear::model* get_linear_model_x_at(int stage, int landmark_index);
			lbf::liblinear::model* get_linear_model_y_at(int stage, int landmark_index);
			bool python_save(std::string filename);
			bool python_load(std::string filename);
		};
	}
}