#pragma once
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
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
			void _init(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, boost::python::numpy::ndarray &mean_shape_ndarray, std::vector<double> &feature_radius);
		public:
			int _num_stages;
			int _num_trees_per_forest;
			int _num_landmarks;
			int _tree_depth;
			std::vector<double> _local_radius_at_stage;
			std::vector<bool> _training_finished_at_stage;
			std::vector<std::vector<randomforest::Forest*>> _forest_at_stage;
			std::vector<std::vector<lbf::liblinear::model*>> _linear_models_x_at_stage;
			std::vector<std::vector<lbf::liblinear::model*>> _linear_models_y_at_stage;
			cv::Mat1d _mean_shape;
			Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, boost::python::numpy::ndarray mean_shape_ndarray, boost::python::list feature_radius);
			Model(int num_stages, int num_trees_per_forest, int tree_depth, int num_landmarks, boost::python::numpy::ndarray mean_shape_ndarray, std::vector<double> &feature_radius);
			Model(std::string filename);
			randomforest::Forest* get_forest(int stage, int landmark_index);
			void set_linear_models(lbf::liblinear::model* model_x, lbf::liblinear::model* model_y, int stage, int landmark_index);
			lbf::liblinear::model* get_linear_model_x_at(int stage, int landmark_index);
			lbf::liblinear::model* get_linear_model_y_at(int stage, int landmark_index);
			void finish_training_at_stage(int stage);
			bool python_save(std::string filename);
			bool python_load(std::string filename);
			boost::python::list python_compute_error(boost::python::numpy::ndarray image_ndarray, 
										boost::python::numpy::ndarray normalized_target_shape_ndarray, 
										boost::python::numpy::ndarray rotation_inv_ndarray, 
										boost::python::numpy::ndarray shift_inv_ndarray);
			std::vector<double> compute_error(cv::Mat1b &image, 
											  cv::Mat1d &target_shape, 
											  cv::Mat1d &rotation_inv, 
											  cv::Mat1d &shift_inv);
			boost::python::numpy::ndarray python_estimate_shape(boost::python::numpy::ndarray image_ndarray);
			boost::python::numpy::ndarray python_get_mean_shape();
			struct liblinear::feature_node* compute_binary_features_at_stage(cv::Mat1b &image, cv::Mat1d &shape, int stage);
		};
	}
}