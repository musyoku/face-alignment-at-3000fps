#include "python/corpus.h"
#include "python/dataset.h"
#include "python/model.h"
#include "python/trainer.h"

using namespace lbf::python;
using boost::python::arg;
using boost::python::args;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(lbf){
	Py_Initialize();
	np::initialize();

	boost::python::class_<Corpus>("corpus")
	.def("get_image", &Corpus::python_get_image)
	.def("get_num_images", &Corpus::get_num_images)
	.def("add", &Corpus::add);

	boost::python::class_<Dataset>("dataset", boost::python::init<Corpus*, Corpus*, np::ndarray, int>((args("training_corpus", "validation_corpus", "mean_shape_ndarray", "augmentation_size"))))
	.def("get_num_training_images", &Dataset::get_num_training_images);
	
	boost::python::class_<Model>("model", boost::python::init<int, int, int, int, boost::python::list>((args("num_stages", "num_trees_per_forest", "tree_depth", "num_landmarks", "feature_radius"))))
	.def("save", &Model::python_save)
	.def("load", &Model::python_load);

	boost::python::class_<Trainer>("trainer", boost::python::init<Dataset*, Model*, int>((args("dataset", "model", "num_features_to_sample"))))
	.def("get_current_estimated_shape", &Trainer::python_get_current_estimated_shape, ((args("data_index"), arg("transform")=true)))
	.def("get_target_shape", &Trainer::python_get_target_shape, ((args("data_index"), arg("transform")=true)))
	.def("estimate_shape_with_only_local_binary_features", &Trainer::python_estimate_shape_with_only_local_binary_features, ((args("stage", "data_index"), arg("transform")=true)))
	.def("train", &Trainer::train)
	.def("train_stage", &Trainer::train_stage)
	.def("train_local_binary_features_at_stage", &Trainer::train_local_binary_features_at_stage);
}