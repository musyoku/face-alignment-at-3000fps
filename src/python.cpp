#include "python/corpus.h"

using namespace lbf::python;

BOOST_PYTHON_MODULE(lbf){
	boost::python::numpy::initialize();
	boost::python::class_<Corpus>("corpus")
	.def("add_test_data", &Corpus::add_test_data)
	.def("add_training_data", &Corpus::add_training_data);
}