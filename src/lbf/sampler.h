#pragma once
#include <random>

namespace lbf {
	namespace sampler {
		extern std::mt19937 mt;
		double bernoulli(double p);
		double uniform(double min, double max);
		double uniform_int(int min, int max);
		void set_seed(int seed);
	}
}