#include <chrono>
#include "sampler.h"

namespace lbf {
	namespace sampler{
		int seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 mt(seed);
		void set_seed(int seed){
			mt = std::mt19937(seed);
		}
		double bernoulli(double p){
			std::uniform_real_distribution<double> rand(0, 1);
			double r = rand(mt);
			if(r > p){
				return 0;
			}
			return 1;
		}
		double uniform(double min, double max){
			std::uniform_real_distribution<double> rand(min, max);
			return rand(mt);
		}
		double uniform_int(int min, int max){
			std::uniform_int_distribution<> rand(min, max);
			return rand(mt);
		}
	}
}