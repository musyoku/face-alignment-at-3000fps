#pragma once

namespace lbf {
	namespace randomforest {
		class Tree {
		public:
			int _max_depth;
			Tree(){};
			Tree(int max_depth);
		};
	}
}