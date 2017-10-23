#pragma once
#include "dataset.h"
#include "model.h"

namespace lbf {
	namespace python {
		class Trainer {
		public:
			Trainer(Dataset* dataset, Model* model);
		};
	}
}