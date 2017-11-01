CC = g++
BOOST = /home/stark/boost
INCLUDE = `python3-config --includes` `pkg-config --cflags opencv` -std=c++11 -I$(BOOST)/include
LDFLAGS = `python3-config --ldflags` `pkg-config --libs opencv` -lboost_serialization -lboost_numpy3 -lboost_python3 -L$(BOOST)/lib
SOFLAGS = -shared -fPIC -march=native -O3

install: ## Python用ライブラリをコンパイル
	$(CC) -Wno-deprecated $(INCLUDE) $(SOFLAGS) -o run/lbf.so src/python.cpp src/lbf/*.cpp src/lbf/randomforest/*.cpp src/python/*.cpp src/lbf/liblinear/*.cpp src/lbf/liblinear/blas/*.c $(LDFLAGS)

install_ubuntu: ## Python用ライブラリをコンパイル
	$(CC) -Wl,--no-as-needed -Wno-deprecated $(INCLUDE) $(SOFLAGS) -o run/lbf.so src/python.cpp src/lbf/*.cpp src/lbf/randomforest/*.cpp src/python/*.cpp src/lbf/liblinear/*.cpp src/lbf/liblinear/blas/*.c $(LDFLAGS)

check_includes:	## Python.hの場所を確認
	python3-config --includes

check_ldflags:	## libpython3の場所を確認
	python3-config --ldflags

module_tests: ## 各モジュールのテスト.
	$(CC) test/module_tests/randomforest/forest.cpp src/lbf/*.cpp src/lbf/randomforest/*.cpp src/python/*.cpp src/lbf/liblinear/*.cpp src/lbf/liblinear/blas/*.c -o test/module_tests/randomforest/forest $(INCLUDE) $(LDFLAGS) -O0 -g
	./test/module_tests/randomforest/forest

running_tests:	## 学習テスト
	$(CC) test/running_tests/memory.cpp src/lbf/*.cpp src/lbf/randomforest/*.cpp src/python/*.cpp src/lbf/liblinear/*.cpp src/lbf/liblinear/blas/*.c -o test/running_tests/memory $(INCLUDE) $(LDFLAGS) -O3 -fopenmp -Wno-deprecated
	$(CC) test/running_tests/save.cpp src/lbf/*.cpp src/lbf/randomforest/*.cpp src/python/*.cpp src/lbf/liblinear/*.cpp src/lbf/liblinear/blas/*.c -o test/running_tests/save $(INCLUDE) $(LDFLAGS) -O3 -fopenmp -Wno-deprecated
	$(CC) test/running_tests/validation.cpp src/lbf/*.cpp src/lbf/randomforest/*.cpp src/python/*.cpp src/lbf/liblinear/*.cpp src/lbf/liblinear/blas/*.c -o test/running_tests/validation $(INCLUDE) $(LDFLAGS) -O0 -g -fopenmp -Wno-deprecated
	$(CC) test/running_tests/train.cpp src/lbf/*.cpp src/lbf/randomforest/*.cpp src/python/*.cpp src/lbf/liblinear/*.cpp src/lbf/liblinear/blas/*.c -o test/running_tests/train $(INCLUDE) $(LDFLAGS) -O3 -fopenmp -Wno-deprecated

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help