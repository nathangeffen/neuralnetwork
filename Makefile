nn-release: nn.cc dp.cc nn.hh dp.hh
	g++ -O3 -Wall nn.cc dp.cc -o nn-release -fopenmp

nn-release-noomp: nn.cc dp.cc nn.hh dp.hh
	g++ -O3 -Wall nn.cc dp.cc -o nn-release-noomp

nn-debug: nn.cc dp.cc nn.hh dp.hh
	g++ -g -Wall nn.cc dp.cc -o nn-debug
