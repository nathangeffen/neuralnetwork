CXX       := g++
CXXFLAGS := -Wall -Wextra -g

SRC_CC     := testing.cc nn_net.cc nn_csv.cc
SRC_HH   := nn.hh nn_net.hh nn_csv.hh nn_math.hh nn_matrix.hh nn_preprocess.hh
INCLUDE :=
LIB     :=
LIBRARIES   :=
EXECUTABLE  := nn


all: $(EXECUTABLE)

$(EXECUTABLE): $(SRC_CC) $(SRC_HH)
	$(CXX) $(CXXFLAGS) $(SRC_CC) -o $(EXECUTABLE)

clean:
	rm -f *.o $(EXECUTABLE)
