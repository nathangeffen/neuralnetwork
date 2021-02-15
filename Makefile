CXX       := g++
CXXFLAGS := -Wall -Wextra -g

SRC_CC     := testing.cc nn.cc dp.cc
SRC_HH   := nn.hh dp.hh
INCLUDE :=
LIB     :=
LIBRARIES   :=
EXECUTABLE  := nn


all: $(EXECUTABLE)

$(EXECUTABLE): $(SRC_CC) $(SRC_HH)
	$(CXX) $(CXXFLAGS) $(SRC_CC) -o $(EXECUTABLE)

clean:
	rm -f *.o $(EXECUTABLE)
