all: clean compile

# Specify the compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

clean: 
	@rm -f obj/*.o

compile: src/Tensor.cpp 
	@$(CXX) $(CXXFLAGS) -c -o obj/Tensor.o  src/Tensor.cpp 

library: obj/Tensor.o
	@ar rcs lib/libcmind.a obj/Tensor.o