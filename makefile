# Specify the compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

clean: 
	@rm -f obj/*.o
	@rm -f lib/*.a
	@rm -f Test
	
example: Examples/Test.cpp
	@g++ -o Test Examples/Test.cpp -L./lib -lcmind
	@./Test

library: src/Tensor.cpp src/Shape.cpp 
	@$(CXX) $(CXXFLAGS) -c -o obj/Tensor.o  src/Tensor.cpp 
	@$(CXX) $(CXXFLAGS) -c -o obj/Shape.o  src/Shape.cpp 
	@ar rcs lib/libcmind.a obj/Tensor.o obj/Shape.o 