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

library: src/Tensor.cpp src/Shape.cpp  src/CSVReader.cpp src/Dataset.cpp src/Loss.cpp src/Math.cpp src/Optimizer.cpp src/Algorithm.cpp
	@$(CXX) $(CXXFLAGS) -c -o obj/Tensor.o  src/Tensor.cpp 
	@$(CXX) $(CXXFLAGS) -c -o obj/Shape.o  src/Shape.cpp 
	@$(CXX) $(CXXFLAGS) -c -o obj/CSVReader.o  src/CSVReader.cpp
	@$(CXX) $(CXXFLAGS) -c -o obj/Dataset.o  src/Dataset.cpp
	@$(CXX) $(CXXFLAGS) -c -o obj/Loss.o  src/Loss.cpp
	@$(CXX) $(CXXFLAGS) -c -o obj/Math.o  src/Math.cpp
	@$(CXX) $(CXXFLAGS) -c -o obj/Optimizer.o  src/Optimizer.cpp
	@$(CXX) $(CXXFLAGS) -c -o obj/Algorithm.o  src/Algorithm.cpp
	@ar rcs lib/libcmind.a obj/Tensor.o obj/Shape.o obj/CSVReader.o obj/Dataset.o obj/Loss.o obj/Math.o obj/Optimizer.o obj/Algorithm.o