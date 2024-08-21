# Specify the compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

clean:
	@rm -f *.o

LinearRegression: LinearRegression.cpp
	@g++ -o LinearRegression LinearRegression.cpp -L../lib -lcmind
	@./LinearRegression

BinaryLogisticRegression: BinaryLogisticRegression.cpp
	@g++ -o BinaryLogisticRegression BinaryLogisticRegression.cpp -L../lib -lcmind
	@./BinaryLogisticRegression