CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-O3 -std=c++11 -flto -ffast-math -fopenmp -m64 -I "../../include/" -I "../common/"
LDFLAGS=-lgomp
LDLIBS=

SRCS=DenseNet.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: resnet50_example

densenet169_example: $(OBJS)
	$(CXX) $(LDFLAGS) -o densenet169 $(OBJS) $(LDLIBS) 

DenseNet.o: DenseNet.cpp

main.o: main.cpp

clean:
	$(RM) $(OBJS)
