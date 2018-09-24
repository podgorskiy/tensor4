CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-O3 -std=c++11 -flto -ffast-math -m64 -I "../../include/" -I "../common/"
LDFLAGS=-O3
LDLIBS=

SRCS=ResNet.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: resnet50_example

resnet50_example: $(OBJS)
	$(CXX) $(LDFLAGS) -o resnet50 $(OBJS) $(LDLIBS) 

ResNet.o: ResNet.cpp

main.o: main.cpp

clean:
	$(RM) $(OBJS)
