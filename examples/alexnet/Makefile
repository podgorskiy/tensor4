CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-O3 -std=c++11 -m64 -I "../../include/" -I "../common/"
LDFLAGS=-O3
LDLIBS=

SRCS=AlexNet.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: alex_net_example

alex_net_example: $(OBJS)
	$(CXX) $(LDFLAGS) -o alexnet $(OBJS) $(LDLIBS) 

tool.o: tool.cc support.hh

AlexNet.o: AlexNet.cpp

main.o: main.cpp

clean:
	$(RM) $(OBJS)
