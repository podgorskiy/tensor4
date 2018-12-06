CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-O3 -std=c++14 -flto -ffast-math -m64 -I "../../include/" -I "../common/"
LDFLAGS=
LDLIBS=

SRCS=AlexNet.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: alex_net_example

alex_net_example: $(OBJS)
	$(CXX) $(LDFLAGS) -o alexnet $(OBJS) $(LDLIBS) 

AlexNet.o: AlexNet.cpp

main.o: main.cpp

clean:
	$(RM) $(OBJS)
