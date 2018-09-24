CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-O3 -std=c++11 -flto -ffast-math -m64 -I "../../include/" -I "../common/"
LDFLAGS=-O3
LDLIBS=

SRCS=VGG.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: vgg16_example

vgg16_example: $(OBJS)
	$(CXX) $(LDFLAGS) -o vgg16 $(OBJS) $(LDLIBS) 

VGG.o: VGG.cpp

main.o: main.cpp

clean:
	$(RM) $(OBJS)
