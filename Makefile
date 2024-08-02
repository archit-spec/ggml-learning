CCC = gcc
SOURCE = simple.c
OBJECTS = simple.o
LIBS = -lggml -lm
INTERFACES = -I /home/dumball/code/ggml/include/
LIBPATH = -L /home/dumball/code/ggml/src

all:
	$(CCC) $(INTERFACES) $(SOURCE) -o simple $(LIBPATH) $(LIBS)
