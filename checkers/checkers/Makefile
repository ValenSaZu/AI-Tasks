# Makefile para Juego de Damas en Linux

CC = g++
CFLAGS = -Wall -std=c++11
LIBS = -lglut -lGLU -lGL

all: checkers

checkers: main.cpp CBoard.h CTree.h CGraphics.h
	$(CC) $(CFLAGS) -o checkers linux.cpp $(LIBS)

clean:
	rm -f checkers

.PHONY: all clean
