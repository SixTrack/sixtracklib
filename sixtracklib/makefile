CFLAGS= -O1 -fPIC
CFLAGS= -g -fPIC -Wall -Wextra -Werror
CFLAGS= -std=c99 -O3 -fPIC -msse -Wall -Wextra -pedantic -Werror
CC=gcc
LIBS=-lm

VPATH = ../common

all: lib

lib: block.so

block.o: block.c block.h beam.h track.h value.h particle.h
	$(CC) -I../common $(CFLAGS) -c -o $@ $<

block.so: block.o
	$(CC) -shared $(LIBS) -o $@ $^

clean:
	-rm -f block.so *.o
