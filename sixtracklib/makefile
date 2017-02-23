CFLAGS= -O1 -fPIC
CFLAGS= -g -fPIC -Wall -Werror
CFLAGS= -std=c99 -O3 -fPIC -msse -Wall -Werror
CC=gcc
LIBS=-lm


all: lib

lib: block.so


block.o: block.c block.h beam.h track.c track.h value.h
	$(CC) $(CFLAGS) -c -o $@ $<

block.so: block.o
	$(CC)  -shared $(LIBS) -o $@ $^

clean:
	-rm block.so block.so block.o
