CFLAGS= -O1 -fPIC
CFLAGS= -g -fPIC -Wall -Wextra -Werror
CFLAGS= -std=c99 -O3 -fPIC -msse -Wall -Werror
CC=gcc
LIBS=-lm


all: lib

lib: block.so

block.o: block.c block.h beam.h track.h track.c value.h particle.h
	$(CC) $(CFLAGS) -c -o $@ $<

block.so: block.o
	$(CC)  -shared $(LIBS) -o $@ $^

clean:
	-rm -f block.so *.o
