SOURCES = $(wildcard src/*.c)
TESTS = $(wildcard test/*.c)

OBJLIB = libntk.a

default: buildtest

buildlib:
	mkdir build
	gcc -std=c99 -Iinclude -pedantic -g -O -Wall -c $(SOURCES)
	mv *.o build/
	mkdir dist
	ar -cvq dist/$(OBJLIB) build/*.o

buildtest: buildlib
	gcc -std=c99 -Iinclude -o $(basename $(TESTS)) $(TESTS) dist/$(OBJLIB)
	mv $(basename $(TESTS)) build/

doc:
	doxygen config/ntk.dox.cfg

clean:
	rm -rf build/ dist/ doc/
