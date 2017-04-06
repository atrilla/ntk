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
	@$(foreach t, $(TESTS), gcc -std=c99 -I src -o $(basename $(t)) $(t) dist/$(OBJLIB); mv $(basename $(t)) build/;)

clean:
	rm -rf build/ dist/
