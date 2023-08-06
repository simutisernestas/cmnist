run:
	gcc-12 mnist.c -o mnist -lpng -lm -Ofast -Wall -Wextra -pedantic && ./mnist

bprof:
	gcc -ggdb3 -Ofast -Wall -Wextra -pedantic -o mnist mnist.c -lpng -lm -DTESTOUT
	valgrind --tool=cachegrind ./mnist
	cg_annotate --auto=yes cachegrind.out.* > log/cachegrind.out
	rm cachegrind.out.*

test:
	mkdir -p log
	gcc-12 mnist.c -o mnist -lpng -lm -Ofast -Wall -Wextra -pedantic -DTESTOUT=1
	./mnist
	python3 check_against_pytorch.py