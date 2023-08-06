run:
	gcc-12 mnist.c -o mnist -lpng -lm -Ofast -Wall && ./mnist

test:
	python3 check_against_pytorch.py