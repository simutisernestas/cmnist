run:
	gcc mnist.c -o mnist -lpng -lm && ./mnist

test:
	python3 check_against_pytorch.py