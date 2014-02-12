NAME = background
CC = g++
CFLAGS = -mmacosx-version-min=10.8 -Wall -g -O0 `pkg-config --cflags opencv`   
INCPATH = -I. 
LIBS = `pkg-config --libs opencv`

all:
	$(CC) $(NAME).cpp $(CFLAGS) $(INCPATH) $(LIBS) -o $(NAME)

clean:
	rm -f *.o 


