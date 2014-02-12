NAME = video-segment
CC = g++
#CFLAGS = -Wall -g -O0 `pkg-config --cflags opencv`
CFLAGS = 
LIBS = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video

all:
	$(CC) $(NAME).cpp $(CFLAGS) $(LIBS) -o $(NAME)

clean:
	rm $(NAME) 
