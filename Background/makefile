NAME = background
CC = g++
CFLAGS = -Wall
#LIBS = -lopencv_video -lopencv_core -lopencv_highgui -lopencv_imgproc
#LPATH = -L/usr/local/lib
LIBS = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video
IPATH = -I.
all:
	$(CC) $(IPATH) $(NAME).cpp $(CFLAGS) $(LIBS) -o $(NAME)
