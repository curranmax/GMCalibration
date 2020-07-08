
CC = g++
CFLAGS = -Wall -std=c++11

TT_SOURCES = main.cpp vec.cpp matrix.cpp simp_plane.cpp gm_model.cpp dot.cpp link_tracking.cpp
TT_OBJECTS = $(TT_SOURCES:.cpp=.o)
TT_NAME = tracking_test

BF_SOURCES = brute_force_main.cpp brute_force.cpp vec.cpp matrix.cpp simp_plane.cpp gm_model.cpp dot.cpp timer.cpp args.cpp
BF_OBJECTS = $(BF_SOURCES:.cpp=.o)
BF_NAME = brute_force

AG_SOURCES = adjust_gm.cpp simple_opt.cpp utils.cpp vec.cpp matrix.cpp gm_model.cpp simp_plane.cpp
AG_OBJECTS = $(AG_SOURCES:.cpp=.o)
AG_NAME = adjust_gm

all: $(AG_NAME)

$(TT_NAME): $(TT_OBJECTS)
	$(CC) $(CFLAGS) $(TT_OBJECTS) -o $(TT_NAME)

$(BF_NAME): $(BF_OBJECTS)
	$(CC) $(CFLAGS) $(BF_OBJECTS) -lncurses -lpthread -o $(BF_NAME)

$(AG_NAME): $(AG_OBJECTS)
	$(CC) $(CFLAGS) $(AG_OBJECTS)  -o $(AG_NAME)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(TT_NAME) $(BF_NAME) $(AG_NAME) *.o