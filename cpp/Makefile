
CC = g++-4.9
CFLAGS = -Wall -std=c++11

TT_SOURCES = main.cpp vec.cpp matrix.cpp simp_plane.cpp gm_model.cpp dot.cpp link_tracking.cpp
TT_OBJECTS = $(TT_SOURCES:.cpp=.o)
TT_NAME = tracking_test

BF_SOURCES = brute_force_main.cpp brute_force.cpp vec.cpp matrix.cpp simp_plane.cpp gm_model.cpp dot.cpp timer.cpp
BF_OBJECTS = $(BF_SOURCES:.cpp=.o)
BF_NAME = brute_force

all: $(BF_NAME)

$(TT_NAME): $(TT_OBJECTS)
	$(CC) $(CFLAGS) $(TT_OBJECTS) -o $(TT_NAME)

$(BF_NAME): $(BF_OBJECTS)
	$(CC) $(CFLAGS) $(BF_OBJECTS) -lncurses -lpthread -o $(BF_NAME)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(TT_NAME) $(BF_NAME) *.o