# Makefile

CC = nvcc
CFLAGS = -O3 -arch=sm_35
SRC = src/main.cu \
      src/layers/dense_layer.cu \
      src/layers/activation.cu \
      src/losses/softmax_loss.cu
OBJ = $(SRC:.cu=.o)
TARGET = perceptron_mlp

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)