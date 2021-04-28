CFLAGS += -O3 -Wall -g
LFLAGS += -lGL -lGLU -lglut
APPS = demo benchmark

ifdef DATATYPE
CFLAGS += -D DATATYPE=$(DATATYPE)
endif


.PHONY: all

all: $(APPS)

demo: demo.c solver.c io.c
	$(CC) $(CFLAGS)  -o $@ $^ $(LFLAGS)

benchmark: benchmark.c solver.c io.c
	$(CC) $(CFLAGS)  -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS)
