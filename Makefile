CFLAGS +=  -O3 -Wall -g
LFLAGS += -lGL -lGLU -lglut
APPS = demo

ifdef DATATYPE
CFLAGS += -D DATATYPE=$(DATATYPE)
endif


.PHONY: all

all: $(APPS)

demo: demo.c solver.c io.c
	$(CC) $(CFLAGS)  -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS)
