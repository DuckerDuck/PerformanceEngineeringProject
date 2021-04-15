CXX_FLAGS +=  -O3 -Wall -g
LFLAGS += -lGL -lGLU -lglut
APPS = demo

.PHONY: all

all: $(APPS)

demo: demo.c solver.c
	$(CXX) $(CXX_FLAGS)  -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS)
