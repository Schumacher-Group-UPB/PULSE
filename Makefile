# Compiler
NVCC = nvcc

# Folders
SRCDIR = source
INCDIR = include
OBJDIR = obj

# Compiler flags
ifeq ($(OS),Windows_NT)
	NVCCFLAGS = -std=c++20 -Xcompiler -openmp -lcufft -rdc=true
else
	NVCCFLAGS = -std=c++17 -Xcompiler -fopenmp -lcufft -rdc=true -diag-suppress 177
endif
SFMLLIBS = -I'external/SFML/include' -L'external/SFML/build/lib/Release'

SFML ?= FALSE
TETM ?= FALSE
FP32 ?= FALSE

# Object files
CPP_SRCS = $(wildcard $(SRCDIR)/*.cpp)
CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
CPP_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SRCS))

ifeq ($(SFML),TRUE)
	ADD_FLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lsfml-main $(SFMLLIBS) -DSFML_RENDER
endif
ifeq ($(TETM),TRUE)
	ADD_FLAGS += -DTETMSPLITTING
endif
ifeq ($(FP32),TRUE)
	ADD_FLAGS += -DUSEFP32
endif

# Targets
ifndef TARGET
	ifeq ($(OS),Windows_NT)
		TARGET = main.exe
	else
		TARGET = main.o
	endif
endif

all: $(OBJDIR) $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -o $(TARGET) $(CPP_OBJS) $(CU_OBJS) $(NVCCFLAGS) -I$(INCDIR) $(ADD_FLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I$(INCDIR) $(ADD_FLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I$(INCDIR) $(ADD_FLAGS)

$(OBJDIR):
	@mkdir $(OBJDIR)

clean:
	@rm -f $(OBJDIR)/*.o $(TARGET)
	@rm -fr $(OBJDIR)