# Compiler
NVCC = nvcc

# Folders
SRCDIR = source
INCDIR = include
OBJDIR = obj

# Compiler flags
CPPFLAGS = -std=c++20 -Xcompiler -openmp 
SFMLLIBS = -I'external/SFML/include' -L'external/SFML/build/lib/Release'

SFML ?= FALSE

# Object files
CPP_SRCS = $(wildcard $(SRCDIR)/*.cpp)
CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
CPP_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SRCS))

ifeq ($(SFML),TRUE)
	SFML_FLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lsfml-main $(SFMLLIBS) -DSFML_RENDER
endif

# Targets
ifeq ($(OS),Windows_NT)
	TARGET = main.exe
else
	TARGET = main.o
endif

all: $(OBJDIR) $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -o $(TARGET) $(CPP_OBJS) $(CU_OBJS) -lcufft -I$(INCDIR) -Xcompiler -openmp -rdc=true $(SFML_FLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(NVCC) $(CPPFLAGS) -c $< -o $@ -I$(INCDIR) $(SFML_FLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I$(INCDIR) -lcufft -rdc=true -Xcompiler -openmp $(SFML_FLAGS) -diag-suppress 177

$(OBJDIR):
	@mkdir $(OBJDIR)

clean:
	@rm -f $(OBJDIR)/*.o main.exe
	@rm -fr $(OBJDIR)