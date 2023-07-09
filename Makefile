# Compiler
NVCC = nvcc

# Folders
SRCDIR = source
INCDIR = include
OBJDIR = obj

# Compiler flags
CPPFLAGS = -std=c++20 -Xcompiler -openmp 
SFMLLIBS = -I'external/SFML-2.6.0/include' -L'external/SFML-2.6.0/lib/'

SFML ?= -DSFML_RENDER

# Object files
CPP_SRCS = $(wildcard $(SRCDIR)/*.cpp)
CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
CPP_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SRCS))

ifeq ($(SFML),-DSFML_RENDER)
	SFML_FLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lsfml-main $(SFMLLIBS) $(SFML)
endif

# Targets
all: clean $(OBJDIR) $(CPP_OBJS) $(CU_OBJS)

	$(NVCC) -o main.exe $(CPP_OBJS) $(CU_OBJS) -lcufft -I$(INCDIR) -Xcompiler -fopenmp $(SFML_FLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(NVCC) $(CPPFLAGS) -c $< -o $@ -I$(INCDIR) $(SFML_FLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I$(INCDIR)

$(OBJDIR):
	mkdir $(OBJDIR)

clean:
	rm -f $(OBJDIR)/*.o main.exe
	rmdir $(OBJDIR)