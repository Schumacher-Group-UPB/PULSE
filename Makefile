# Compiler
COMPILER = nvcc

# Folders
SRCDIR = source
INCDIR = include
OBJDIR = obj

SFML ?= FALSE
FP32 ?= FALSE
CPU ?= FALSE

PRETTYCMD ?= FALSE
CMD_COLORS ?= FALSE
CMD_SYMBOLS ?= FALSE

# GPU Architexture flag. If false, none is used
ARCH ?= NONE

# SFML PATH
SFML_PATH = external/SFML/
# Optimization
OPTIMIZATION = -O3 -save-temps
# NUMA
NUMA = FALSE

# Compiler flags. Warning 4005 is for redefinitions of macros, which we actively use.
GCCFLAGS = -std=c++20 -fopenmp -x c++ -fverbose-asm -mtune=native -march=native -save-temps -flto -funroll-loops -finline-limit=20000 -fopt-info-vec
ifeq ($(OS),Windows_NT)
	NVCCFLAGS = -std=c++20 -Xcompiler -openmp -lcufft -lcurand -lcudart -lcudadevrt  -Xcompiler="-wd4005" -rdc=true --expt-extended-lambda --dlink-time-opt --generate-line-info
else
	NVCCFLAGS = -std=c++20 -Xcompiler -fopenmp -lcufft -lcurand -lcudart -lcudadevrt  -diag-suppress 177 -lstdc++ -rdc=true --expt-extended-lambda --dlink-time-opt 
endif
SFMLLIBS = -I$(SFML_PATH)/include/ -L$(SFML_PATH)/lib

ifneq ($(ARCH),NONE)
    ifeq ($(ARCH),ALL)
        NVCCFLAGS += -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86
    else
        NVCCFLAGS += -gencode arch=compute_$(ARCH),code=sm_$(ARCH) -gencode arch=compute_$(ARCH),code=compute_$(ARCH)
    endif
endif

OBJDIR_SUFFIX = 
ifeq ($(FP32),TRUE)
    OBJDIR_SUFFIX := $(OBJDIR_SUFFIX)/fp32
else
    OBJDIR_SUFFIX := $(OBJDIR_SUFFIX)/fp64
endif
ifeq ($(CPU),TRUE)
    OBJDIR_SUFFIX := $(OBJDIR_SUFFIX)/cpu
else
    OBJDIR_SUFFIX := $(OBJDIR_SUFFIX)/gpu
endif
OBJDIR := $(OBJDIR)/$(OBJDIR_SUFFIX)

# Object files
ifeq ($(SFML),FALSE)
CPP_SRCS := $(shell find $(SRCDIR) -not -path "*sfml*" -name "*.cpp")
else
CPP_SRCS = $(shell find $(SRCDIR) -name "*.cpp")
endif
CU_SRCS = $(shell find $(SRCDIR) -name "*.cu")

CPP_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.obj,$(CU_SRCS))


ifeq ($(SFML),TRUE)
	ADD_FLAGS = -lsfml-graphics -lsfml-window -lsfml-system $(SFMLLIBS) -DSFML_RENDER
endif
ifeq ($(FP32),TRUE)
	ADD_FLAGS += -DUSE_HALF_PRECISION
endif
ifeq ($(CPU),TRUE)
	ADD_FLAGS += -DUSE_CPU
	ADD_FLAGS += -lfftw3f -lfftw3
endif

ifeq ($(PRETTYCMD),FALSE)
	ifeq ($(CMD_COLORS),FALSE)
		ADD_FLAGS += -DPC3_NO_ANSI_COLORS
	endif
	ifeq ($(CMD_SYMBOLS),FALSE)
		ADD_FLAGS += -DPC3_NO_EXTENDED_SYMBOLS
	endif
endif

# Targets
ifndef TARGET
	ifeq ($(OS),Windows_NT)
		TARGET = main.exe
	else
		TARGET = main.o
	endif
endif


ifeq ($(COMPILER),nvcc)
	COMPILER_FLAGS = $(NVCCFLAGS) $(OPTIMIZATION)
else
	COMPILER_FLAGS = $(GCCFLAGS) $(OPTIMIZATION)
endif

ifneq ($(NUMA),FALSE)
	COMPILER_FLAGS += -DUSE_NUMA -DPULSE_NUMA_DOMAINS=$(NUMA) -lnuma
endif


all: $(OBJDIR) $(CPP_OBJS) $(CU_OBJS)
	$(COMPILER) -o $(TARGET) $(CPP_OBJS) $(CU_OBJS) $(COMPILER_FLAGS) -I$(INCDIR) $(ADD_FLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(COMPILER) $(COMPILER_FLAGS) -c $< -o $@ -I$(INCDIR) $(ADD_FLAGS)

$(OBJDIR)/%.obj: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(COMPILER) $(COMPILER_FLAGS) -c $< -o $@ -I$(INCDIR) $(ADD_FLAGS)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

clean:
	@rm -fr obj/