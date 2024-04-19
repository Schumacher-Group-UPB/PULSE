![resources/banner.png](resources/banner.png)

P.U.L.S.E. is a CUDA-accelerated Solver for the nonlinear two-dimensional Schrödinger Equation. Primarily developped to simulate Polariton Condensates, PULSE is able to do much more than that!
We use cache-friendly grid division to achieve the maximum possible performance while avoiding code pollution through agressive optimizations.
P.U.L.S.E. can also solve ...

# Requirements
- [MSVC](https://visualstudio.microsoft.com/de/downloads/) [Windows]
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- Optional: [SFML](https://www.sfml-dev.org/download.php) v 2.6.x
- Optional: [Gnuplot](http://www.gnuplot.info/) for fast plotting

If you are on Windows it is required to install some kind of UNIX based software distribution like [msys2](https://www.msys2.org/) or any wsl UNIX distribution for the makefile to work.
You also need to add the VS cl.exe as well as the CUDA nvcc.exe to your path.

# Build
Build with SFML rendering
- Clone the repositry using `git clone --recursive https://github.com/davidbauch/PC3`. This will also donwload the SFML repository.
- Build SFML using CMake and/or MSVC
- Alternatively, download SFML 2.6.1 or higher for MSVC if you are on Windows or for gcc if you are on linux.
- Compile P.U.L.S.E. using `make SFML=TRUE/FALSE [TETM=TRUE/FALSE FP32=TRUE/FALSE]`. Note, that arguments in `[]` are optional and default to `FALSE` if omitted. Pass *either* `TRUE` *or* `FALSE` to the arguments.

When using SFML rendering, you need to either install all SFML libraries correctly, or copy the .dll files that come either with building SFML yourself or with the download of precompiled versions to the main folder of your PULSE executable. If you do not do this and still compile with SFML support, PULSE will crash on launch. For the compilation, you also *need* to provide the path to your SFML installation if it's not already in your systems path. You can do this by setting the `SFML_PATH=...` variable when compiling, similar to passing `SFML=TRUE`. The SFML path needs to contain the SFML `include/...` as well as the `lib/...` folder. These are NOT contained directly in the recursively cloned SFML repository, but rather get created when building SFML yourself. They are also contained in any precompiled version of SFML.

Build without rendering
- Clone the repositry using `git clone https://github.com/davidbauch/PC3`
- Compile P.U.L.S.E. using `make [TETM=TRUE/FALSE FP32=TRUE/FALSE]`

## FP32 - Single Precision
By default, the program is compiled using double precision 64b floats.
For some cases, FP32 may be sufficient for convergent simulations.
To manually change the precision to 32b floats, use the 

`-DUSEFP32`

compiler flag. Alternatively, use

`FP32=TRUE`

when using the makefile.

# Getting Started
P.U.L.S.E. uses single and double hyphen commandline arguments with multiple parameters per argument. Use

`./main.exe[.o] --help` 

to get an overview of the available arguments. Also, look into [the shell launchscript](/launch.sh) for a detailed commented parameterset.

## Examples
We provide multiple examples for using PULSE in scientific work. See the [examples folder](/examples/) for an overview.

# Current Stats
P.U.L.S.E. is currently benchmarked against common Matlab Solvers for the nonlinear Schrödinger Equation as well as against itself as a CPU version.

We reproduce recent research results using P.U.L.S.E. and compare the runtimes. 

All benchmarks are evaluated with minimal output overhead. Hence, we set 

`--outEvery = 1e10` 

and

 `--output max`
 
  to only output the final maximum value.

## Example 1: 
- Gif of result
- Runtime Matlab, PULSE, PULSE_CPU

...

Settings: 800 Grid, RK4, 3070Ti
|  | FP32  | FP64 |
| - | - | - |
| Scalar | `~135ms/ps`  | `~465ms/ps`  |
| TE/TM | tbd.  | `~920ms/ps`  |

# TODO
- Better Benchmarking
- Display Examples with Videos / Gifs

