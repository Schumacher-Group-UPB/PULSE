![resources/banner.png](resources/banner.png)

P.U.L.S.E. is a CUDA-accelerated Solver for the nonlinear two-dimensional Schrödinger Equation. Primarily developped to simulate Polariton Condensates, PULSE is able to do much more than that!
We use cache-friendly grid division to achieve the maximum possible performance while avoiding code pollution through agressive optimizations.
P.U.L.S.E. can also solve ...

# Requirements
- [MSVC](https://visualstudio.microsoft.com/de/downloads/) [Windows]
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- Optional: [SFML](https://www.sfml-dev.org/download.php) v 2.6.x
- Optional: [Gnuplot](http://www.gnuplot.info/) for fast plotting

# Build
Build with SFML rendering
- Clone the repositry using `git clone --recursive https://github.com/davidbauch/PC3`
- Build SFML using CMake and MSVC
- Alternatively, download SFML for MSVC 2023
- Compile P.U.L.S.E. using `make SFML=TRUE [TETM=TRUE FP32=TRUE]`

Build without rendering
- Clone the repositry using `git clone https://github.com/davidbauch/PC3`
- Compile P.U.L.S.E. using `make [FP32=TRUE]`

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

