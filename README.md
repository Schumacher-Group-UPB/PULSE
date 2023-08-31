# PC3
Polariton Condensates simulations using CUDA

# Requirements
- [MSVC](https://visualstudio.microsoft.com/de/downloads/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- Optional: [SFML](https://www.sfml-dev.org/download.php) v 2.6.x

# Build
Build with SFML rendering
- Clone the repositry using `git clone --recursive https://github.com/davidbauch/PC3`
- Build SFML using CMake and MSVC
- Alternatively, download SFML for MSVC 2023
- Compile PC3 using `make SFML=TRUE [TETM=TRUE FP32=TRUE]`

Build without rendering
- Clone the repositry using `git clone https://github.com/davidbauch/PC3`
- Compile PC3 using `make [TETM=TRUE FP32=TRUE]`

# TE/TM Splitting
By default, the program is compiled without TE/TM splitting enabled.
To manually enable TE/TM splitting, compile the program using the 

`-DTETMSPLITTING` 

compiler flag. Alternatively, use 

`TETM=TRUE` 

when using the makefile.

# FP32 - Single Precision
By default, the program is compiled using double precision 64b floats.
For some cases, FP32 may be sufficient for convergent simulations.
To manually change the precision to 32b floats, use the 

`-DUSEFP32`

compiler flag. Alternatively, use

`FP32=TRUE`

when using the makefile.

# Current Stats
PC3 is currently benchmarked as:

Settings: 800 Grid, RK4, 3070Ti
|  | FP32  | FP64 |
| - | - | - |
| Scalar | `~135ms/ps`  | `~465ms/ps`  |
| TE/TM | tbd.  | `~920ms/ps`  |
