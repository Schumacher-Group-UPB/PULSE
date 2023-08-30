# PC3
Polariton Condensates simulations using CUDA

# Requirements
- [MSVC](https://visualstudio.microsoft.com/de/downloads/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- Optional: [SFML](https://www.sfml-dev.org/download.php) v 2.6.x

# TE/TM Splitting
By default, the program is compiled without TE/TM splitting enabled.
To manually enable TE/TM splitting, compile the program using the 

`-DTETMSPLITTING` 

compiler flag. Alternatively, use 

`TETM=TRUE` 

when using the makefile.

# Build
Build with SFML rendering
- Clone the repositry using `git clone --recursive https://github.com/davidbauch/PC3`
- Build SFML using CMake and MSVC
- Alternatively, download SFML for MSVC 2023
- Compile PC3 using `make SFML=TRUE` or `make SFML=TRUE TETM=TRUE`

Build without rendering
- Clone the repositry using `git clone https://github.com/davidbauch/PC3`
- Compile PC3 using `make` or `make TETM=TRUE`
