# PC3
Polariton Condensates simulations using CUDA

# Requirements
- [MSVC](https://visualstudio.microsoft.com/de/downloads/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- Optional: [SFML](https://www.sfml-dev.org/download.php)

# Build
Build with SFML rendering
- Clone the repositry using `git clone --recursive https://github.com/davidbauch/PC3`
- Build SFML using CMake and MSVC
- Alternatively, download SFML for MSVC 2023
- Compile PC3 using `make SFML=TRUE`

Build without rendering
- Clone the repositry using `git clone https://github.com/davidbauch/PC3`
- Compile PC3 using `make`
