![resources/banner.png](resources/banner.png)

PULSE is a CUDA-accelerated Solver for the nonlinear two-dimensional Schrödinger Equation. Primarily developped to simulate Polariton Condensates, PULSE is able to do much more than that!

We provide multiple examples using PULSE in scientific work. See the [examples folder](/examples/) for an overview. We provide Jupyter Notebooks as well as MatLab files to launch PULSE into different configurations. Simple use one of the precompiled binaries from the [current release](https://github.com/AG-Schumacher-UPB/PULSE/releases) and drop it into the same folder as the example you want to run. Make sure to edit the respective example file to match the executable.

# Requirements
- [MSVC](https://visualstudio.microsoft.com/de/downloads/) [Windows] or [GCC](https://gcc.gnu.org/) [Linux]
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- Optional: [SFML](https://www.sfml-dev.org/download.php) v 2.6.x
- Optional: [FFTW](https://www.fftw.org/) for the CPU version
- Optional: [Gnuplot](http://www.gnuplot.info/) for fast plotting

If you are on Windows it is required to install some kind of UNIX based software distribution like [msys2](https://www.msys2.org/) or any wsl UNIX distribution for the makefile to work.
You also need to add the VS cl.exe as well as the CUDA nvcc.exe to your path if you want to compile PULSE yourself.
Also, make sure to check the C++ Desktop Development section in the VS installer! Then, add [cl.exe](https://stackoverflow.com/questions/7865432/command-line-compile-using-cl-exe) to your [path](https://stackoverflow.com/questions/9546324/adding-a-directory-to-the-path-environment-variable-in-windows)

# Getting Started with PULSE

To successfully execute a precompiled version of PULSE for the first time, follow the steps outlined below:

#### 1. Verify Your Hardware

PULSE is designed to run exclusively on Nvidia GPUs. Verify your hardware to ensure compatibility:

- **Windows**: Open the Start menu, type "Device Manager," and press Enter to launch the Control Panel. Expand the "Display adapters" section to see your GPU listed. Right-click on the listed GPU and select "Properties" to view the manufacturer details if necessary.
- **Linux**: Use the command `lspci` to identify the GPU.

#### Steps 2 - 4 will give you detailed instructions on how to install the mandatory requirements. If you have already installed them or do not need help, please continue with step 5.

#### 2. Install Visual Studio Microsoft (Windows) or the GNU Compiler GCC (linux)

Select the right software for your operating system. Download and install it. If you are installing Microsoft Visual Studios, make sure you check "C++ Desktop Development section" during the installation process (see Screenshot). **Do not make any changes to the installation path for Visual Studio!**
![image](https://github.com/AG-Schumacher-UPB/PULSE/assets/139117697/9f6fed2a-ce10-49d9-8a23-3bf5c37b91b0)

#### 3. Install CUDA and MSYS2

Download the latest CUDA-Version [here](https://developer.nvidia.com/cuda-downloads) and follow the instructions given. **Do not make any changes to the installation path for CUDA!**
Download MSYS2 [here](https://www.msys2.org/) and install it.

#### 4. Add the new executables to your path

Open your Enviroment Variables
-**Windows**: Right-click on Start then click on "System" in the context menu. Click "Advanced system settings" and go to "Advanced" tab. Now click Enviroment Variables. Here, double-click on Path in the lower section. Click on new to add to your path.

Now you need to find the path to your cl.exe of Visual Studio and nvcc.exe for CUDA, if you have not changed the preset path during installation you should find your executable at the same location as marked orange in the screenshot.
![image](https://github.com/AG-Schumacher-UPB/PULSE/assets/139117697/127b096e-1f0d-4bda-ac54-a41305891785)

Note that in your case the version-number in the path (\14.37.32822\ for VS and \v12.3\ for CUDA) can be different.

#### Great! That was the hardest part. Now you can continue with executing PULSE for the very first time.

#### 5. Download and Prepare PULSE

Download the precompiled PULSE executable from [here](https://github.com/AG-Schumacher-UPB/PULSE/releases). This version supports single-precision float operations on your GPU and includes the SFML multimedia library for visualization.

1. Place the downloaded executable into an empty directory on your system.
2. Open a console window and navigate to your newly created PULSE directory.

#### 6. Execute PULSE

You are now ready to run PULSE for the first time. Copy and execute the following command in your console:

```sh
./pulse.exe[.o] --N 400 400 --L 40 40 --boundary zero zero --tmax 1000 --initialState 0.1 add 70 70 0 0 plus 1 0 gauss+noDivide --gammaC 0.15 --pump 100 add 4.5 4.5 0 0 both 1 none gauss+noDivide+ring --outEvery 5 --path output\
```

This command will:

- Execute PULSE on a 400x400 grid with zero boundary conditions for a real-space grid of 40x40 micrometers.
- Evolve the system for 1 ns (`--tmax`).
- Use PULSE-predefined pump and initial conditions, defining the initial state as a vortex with topological charge +1 (`--initialState`).
- Set the polariton loss rate of the condensate to 0.15 (`--gammaC`).
- Create a ring-shaped pump (`--pump`).
- Set the data output rate to every 5 picoseconds (`--outEvery`) and specify the output directory (`--path output`).

For further details on the command syntax, use `./pulse.exe[.o] --help`.

#### 7. Review Results

Upon successful execution, the time-evolution will be displayed. After the program completes, it will print a summary of the process. The output directory will contain the desired results.

Congratulations on performing your first GPU-accelerated calculation using PULSE. For a comprehensive introduction to all other features of PULSE, please refer to the extended documentation.

If you want to compile your own (modified) version of PULSE please read on.

# Build PULSE yourself

If you desire custom modifications to the code, or none of the precompiled versions work for you, you may as well build PULSE yourself. We use a simple Makefile to create binaries for either Windows or Linux.

### Build with SFML rendering
1 -  Clone the repository using
```    
    git clone --recursive https://github.com/AG-Schumacher-UPB/PULSE
``` 
This will also download the SFML repository. I suggest also downloading a precompiled version of the library using the link at the top.

2 - Build SFML using CMake and/or MSVC

Alternatively, download SFML 2.6.1 or higher for MSVC if you are on Windows or for gcc if you are on linux.

3 - Compile P.U.L.S.E. using 

```
make SFML=TRUE/FALSE [SFML_PATH=external/SFML/ FP32=TRUE/FALSE ARCH=NONE/ALL/XY]
```
Note, that arguments in `[]` are optional and default to `FALSE` (or `NONE`) if omitted. Pass only one parameter to the arguments, for example *either* `TRUE` *or* `FALSE`.

When using SFML rendering, you need to either install all SFML libraries correctly, or copy the .dll files that come either with building SFML yourself or with the download of precompiled versions to the main folder of your PULSE executable. If you do not do this and still compile with SFML support, PULSE will crash on launch. For the compilation, you also *need* to provide the path to your SFML installation if it's not already in your systems path. You can do this by setting the `SFML_PATH=...` variable when compiling, similar to passing `SFML=TRUE`. The SFML path needs to contain the SFML `include/...` as well as the `lib/...` folder. These are NOT contained directly in the recursively cloned SFML repository, but rather get created when building SFML yourself. They are also contained in any precompiled version of SFML, so I suggest to simply download a precompiled version.

### Build without rendering
1 - Clone the repositry using 
```bash
git clone https://github.com/davidbauch/PC3
```

2 - Compile P.U.L.S.E. using 
```bash
make [TETM=TRUE/FALSE ARCH=NONE/ALL/XY]`
```

### Build with CPU Kernel
If you, for some reason, want to compile this program as a CPU version, you can do this by adding the `CPU=TRUE` compiler flag to `make`. This is probably only usefull if you do not have a NVIDIA GPU.
While nvcc can compile this into CPU code, it generally makes more sense to use [GCC](https://gcc.gnu.org/) or any other compiler of your choice, as those are generally faster and better for CPU code than nvcc.
You can specify the compiler using the `COMPILER=` flag to `make`.

```bash
make [SFML=TRUE/FALSE FP32=TRUE/FALSE TETM=TRUE/FALSE CPU=TRUE COMPILER=g++]
```

## FP32 - Single Precision
By default, the program is compiled using double precision 64b floats.
For some cases, FP32 may be sufficient for convergent simulations.
To manually change the precision to 32b floats, use

```
FP32=TRUE
```

when using the makefile.

# CUDA Architexture
You can also specify the architexture used when compiling PULSE. The release binaries are compiled with a variety of Compute Capabilities (CC). To ensure maximum performance, picking the CC for your specific GPU and using 

```
ARCH=xy
```

when using the Makefile, where xy is your CC, is most beneficial.

# Current Issues

- RK45 not working properly. TODO: better time step control
- SSFM not working properly.

In both cases, a fallback to RK4 is used as a temporary workaround

- Some code refactoring required to prettify things

# Trouble Shooting

Here are some common errors and how to hopefully fix them

### Errors on Compilation even though VS and CUDA are installed, CUDA and cl are in the path variable
If you get syntax or missing file errors, your Visual Studio installation may be incompatible with your current CUDA version. Try updating or downgrading either CUDA or VS, depending on what's older on your system. Older versions of VS can be downloaded [from here](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#fixed-version-bootstrappers). Don't forget to add the new VS installation to your path. You can download older version for CUDA directly from Nvidias website. 

Current working combinations: VS Community Edition or VS Build Tools 17.9.2 - CUDA 12.4
 
# Current Stats
P.U.L.S.E. is currently benchmarked against common Matlab Solvers for the nonlinear Schrödinger Equation as well as against itself as a CPU version.

We reproduce recent research results using P.U.L.S.E. and compare the runtimes. 

TODO

## Current Benchmark Times
TODO

|  | FP32  | FP64 | CPU |
| - | - | - | - |
| RTX 3070ti / AMD Ryzen 6c | 0 | 0 | 0 |
| RTX 4090 / AMD Ryzen 48c | 0 | 0 | 0 |
| A100 / AMD Milan | 0 | 0 | 0 |

old values: Scalar  `~135ms/ps`  `~465ms/ps` 
old values: TE/TM  tbd.   `~920ms/ps` 

# Custom Kernel Variables
Right now, changing Kernels is quite easy to do. Just go to [the kernel source directory](source/cuda_solver/kernel/compute/) and edit one of the Kernel files. Recompile and you are good to go!

## Adding new variables to the Kernels
Adding new user defined variables to the Kernels is also quite straight forward. All of the inputs are hardcoded into the program, but we can insert code at two places to make them available in the Kernels.
### Definition of the variable in the [system header file](include/system/system.hpp). 

Insert custom variable definitions into the `Parameters` struct at the designated location. This is marked in the source file so you can't miss it.

Examples:

```C++
real_number custom_var; // No default value at definition
  
real_number custom_var = 0.5; // Default value at definition
```

### Read-In of custom variables. 

If setting the value directly at the definition is not an option, you can also add parsing of your variable. Go to [the system's initialization source file](source/system/system_initialization.cpp) and edit the source file according to your needs at the designated location. Examples on how to parse two or more parameters from a single input can also be found there. 

Examples:

```C++
if ( ( index = findInArgv( "--custom_var", argc, argv ) ) != -1 )
    p.custom_var = getNextInput( argv, argc, "custom_var", ++index );

if ( ( index = findInArgv( "--custom_vars", argc, argv ) ) != -1 ) {
    p.custom_var_1 = getNextInput( argv, argc, "custom_var_1", ++index ); // <-- Note the "++index"
    p.custom_var_2 = getNextInput( argv, argc, "custom_var_2", index ); 
    p.custom_var_3 = getNextInput( argv, argc, "custom_var_3", index );
}
```

If done correctly, you can now add your own variables to the Kernels, parse them using the same syntax as you can for the remaining parameters and use them in the Kernels by calling `p.custom_var`!

## Adding new Envelopes to the Kernels
Adding new matrices and their respective spatial envelopes is quite a bit more challenging, but also doable. You need to add the matrix to the solver, which is quite easy, add an envelope to the system header and then read-in that envelope.

### Definition of the matrix in the [matrix container header file](include/solver/matrix_container.hpp)
Again add the definition of your matrix at the designated location in the code. 

Example:

```C++
// Macro for definition: 
// DEFINE_MATRIX(type, name, size_scaling, condition_for_construction)

DEFINE_MATRIX(complex_number, custom_matrix_plus, 1, true) \
DEFINE_MATRIX(complex_number, custom_matrix_minus, 1, use_twin_mode) \
//                       This ^^^^^^^^^^^^^^^^^^^ is your matrix definition
```

You usually only have to change the name of your matrix. If you want to support TE/TM splitting, make sure to use two matrices, ideally with a trailing `_plus` or `_minus`. Don't forget the trailing backslashes.

Your matrix is now available inside the Kernels using `dev_ptrs.custom_matrix_plus[i]`!

### Defining envelope parsing to fill the custom matrix during [the system initialization](source/system/system_initialization.cpp)
Custom envelopes require three things: Definition of the envelope variable, parsing of the envelope and calculating/transfering it onto your custom matrix.

Define your envelope in a group with the others in the [system header file](include/system/system.hpp). Search for `PC3::Envelope pulse, pump, mask, initial_state, fft_mask, potential;` inside the file, and add your envelope to the list.

Example:
    
```C++
PC3::Envelope pulse, pump, mask, initial_state, fft_mask, potential, custom_envelope;
// This is your envelope defintion                                   ^^^^^^^^^^^^^^^
```

Add parsing of your envelope inside the [system initialization source file](source/system/system_initialization.cpp). Search for the designated location inside the code. The other envelopes also get parsed there.

Example:

```C++
custom_envelope = PC3::Envelope::fromCommandlineArguments( argc, argv, "customEnvelope", false );
// ^^^^^^^^^^^ this is your envelope name in the code and        this   ^^^^^^^^^^^^^ is the name used for parsing the command line.
```

This envelope can then be passed via the commandline using `--customEnvelope ...`

Finally, add the evaluation of the envelope. Go to the [solver initialization source file](source/cuda_solver/solver/solver_initialization.cpp), search for the designated location in the code and copy one of the already defined initialization sections. Change the variable names and you are done!

Example:
```C++
std::cout << "Initializing Custom Envelopes..." << std::endl;
if ( system.custom_envelope.size() == 0 ) {
    //      ^^^^^^^^^^^^^^^  make sure this matches your definition
    std::cout << "No custom envelope provided." << std::endl;
} else {
    system.custom_envelope( matrix.custom_matrix_plus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, 0.0 /* Default if no mask is applied */ );
    //     ^^^^^^^^^^^^^^^  and    ^^^^^^^^^^^^^^^^^^^^ this too
    if ( system.p.use_twin_mode ) {
        system.custom_envelope( matrix.custom_matrix_minus.getHostPtr(), PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, 0.0 /* Default if no mask is applied */ );
        //     ^^^^^^^^^^^^^^^  and    ^^^^^^^^^^^^^^^^^^^^^ this too
    }
}
```

You are done! The host matrix will automatically get synchronized with the device matrix on the GPU whenever its device pointer is used.

As a final note, you can also add outputting of your matrix in the [solver matrix output methods](source/cuda_solver/solver/solver_output_matrices.cpp). Go the the `outputInitialMatrices()` method and insert your code at the designated location.

Example:

```C++
if ( system.doOutput( "all" ) ) // Or add your custom keys here
    system.filehandler.outputMatrixToFile( matrix.custom_matrix.getHostPtr(), system.p.N_x, system.p.N_y, header_information, "custom_matrix" );
    //                             make sure this ^^^^^^^^^^^^^ again matches your definition.    This is the output file name ^^^^^^^^^^^^^ without the ".txt"
```

You can of course also load external .txt matrices using the regular envelope syntax.

# TODO
- Better Benchmarking
- Display Examples with Videos / Gifs
- Split-step Fourier method as an alternative to RK4
- Imaginary time algorithm
