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
You also need to add the VS cl.exe as well as the CUDA nvcc.exe to your path if you want to compile PULSE yourself.
Also, make sure to check the C++ Desktop Development section in the VS installer! Then, add [cl.exe](https://stackoverflow.com/questions/7865432/command-line-compile-using-cl-exe) to your [path](https://stackoverflow.com/questions/9546324/adding-a-directory-to-the-path-environment-variable-in-windows)

# Build
### Build with SFML rendering
1 -  Clone the repository using
```    
    git clone --recursive https://github.com/davidbauch/PC3 
``` 
This will also donwload the SFML repository.

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

# Getting Started
P.U.L.S.E. uses single and double hyphen commandline arguments with multiple parameters per argument. Use

```
./main.exe[.o] --help
```

to get an overview of the available arguments. Also, look into [the shell launchscript](/launch.sh) for a detailed commented parameterset.

# Trouble Shooting

Here are some common errors and how to hopefully fix them

### Errors on Compilation even though VS and CUDA are installed, CUDA and cl are in the path variable
If you get syntax or missing file errors, your Visual Studio installation may be incompatible with your current CUDA version. Try updating or downgrading either CUDA or VS, depending on what's older on your system. Older versions of VS can be downloaded [from here](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#fixed-version-bootstrappers). Don't forget to add the new VS installation to your path. You can download older version for CUDA directly from Nvidias website. 

Current working combinations: VS Community Edition or VS Build Tools 17.9.2 - CUDA 12.4
 
# Current Stats
P.U.L.S.E. is currently benchmarked against common Matlab Solvers for the nonlinear Schrödinger Equation as well as against itself as a CPU version.

We reproduce recent research results using P.U.L.S.E. and compare the runtimes. 

...

# Examples
We provide multiple examples for using PULSE in scientific work. See the [examples folder](/examples/) for an overview.


## Example 1: 
- Gif of result
- Runtime Matlab, PULSE, PULSE_CPU

...

Runstring:
```bash
./pulse.exe ...
```

|  | FP32  | FP64 | CPU |
| - | - | - | - |
| RTX 3070ti / AMD Ryzen 6c | 0 | 0 | 0 |
| RTX 4090 / AMD Ryzen 48c | 0 | 0 | 0 |
| A100 / AMD Milan | 0 | 0 | 0 |

alt: Scalar  `~135ms/ps`  `~465ms/ps` 

alt: TE/TM  tbd.   `~920ms/ps` 

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
Again add the definition of your matrix at the designated location in the code

Example:

```C++
DEFINE_MATRIX(complex_number, custom_matrix_plus, 1) \
DEFINE_MATRIX(complex_number, custom_matrix_minus, 1) \
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