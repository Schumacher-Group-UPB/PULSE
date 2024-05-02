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

if ( ( index = findInArgv( "--custom_vars", argc, argv ) ) != -1 )
    p.custom_var_1 = getNextInput( argv, argc, "custom_var_1", ++index ); // <-- Note the "++index"
    p.custom_var_2 = getNextInput( argv, argc, "custom_var_2", index ); 
    p.custom_var_3 = getNextInput( argv, argc, "custom_var_3", index );
```

If done correctly, you can now add your own variables to the Kernels, parse them using the same syntax as you can for the remaining parameters and use them in the Kernels by calling `p.custom_var`!

## Adding new Envelopes to the Kernels
Adding new matrices and their respective spatial envelopes is quite a bit more challenging, but also doable. You need to add the matrix to the solver, which is quite easy, add an envelope to the system header and then read-in that envelope.

### Definition of the matrix in the [matrix container header file](include/solver/matrix_container.hpp)
Again add the definition of your matrix at the designated location in the code

Example:

```C++
PC3::CUDAMatrix<complex_number> custom_matrix;
```

You also need to add the construction of the matrices on both the CPU and GPU memory further down the file at the designated location.

Example:

```C++
custom_matrix.construct( N_x, N_y, "custom_matrix" );
```

The Kernel does not need the larger container class for the matrices. Hence, we provide a stripped-down container struct that contains all of the pointers to the GPU memory locations of the matrices. This is done in the same file in the `Pointers` struct, again at the designated location in the code.

Example:

```C++
complex_number* custom_matrix; // Inside the Pointers struct definition
...
custom_matrix.getDevicePtr() // Inside the pointers() method
```

Your matrix is now available inside the Kernels using `dev_ptrs.custom_matrix[i]`!

### Defining envelope parsing to fill the custom matrix during [the system initialization](source/system/system_initialization.cpp)
Custom envelopes require three things: Definition of the envelope variable, parsing of the envelope and calculating/transfering it onto your custom matrix.

Define your envelope in a group with the others in the [system header file](include/system/system.hpp). Search for `PC3::Envelope pulse, pump, mask, initial_state, fft_mask, potential;` inside the file, and add your envelope to the list.

Example:
    
```C++
PC3::Envelope pulse, pump, mask, initial_state, fft_mask, potential, custom_envelope;
```

Add parsing of your envelope inside the [system initialization source file](source/system/system_initialization.cpp). Search for the designated location inside the code. The other envelopes also get parsed there.

Example:

```C++
custom_envelope = PC3::Envelope::fromCommandlineArguments( argc, argv, "customEnvelope", false );
```

This envelope can then be passed via the commandline using `--customEnvelope ...`

Finally, add the evaluation of the envelope. Go to the [solver initialization source file](source/cuda_solver/solver/solver_initialization.cpp), search for the designated location in the code and copy one of the already defined initialization sections. Change the variable names and you are done!

Example:
```C++
std::cout << "Initializing Custom Envelopes..." << std::endl;
if ( system.custom_envelope.size() == 0 ) {
    std::cout << "No custom envelope provided." << std::endl;
} else {
    system.calculateEnvelope( matrix.custom_envelope_plus.getHostPtr(), system.custom_envelope, PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Plus, 0.0 /* Default if no mask is applied */ );
    if ( system.use_twin_mode ) {
        system.calculateEnvelope( matrix.custom_envelope_minus.getHostPtr(), system.custom_envelope, PC3::Envelope::AllGroups, PC3::Envelope::Polarization::Minus, 0.0 /* Default if no mask is applied */ );
    }
}
```

You are done! The host matrix will automatically get synchronized with the device matrix on the GPU whenever its device pointer is used.

As a final note, you can also add outputting of your matrix in the [solver matrix output methods](source/cuda_solver/solver/solver_output_matrices.cpp). Go the the `outputInitialMatrices()` method and insert your code at the designated location.

Example:

```C++
if ( system.doOutput( "all" ) ) // Or add your custom keys here
    system.filehandler.outputMatrixToFile( matrix.custom_matrix.getHostPtr(), system.p.N_x, system.p.N_y, header_information, "custom_matrix" );
```

And the possible loading of matrices from the loading folder by editing the code of the [solver matrix load method](source/cuda_solver/solver/solver_load_matrices.cpp). 

Example:

```C++
filehandler.loadMatrixFromFile( filehandler.loadPath + "custom_matrix.txt", matrix.custom_matrix.getHostPtr() );
```

# TODO
- Better Benchmarking
- Display Examples with Videos / Gifs