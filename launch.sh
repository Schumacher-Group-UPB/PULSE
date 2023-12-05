#!/bin/bash

## Output Path ##
output_path="data/test/"
# Path to the program to launch
launch_program="./main_2.0_wr_fp32.exe"

system_parameters=(

    ############################ Basic Configuration ###########################

    ##-------------------------- Loading Parameters --------------------------##
    # To correctly load matrices, their filenames have to equal the output filenames
    #   of the program. For example, if the program outputs "wavefunction_plus.txt", 
    #   then the input file has to be named "wavefunction_plus.txt".
    #"--loadFrom data/load_jan" # Path to Load input matrices from
    ## -- Input Keywords -- #
    # The files to actually load can then be specified using keywords.
    #   For example, if you want to load the pump, then pump_plus/minus.txt has to
    #   exist in the load path, as well as the keyword "pump" (or "mat" or "all") 
    #   has to be specified.
    #"--input pump,potential,initial,fft" # Keywords of input matrices to load

    ##-------------------------- Output Parameters ---------------------------##
    #"--output none" # Keywords of matrices to output to file
    "--outEvery 40" # Output everx x iterations (not ps)
    #"--history [x]" # Output at most x history points. The history points are cached every outEvery iterations.
    #"--historyMatrix [start] [end] [increment]" # Output the history matrix every outEvery iterations. 
    #"--outMat Scaling" # Output all matrices every outEvery iterations, but scale their dimensions by the given factor
    #-nosfml # Disable SFML output

    ##------------------------- Numeric Configuration ------------------------##
    # -- Grid Configuration -- #
    "--N 500" # Single Direction Grid Resolution
    "--xmax 200" # X-Range in mum. The grid is calculated from -xmax to xmax
    "-periodic" # Periodic Boundary Conditions
    # -- Temporal Configuration -- #
    "--tmax 5000" # Evaluated Time in ps
    #"--tstep 0.02" # Time step. If omitted, the magic time step is used
    #"-rk45" # Use the RK45 solver
    #"--tol 1E-2" # Tolerance for the RK45 Solver
    # -- Structure -- #
    #"-tetm" # Use TE/TM Splitting
    
    ############################## Mask Inputs #################################

    ##---------------------------- Mask Syntax -------------------------------##
    #"--(mask) [Amplitude] [behaviour:add,multiply,replace,adaptive,complex] [Width] [X] [Y] [polarization:plus,minus,both] [Exponent] [Charge or none] [gauss,outerExponent,ring,noDivide,local] # FFT Mask
    # Amplitude: float
    # Behaviour: add, multiply, replace, adaptive, complex
    #   add: Add the mask to the current mask
    #   multiply: Multiply the mask with the current mask
    #   replace: Replace the current mask with the new mask
    #   adaptive: Set the amplitude to the current mask value
    #   complex: Multiply the amplitude with 1i
    # Width: float
    # X: float
    # Y: float
    # Polarization: plus, minus, both
    #   plus: Only apply the mask to the plus polarization
    #   minus: Only apply the mask to the minus polarization
    #   both: Apply the mask to both polarizations
    # Exponent: float
    # Charge: int, topological charge
    #   none or 0: No topological charge
    # Type: gauss, outerExponent, ring, noDivide, local
    #   gauss: Gaussian mask: A*exp(-([(x/w)^2]^exponent)
    #   outerExponent: Gauss with exponent outside of exp: A*[exp(-(x/w)^2)]^exponent
    #   ring: Ring mask: Gauss with Amplitude normalized over x
    #   noDivide: Do not divide by the mask with Width or sqrt(2pi)

    ##-------------------------- FFT Configuration ---------------------------##
    #"--pump [Amp] [Behaviour] [Width] [X] [Y] [Pol] [Exponent] [M] [Type]"
    #--fftMask 1 add 0.25 0 0 plus 1 none gauss+noDivide+local # Example: 25% of Area is masked with a Gaussian.
    #"--fftEvery 100" # FFT Every x ps

    ##-------------------------- Pump Configuration --------------------------##
    #"--pump [Amp] [Behaviour] [Width] [X] [Y] [Pol] [Exponent] [M] [Type]"

    ##----------------------- Potential Configuration ------------------------##
    #"--potential [Amp] [Behaviour] [Width] [X] [Y] [Pol] [Exponent] [M] [Type]"

    ##------------------------- Pulse Configuration --------------------------##
    # The pulses follow the same syntax as the masks. Additionally, they also take
    # a T0, Frequency and TWidth parameter.
    #"--pulse [Amp] [Behaviour] [Width] [X] [Y] [Pol] [Exponent] [M] [Type] [T0] [Frequency] [TWidth]"
    
    ##---------------------- Initial State Configuration ---------------------##
    #"--initialState [Amp] [Behaviour] [Width] [X] [Y] [Pol] [Exponent] [M] [Type]"
    "--initRandom 0.5 random" # Randomly initialize the system from -Amp to Amp with defined seed [random or number]

    ############################ System Parameters #############################
    
    ##--------------------------- Scalar Parameters --------------------------##
    #"--gammaC 0.05"
    #"--gammaR 0.15"
    #"--gc 6e-6"
    #"--gr 10e-6"
    #"--meff 5.864e-4"
    #"--R 0.01"

    ##---------------------------- TE/TM Parameters --------------------------##
    #"--g_pm -0.002"
    #"--deltaLT 0.01"

)

################################################################################
########################### End of User Defined Inputs #########################
################################################################################

# Construct the command to launch the program
command=("$launch_program" "${system_parameters[@]}" "--path" "$output_path")

# Construct Folder
mkdir -p $output_path

# Use eval to properly split the arguments and redirect output to both console and file
eval "${command[@]}" | tee "$output_path/output.log"

# Check if "wavefunction_minus.txt" exists in the output_path directory.
# If not, then plot the scalar model.
if [ ! -f "${output_path}wavefunction_minus.txt" ]; then
    # FFT Plot
    echo "Plotting Scalar Model"
    gnuplot "-e" "set size square; 
    set view map;
    set term png size 1500,500; 
    set output '${output_path}fft.png'; 
    set multiplot layout 1,3; 
    stats '${output_path}fft_mask_plus.txt' nooutput; 
    set xrange[STATS_min_x:STATS_max_x]; 
    set yrange[STATS_min_y:STATS_max_y]; 
    splot '${output_path}fft_mask_plus.txt' u 1:2:3 w pm3d t 'Mask'; 
    stats '${output_path}fft_plus.txt' nooutput; 
    set xrange[STATS_min_x:STATS_max_x]; 
    set yrange[STATS_min_y:STATS_max_y]; 
    splot '${output_path}fft_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w pm3d t 'FFT';
    splot '${output_path}fft_plus.txt' u 1:2:(log(sqrt(\$3*\$3+\$4*\$4))) w pm3d t 'FFT Logscale';"
    
    # History Plot
    gnuplot "-e" "set size square; 
    set view map;
    set term png size 500,500; 
    stats '${output_path}history_plus.txt' nooutput; 
    set xrange[STATS_min_x:STATS_max_x]; 
    set yrange[STATS_min_y:STATS_max_y]; 
    set output '${output_path}history.png'; 
    set multiplot layout 1,1; 
    set xlabel 'Output Ite';
    set ylabel 'Cut Coordinate';
    splot '${output_path}history_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w pm3d t 'History';"
    
    # Initial Condition
    gnuplot "-e" "set size square; 
    set view map;
    set term png size 1500,500; 
    stats '${output_path}initial_condition_plus.txt' nooutput; 
    set xrange[STATS_min_x:STATS_max_x]; 
    set yrange[STATS_min_x:STATS_max_x]; 
    set output '${output_path}initial_condition.png'; 
    set multiplot layout 1,3; 
    splot '${output_path}initial_condition_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Initial Condition';
    splot '${output_path}initial_condition_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Initial Condition';
    set view 50,30;
    splot '${output_path}initial_condition_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Initial Condition';"
    
    # Pump
    gnuplot "-e" "set size square; 
    set view map;
    set term png size 1500,500; 
    stats '${output_path}pump_plus.txt' nooutput; 
    set xrange[STATS_min_x:STATS_max_x]; 
    set yrange[STATS_min_x:STATS_max_x]; 
    set output '${output_path}pump.png'; 
    set multiplot layout 1,3; 
    splot '${output_path}pump_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Pump'; 
    splot '${output_path}pump_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Pump';
    set view 50,30;
    splot '${output_path}pump_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Pump';"

    # Wavefunction
    gnuplot "-e" "set size square; 
    set view map;
    set term png size 1500,500; 
    stats '${output_path}wavefunction_plus.txt' nooutput; 
    set xrange[STATS_min_x:STATS_max_x]; 
    set yrange[STATS_min_x:STATS_max_x]; 
    set output '${output_path}wavefunction.png'; 
    set multiplot layout 1,3; 
    splot '${output_path}wavefunction_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Wavefunction';
    splot '${output_path}wavefunction_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Wavefunction';
    set view 50,30;
    splot '${output_path}wavefunction_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Wavefunction';"

    # Reservoir
    gnuplot "-e" "set size square; 
    set view map;
    set term png size 1920,1090; 
    stats '${output_path}reservoir_plus.txt' nooutput; 
    set xrange[STATS_min_x:STATS_max_x]; 
    set yrange[STATS_min_x:STATS_max_x]; 
    set output '${output_path}reservoir.png'; 
    set multiplot layout 1,3; 
    splot '${output_path}reservoir_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Reservoir';
    splot '${output_path}reservoir_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Reservoir';
    set view 50,30;
    splot '${output_path}reservoir_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Reservoir';"


    # Potential
    gnuplot "-e" "set size square; 
    set view map;
    set term png size 1500,500; 
    stats '${output_path}potential_plus.txt' nooutput; 
    set xrange[STATS_min_x:STATS_max_x]; 
    set yrange[STATS_min_x:STATS_max_x]; 
    set output '${output_path}potential.png'; 
    set multiplot layout 1,2; 
    splot '${output_path}potential_plus.txt' u 1:2:3 w pm3d t 'Potential';
    set view 50,30;
    splot '${output_path}potential_plus.txt' u 1:2:3 w pm3d t 'Potential';"

    # Lines
    gnuplot "-e" "set size square; 
    set term png size 1500,500; 
    set output '${output_path}lines.png'; 
    set multiplot layout 1,2; 
    plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); 
    plot '${output_path}max.txt' u 1:2 w l lw 3 t columnhead(2);"
    exit 0
fi

# Launch gnuplot scripts
echo "Plotting TE/TM Model"

# FFT Plot
gnuplot "-e" "set size square; 
set view map;
set term png size 1500,1000; 
set output '${output_path}fft.png'; 
set multiplot layout 2,3; 
stats '${output_path}fft_mask_plus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_y:STATS_max_y]; 
splot '${output_path}fft_mask_plus.txt' u 1:2:3 w pm3d t 'Mask+'; 
stats '${output_path}fft_plus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_y:STATS_max_y]; 
splot '${output_path}fft_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w pm3d t 'FFT+';
splot '${output_path}fft_plus.txt' u 1:2:(log(sqrt(\$3*\$3+\$4*\$4))) w pm3d t 'FFT+ Logscale';
set view map;
stats '${output_path}fft_mask_minus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_y:STATS_max_y]; 
splot '${output_path}fft_mask_minus.txt' u 1:2:3 w pm3d t 'Mask-'; 
stats '${output_path}fft_minus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_y:STATS_max_y]; 
splot '${output_path}fft_minus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w pm3d t 'FFT-';
splot '${output_path}fft_minus.txt' u 1:2:(log(sqrt(\$3*\$3+\$4*\$4))) w pm3d t 'FFT- Logscale';"

# History Plot
gnuplot "-e" "set size square; 
set view map;
set term png size 1000,500; 
set output '${output_path}history.png'; 
set multiplot layout 1,2; 
stats '${output_path}history_plus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_y:STATS_max_y]; 
set xlabel 'Output Ite';
set ylabel 'Cut Coordinate';
splot '${output_path}history_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w pm3d t 'History+';
stats '${output_path}history_minus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_y:STATS_max_y]; 
set xlabel 'Output Ite';
set ylabel 'Cut Coordinate';
splot '${output_path}history_minus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w pm3d t 'History-';"

# Initial Condition
gnuplot "-e" "set size square; 
set view map;
set term png size 1500,1000; 
set output '${output_path}initial_condition.png'; 
set multiplot layout 2,3; 
stats '${output_path}initial_condition_plus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}initial_condition_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Initial Condition+';
splot '${output_path}initial_condition_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Initial Condition+';
set view 50,30;
set view map;
splot '${output_path}initial_condition_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Initial Condition+';
stats '${output_path}initial_condition_minus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}initial_condition_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Initial Condition-';
splot '${output_path}initial_condition_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Initial Condition-';
set view 50,30;
splot '${output_path}initial_condition_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Initial Condition-';"

# Pump
gnuplot "-e" "set size square; 
set view map;
set term png size 1500,1000; 
set output '${output_path}pump.png'; 
set multiplot layout 2,3; 
stats '${output_path}pump_plus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}pump_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Pump+'; 
splot '${output_path}pump_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Pump+';
set view 50,30;
splot '${output_path}pump_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Pump+';
set view map;
stats '${output_path}pump_minus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}pump_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Pump-'; 
splot '${output_path}pump_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Pump-';
set view 50,30;
splot '${output_path}pump_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Pump-';"

# Wavefunction
gnuplot "-e" "set size square; 
set view map;
set term png size 1500,1000; 
set output '${output_path}wavefunction.png'; 
set multiplot layout 2,3; 
stats '${output_path}wavefunction_plus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}wavefunction_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Wavefunction+';
splot '${output_path}wavefunction_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Wavefunction+';
set view 50,30;
splot '${output_path}wavefunction_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Wavefunction+';
set view map;
stats '${output_path}wavefunction_minus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}wavefunction_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Wavefunction-';
splot '${output_path}wavefunction_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Wavefunction-';
set view 50,30;
splot '${output_path}wavefunction_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Wavefunction-';"

# Reservoir
gnuplot "-e" "set size square; 
set view map;
set term png size 1920,1090; 
set output '${output_path}reservoir.png'; 
set multiplot layout 1,3; 
stats '${output_path}reservoir_plus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}reservoir_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Reservoir+';
splot '${output_path}reservoir_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Reservoir+';
set view 50,30;
splot '${output_path}reservoir_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Reservoir+';
stats '${output_path}reservoir_minus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}reservoir_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Reservoir-';
splot '${output_path}reservoir_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w pm3d t 'Phase of Reservoir-';
set view 50,30;
splot '${output_path}reservoir_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w pm3d t 'Reservoir-';"


# Potential
gnuplot "-e" "set size square; 
set view map;
set term png size 1500,500; 
set output '${output_path}potential.png'; 
set multiplot layout 1,2; 
stats '${output_path}potential_plus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}potential_plus.txt' u 1:2:3 w pm3d t 'Potential+';
set view 50,30;
splot '${output_path}potential_plus.txt' u 1:2:3 w pm3d t 'Potential+';
stats '${output_path}potential_minus.txt' nooutput; 
set xrange[STATS_min_x:STATS_max_x]; 
set yrange[STATS_min_x:STATS_max_x]; 
splot '${output_path}potential_minus.txt' u 1:2:3 w pm3d t 'Potential-';
set view 50,30;
splot '${output_path}potential_minus.txt' u 1:2:3 w pm3d t 'Potential-';"

# Lines
gnuplot "-e" "set size square; 
set term png size 1500,500; 
set output '${output_path}lines.png'; 
set multiplot layout 1,2; 
plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); 
plot '${output_path}max.txt' u 1:2 w l lw 3 t columnhead(2);"
exit 0