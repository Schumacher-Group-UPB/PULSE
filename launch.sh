#!/bin/bash

# User-defined launch parameters

output_path="data/kekwtest14/"

system_parameters=(
    "--pump 60 add 15 0 0 plus 1 gauss+ring" # Center Pump
    #"--pump -1 add+adaptive 2.5 100 15 plus 5 gauss+outerExponent" # Potential Well
    #"--pump -1 add+adaptive 2.5 100 7.5 plus 5 gauss+outerExponent" # Potential Well
    #"--pump -0.975 add+adaptive 2.5 100 0 plus 5 gauss+outerExponent" # Potential Well
    #"--pump -0.95 add+adaptive 2.5 100 -7.5 plus 5 gauss+outerExponent" # Potential Well
    #"--pump -0.93 add+adaptive 2.5 100 -15 plus 5 gauss+outerExponent" # Potential Well
    #"--pump 10 add 2.5 100 19 plus 1 gauss" # Narrow
    #"--loadFrom data/load2"
    #"--input all"
    "--mask 5 add 4.5 0 7.5 plus 5 gauss" # Soll
    #"--pulse 2 add 2.5 0 19 plus 1 gauss 150 0.1 10 2" # test pulse
    "--pulse 0.001 add 2 0 0 plus 1 gauss+ring 350 0.1 10 2" # test pulse
    #"--pulse 0.0001 add 30 0 0 plus 1 gauss+ring 1350 0.1 10 -2" # test pulse
    #"--fftMask 1 add 0.4 0 0 plus 10 gauss+local+noDivide" # FFT Mask
    "-masknorm"
    #"--initRandom 0.5 53242423412"
    "--initialState 5 add 15 0 0 plus 1 gauss+ring" # Center Pump
    #"--initialState 10 add 4.5 0 7.5 plus 5 gauss" # Soll
    "--outEvery 50"
    "--tmax 20000" 
    "--N 500"
    "--gammaC 0.05"
    "--gammaR 0.07"
    "--gc 1e-6"
    "--gr 12e-6"
    "--meff 0.00056856"
    "--R 0.01"
    "--xmax 50"
    "-periodic"
    #-nosfml
    #"-rk45"
    #"--tol 1E-2"
    "--fftEvery 1"
    #"--tstep 0.00001"
)

# Path to the program to launch
#launch_program="./main_2.0_wr_fp64_cpu.exe"
launch_program="./main_2.0_wr_fp32.exe"

# Construct the command to launch the program
command=("$launch_program" "${system_parameters[@]}" "--path" "$output_path")

# Construct Folder
mkdir -p $output_path

# Use eval to properly split the arguments and redirect output to both console and file
eval "${command[@]}" | tee -a "$output_path/output.log"

# Check if "wavefunction_minus.txt" exists in the output_path directory.
# If not, then plot the scalar model.
if [ ! -f "${output_path}wavefunction_minus.txt" ]; then
    echo "Plotting Scalar Model"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}history_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; set output '${output_path}history.png'; set multiplot layout 1,1; plot '${output_path}history_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}initial_condition_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}initial_condition.png'; set multiplot layout 1,1; plot '${output_path}initial_condition_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}wavefunction_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}wavefunction.png'; set multiplot layout 1,1; plot '${output_path}wavefunction_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}reservoir_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}reservoir.png'; set multiplot layout 1,1; plot '${output_path}reservoir_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}wavefunction_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}angle.png'; set multiplot layout 1,1; plot '${output_path}wavefunction_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
    gnuplot "-e" "set term png size 1000,500; set output '${output_path}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); plot '${output_path}max.txt' u 1:2 w l lw 3 t columnhead(2);"
    exit 0
fi

# Launch gnuplot scripts
echo "Plotting TE/TM Model"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}history_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; set output '${output_path}history.png'; set multiplot layout 1,2; plot '${output_path}history_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image; plot '${output_path}history_minus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}initial_condition_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}initial_condition.png'; set multiplot layout 1,2; plot '${output_path}initial_condition_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}initial_condition_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}wavefunction_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}wavefunction.png'; set multiplot layout 1,2; plot '${output_path}wavefunction_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}wavefunction_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}reservoir_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}reservoir.png'; set multiplot layout 1,2; plot '${output_path}reservoir_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}reservoir_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}wavefunction_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}angle.png'; set multiplot layout 1,2; plot '${output_path}wavefunction_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image; plot '${output_path}wavefunction_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
gnuplot "-e" "set term png size 1000,500; set output '${output_path}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); plot for [i=2:3] '${output_path}max.txt' u 1:i w l lw 3 t columnhead(i);"