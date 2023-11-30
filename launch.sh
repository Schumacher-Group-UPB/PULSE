#!/bin/bash

# User-defined launch parameters

output_path="data/test/"

system_parameters=(
    #"--loadFrom data/load_jan"
    #"--input pump,potential,initial,fft"
    #"--output none"
    #"--pump 55 add 35 0 0 plus 1 none gauss"
    #"--pulse 0.1 add 16 0 0 plus 1 6 ring+noDivide 400 0.1 25"
    #"--potential -0.0001 add 8 25 0 plus 8 none gauss"
    #"--potential -0.0001 add 8 -25 0 plus 8 none gauss"
    #"--potential -0.0001 add 8 0 25 plus 8 none gauss"
    #"--potential -0.0001 add 8 0 -25 plus 8 none gauss"
    #"--fftMask 1 add 0.2 0 0 plus 16 none gauss+local+noDivide" # FFT Mask
    "--initRandom 0.5 53242423412"
    #"--initialState 10 add 15 0 0 plus 1 none gauss"
    "--pump 3 add 500 0 0 plus 10 none gauss+noDivide"
    "--potential -0.0001 add 80 0 0 both 18 none gauss+noDivide"
    "--potential -0.00001 add 70 0 0 both 18 none gauss+noDivide"
    "--potential -0.00002 add 60 0 0 both 18 none gauss+noDivide"
    "--potential -0.00003 add 50 0 0 both 18 none gauss+noDivide"
    "--potential -0.00004 add 40 0 0 both 18 none gauss+noDivide"
    "--potential -0.00005 add 30 0 0 both 18 none gauss+noDivide"
    "--potential -0.00006 add 20 0 0 both 18 none gauss+noDivide"
    "--potential -0.00007 add 10 0 0 both 18 none gauss+noDivide"
    #"--pulse 0.01 add 40 0 0 plus 1 6 ring+noDivide 10000 0.1 25"
    #"--initialState 10 add 2.5 0 19 plus 1 0 gauss"
    "--outEvery 40"
    "--tmax 50000" 
    "--N 500"
    "--gammaC 0.05"
    "--gammaR 0.15"
    "--gc 6e-6"
    "--gr 10e-6"
    "--meff 5.864e-4"
    "--R 0.01"
    "--xmax 200"
    "--g_pm  -0.002"
    "--deltaLT 0.01"
    #"--tstep 0.001"
    "-periodic"
    #"-tetm"
    #"--output none"
    #"--tstep 0.02"
    #-nosfml
    #"-rk45"
    "--tol 1E-2"
    #"--tstep 0.00001"
    "--fftEvery 100" # FFT Every 1 ps
    "--threads 1"
)

# Path to the program to launch
#launch_program="./main_2.0_wr_fp32_rigged.exe"
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
    gnuplot "-e" "set size square; set term png size 1000,500; set output '${output_path}fft_plus.png'; set multiplot layout 1,2; stats '${output_path}fft_mask_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; plot '${output_path}fft_mask_plus.txt' u 1:2:3 w image; stats '${output_path}fft_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; plot '${output_path}fft_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image;"
    gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}history_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; set output '${output_path}history.png'; set multiplot layout 1,1; plot '${output_path}history_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image;"
    gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}initial_condition_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}initial_condition.png'; set multiplot layout 1,1; plot '${output_path}initial_condition_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}pump_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}pump.png'; set multiplot layout 1,2; plot '${output_path}pump_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}pump_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
    gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}wavefunction_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}wavefunction.png'; set multiplot layout 1,1; plot '${output_path}wavefunction_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}potential_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}potential.png'; set multiplot layout 1,1; plot '${output_path}potential_plus.txt' u 1:2:3 w image;"
    gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}reservoir_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}reservoir.png'; set multiplot layout 1,1; plot '${output_path}reservoir_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}wavefunction_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}angle.png'; set multiplot layout 1,1; plot '${output_path}wavefunction_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
    gnuplot "-e" "set size square; set term png size 1000,500; set output '${output_path}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); plot '${output_path}max.txt' u 1:2 w l lw 3 t columnhead(2);"
    exit 0
fi

# Launch gnuplot scripts
echo "Plotting TE/TM Model"
gnuplot "-e" "set size square; set term png size 1000,500; set output '${output_path}fft_plus.png'; set multiplot layout 1,2; stats '${output_path}fft_mask_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; plot '${output_path}fft_mask_plus.txt' u 1:2:3 w image; stats '${output_path}fft_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; plot '${output_path}fft_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image;"
gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}history_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; set output '${output_path}history.png'; set multiplot layout 1,2; plot '${output_path}history_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image; plot '${output_path}history_minus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image;"
gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}initial_condition_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}initial_condition.png'; set multiplot layout 1,2; plot '${output_path}initial_condition_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}initial_condition_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}wavefunction_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}wavefunction.png'; set multiplot layout 1,2; plot '${output_path}wavefunction_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}wavefunction_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}reservoir_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}reservoir.png'; set multiplot layout 1,2; plot '${output_path}reservoir_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}reservoir_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set size square; set term png size 1000,500; stats '${output_path}wavefunction_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}angle.png'; set multiplot layout 1,2; plot '${output_path}wavefunction_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image; plot '${output_path}wavefunction_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
gnuplot "-e" "set size square; set term png size 1000,500; set output '${output_path}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); plot for [i=2:3] '${output_path}max.txt' u 1:i w l lw 3 t columnhead(i);"