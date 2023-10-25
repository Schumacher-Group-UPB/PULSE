#!/bin/bash

# User-defined launch parameters

output_path="data/should_sammel_tetm/"

system_parameters=(
    "--pump 40 30 0 0 1 1 gauss" # Center Pump
    "--pump 10 2.5 0 26 1 1 gauss" # Narrow
    "--pump adaptive 2.5 0 15 1 5 gauss" # Potential Well
    "--pump adaptive 2.5 0 7.5 1 5 gauss" # Potential Well
    "--pump adaptive 2.5 0 0 1 5 gauss" # Potential Well
    "--pump adaptive 2.5 0 -7.5 1 5 gauss" # Potential Well
    "--pump adaptive 2.5 0 -15 1 5 gauss" # Potential Well
    "--outEvery 200"
    "--tmax 5000" 
    "--N 500"
    #"--tstep 0.01"
    "--gammaC 0.05"
    "--gammaR 0.07"
    "--gc 1e-6"
    "--gr 12e-6"
    "--meff 0.00056856"
    "--R 0.01"
    "--xmax 150"
    #"--tol 10"
    #-nosfml
    #"--load data2" 
    #"-rk45"
)

# Path to the program to launch
launch_program="./main_wr_tetm_fp32.exe"

# Construct the command to launch the program
command=("$launch_program" "${system_parameters[@]}" "--path" "$output_path")

# Construct Folder
mkdir -p $output_path

# Use eval to properly split the arguments and redirect output to both console and file
eval "${command[@]}" | tee -a "$output_path/output.log"

# Check if "Psi_Minus.txt" exists in the output_path directory.
# If not, then plot the scalar model.
if [ ! -f "${output_path}Psi_Minus.txt" ]; then
    echo "Plotting Scalar Model"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}history_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; set output '${output_path}history.png'; set multiplot layout 1,1; plot '${output_path}history_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}psi.png'; set multiplot layout 1,1; plot '${output_path}psi_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}n_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}n.png'; set multiplot layout 1,1; plot '${output_path}n_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}angle.png'; set multiplot layout 1,1; plot '${output_path}psi_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
    gnuplot "-e" "set term png size 1000,500; set output '${output_path}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); plot '${output_path}max.txt' u 1:2 w l lw 3 axes x2y1 t columnhead(2);"
    exit 0
fi

# Launch gnuplot scripts
gnuplot "-e" "set term png size 1000,500; stats '${output_path}history_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_y:STATS_max_y]; set output '${output_path}history.png'; set multiplot layout 1,2; plot '${output_path}history_plus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image; plot '${output_path}history_minus.txt' u 1:2:(sqrt(\$3*\$3+\$4*\$4)) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}psi.png'; set multiplot layout 1,2; plot '${output_path}psi_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}psi_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}n_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}n.png'; set multiplot layout 1,2; plot '${output_path}n_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}n_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}angle.png'; set multiplot layout 1,2; plot '${output_path}psi_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image; plot '${output_path}psi_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
gnuplot "-e" "set term png size 1000,500; set output '${output_path}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); plot for [i=2:3] '${output_path}max.txt' u 1:i w l lw 3 axes x2y1 t columnhead(i);"