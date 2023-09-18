#!/bin/bash

# User-defined launch parameters

output_path="data/new_test_fp32_2/"

system_parameters=(
    "--pump 40 10 0 0 0"
    "--pulse 10 1e-2 1 2 1 1 10 0 0"
    "--pulse 100 1e-2 1 2 -1 1 10 0 0"
    "--outEvery 200"
    "--tmax 200" 
    "--N 800"
    #"--tol 10"
    #-nosfml
    #"--t0 100"
    #"--tstep 0.01"
    #"--load data2" 
    #"-rk45"
)

# Path to the program to launch
launch_program="./main.exe"

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
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}history_plus.txt' nooutput; set yrange[0:STATS_records]; set xrange[0:]; set output '${output_path}history.png'; set multiplot layout 1,1; plot '${output_path}history_plus.txt' matrix w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}psi.png'; set multiplot layout 1,1; plot '${output_path}psi_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}n_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}n.png'; set multiplot layout 1,1; plot '${output_path}n_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
    gnuplot "-e" "set term png size 1000,500; stats '${output_path}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}angle.png'; set multiplot layout 1,1; plot '${output_path}psi_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
    gnuplot "-e" "set term png size 1000,500; set output '${output_path}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); plot for [i=1:1] '${output_path}max.txt' u i w l lw 3 axes x2y1 t columnhead(i);"
    exit 0
fi

# Launch gnuplot scripts
gnuplot "-e" "set term png size 1000,500; stats '${output_path}history_plus.txt' nooutput; set yrange[0:STATS_records]; set xrange[0:]; set output '${output_path}history.png'; set multiplot layout 1,2; plot '${output_path}history_plus.txt' matrix w image; plot '${output_path}history_minus.txt' matrix w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}psi.png'; set multiplot layout 1,2; plot '${output_path}psi_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}psi_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}n_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}n.png'; set multiplot layout 1,2; plot '${output_path}n_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path}n_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot "-e" "set term png size 1000,500; stats '${output_path}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path}angle.png'; set multiplot layout 1,2; plot '${output_path}psi_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image; plot '${output_path}psi_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
gnuplot "-e" "set term png size 1000,500; set output '${output_path}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path}times.txt' u 1:i w l lw 3 t columnhead(i); plot for [i=1:2] '${output_path}max.txt' u i w l lw 3 axes x2y1 t columnhead(i);"