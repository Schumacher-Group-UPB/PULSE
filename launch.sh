#!/bin/bash

# User-defined launch parameters

output_path="data/"

system_parameters=(
    "--pump 40 10 0 0 0"
    "--outEvery 50"
    "--tmax 200"
    "-nosfml"
    "--N 400"
)

# Path to the program to launch
launch_program="./main.exe"

# Construct the command to launch the program
command=("$launch_program" "${system_parameters[@]}" "--path" "$output_path")

# Use eval to properly split the arguments
eval "${command[@]}"

# Launch other programs
gnuplot -e "set term png; set size ratio -1; set size 1000,500; set output '"$output_path"history.png'; set multiplot layout 1,2; plot '"$output_path"history_plus.txt' matrix w image; plot '"$output_path"history_minus.txt' matrix w image;"
gnuplot -e "set term png; set size ratio -1; set size 1000,500; set output '"$output_path"psi.png'; set multiplot layout 1,2; plot '"$output_path"psi_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '"$output_path"psi_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot -e "set term png; set size ratio -1; set size 1000,500; set output '"$output_path"n.png'; set multiplot layout 1,2; plot '"$output_path"n_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '"$output_path"n_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
gnuplot -e "set term png; set size ratio -1; set size 1000,500; set output '"$output_path"angle.png'; set multiplot layout 1,2; plot '"$output_path"psi_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image; plot '"$output_path"psi_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"