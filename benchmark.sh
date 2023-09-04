#!/bin/bash


output_path="benchmark/"

system_parameters=(
    "--pump 40 10 0 0 0"
    "--pulse 10 1e-2 1 2 1 1 10 0 0"
    "--pulse 40 1e-2 1 2 -1 1 10 0 0"
    "--outEvery 200"
    "--tmax 50"
    -nosfml
)

plot=false
makef=false

# Path to the program to launch
launch_program="./main_[FP].exe"

if [ "$makef" = true ]; then
    make clean
    make SFML=FALSE TARGET=${launch_program/\[FP\]/fp32} -j10 FP32=TRUE
    make clean
    make SFML=FALSE TARGET=${launch_program/\[FP\]/fp64} -j10
fi

# Construct Folder
for fp in "fp32" "fp64"; do
    output_path_fp="$output_path$fp/"
    launch_program_fp="${launch_program/\[FP\]/$fp}"
    for ((N=200; N<=1600; N+=200)); do
        output_path_N="$output_path_fp$N/"
        mkdir -p "$output_path_N"
        command=("$launch_program_fp" "${system_parameters[@]}" "--path" "$output_path_N" "--N" "$N")
        eval "${command[@]}" | tee -a "$output_path_N/output.log"

        # Plot if plot=true
        if [ "$plot" = true ]; then
            # Check if "Psi_Minus.txt" exists in the output_path directory.
            # If not, then plot the scalar model.
            if [ ! -f "${output_path_N}Psi_Minus.txt" ]; then
                echo "Plotting Scalar Model"
                gnuplot "-e" "set term png size 1000,500; stats '${output_path_N}history_plus.txt' nooutput; set yrange[0:STATS_records]; set xrange[0:]; set output '${output_path_N}history.png'; set multiplot layout 1,1; plot '${output_path_N}history_plus.txt' matrix w image;"
                gnuplot "-e" "set term png size 1000,500; stats '${output_path_N}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path_N}psi.png'; set multiplot layout 1,1; plot '${output_path_N}psi_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
                gnuplot "-e" "set term png size 1000,500; stats '${output_path_N}n_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path_N}n.png'; set multiplot layout 1,1; plot '${output_path_N}n_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
                gnuplot "-e" "set term png size 1000,500; stats '${output_path_N}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path_N}angle.png'; set multiplot layout 1,1; plot '${output_path_N}psi_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
                gnuplot "-e" "set term png size 1000,500; set output '${output_path_N}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path_N}times.txt' u 1:i w l lw 3 t columnhead(i); plot for [i=1:1] '${output_path_N}max.txt' u i w l lw 3 axes x2y1 t columnhead(i);"
                exit 0
            fi

            # Launch gnuplot scripts
            gnuplot "-e" "set term png size 1000,500; stats '${output_path_N}history_plus.txt' nooutput; set yrange[0:STATS_records]; set xrange[0:]; set output '${output_path_N}history.png'; set multiplot layout 1,2; plot '${output_path_N}history_plus.txt' matrix w image; plot '${output_path_N}history_minus.txt' matrix w image;"
            gnuplot "-e" "set term png size 1000,500; stats '${output_path_N}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path_N}psi.png'; set multiplot layout 1,2; plot '${output_path_N}psi_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path_N}psi_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
            gnuplot "-e" "set term png size 1000,500; stats '${output_path_N}n_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path_N}n.png'; set multiplot layout 1,2; plot '${output_path_N}n_plus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image; plot '${output_path_N}n_minus.txt' u 1:2:(\$3*\$3 + \$4*\$4) w image;"
            gnuplot "-e" "set term png size 1000,500; stats '${output_path_N}psi_plus.txt' nooutput; set xrange[STATS_min_x:STATS_max_x]; set yrange[STATS_min_x:STATS_max_x]; set output '${output_path_N}angle.png'; set multiplot layout 1,2; plot '${output_path_N}psi_plus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image; plot '${output_path_N}psi_minus.txt' u 1:2:(arg(\$3*\$3+{0,1}*\$4*\$4)) w image;"
            gnuplot "-e" "set term png size 1000,500; set output '${output_path_N}lines.png'; set multiplot layout 1,2; plot for [i=2:3] '${output_path_N}times.txt' u 1:i w l lw 3 t columnhead(i); plot for [i=1:2] '${output_path_N}max.txt' u i w l lw 3 axes x2y1 t columnhead(i);"
        fi
    done
done

# Define Python script as a multiline string
python_script='
import os
import matplotlib.pyplot as plt
import numpy as np

filepath = "'$output_path'"
print(filepath)

# Get Files
parents = ("fp32", "fp64")
folders = ("200","400","600","800","1000","1200","1400","1600")

N = len(parents)

data = {}
for parent in parents:
    data[parent] = [[],[],[],[]] # X, Total Runtime, ms/ps, s/it
    for folder in folders:
        fp = os.path.join(filepath,parent,folder,"output.log")
        tr,msps,sit = 0,0,0
        try:
            with open(fp, "r") as current:
                lines = [line for line in current.readlines() if "Total Runtime:" in line][0]
                information = lines.replace("Total Runtime: ","").replace("ms/ps -->","").replace("s/it","").replace("s","").split(" ")
                # Remove empty strings
                information = [x for x in information if len(x) > 0]
                print(information)
                tr,msps,sit = information
        except Exception as e:
            print(f"Error reading {fp}, error -> {e}")
        data[parent][0].append(int(folder))
        data[parent][1].append(float(tr))
        data[parent][2].append(float(msps))
        data[parent][3].append(float(sit))

fig,axes = plt.subplots(3,N)

axes = list(zip(*axes)) # Transpose axes
for ax,parent in zip(axes, parents):
    ax[0].set_title(parent)
    ax[0].plot(data[parent][0], data[parent][1], "o-")
    ax[0].text(0.5,0.95,f"Runtime",ha="center",va="top",transform=ax[0].transAxes)
    ax[1].plot(data[parent][0], data[parent][2], "o-")
    ax[1].text(0.5,0.95,"Time/ps",ha="center",va="top",transform=ax[1].transAxes)
    ax[2].plot(data[parent][0], data[parent][3], "o-")
    ax[2].text(0.5,0.95,"Time/it",ha="center",va="top",transform=ax[2].transAxes)
    [ax.set_xticklabels([]) for ax in ax[:-1]]
    [ax.set_yticks( np.linspace( np.min(v), np.max(v), 3 ) ) for ax,v in zip(ax, data[parent][1:])]
    [ax.yaxis.set_major_formatter(format) for ax,format in zip(ax, ["{x:.0f}s",lambda x,pos: f"{x*1e-3:.0f}s",lambda x,pos: f"{x*1e3:.0f}ms"])]


[ax.tick_params(axis="y", labelright=True, labelleft=False) for ax in axes[1]]

plt.tight_layout()

plt.savefig(os.path.join(filepath,"benchmark.pdf"))
'

# Call Python script
python -c "$python_script"
