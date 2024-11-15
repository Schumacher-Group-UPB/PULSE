#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p normal
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBACTH -A pc2-mitarbeiter
#SBATCH -t 7-0

module reset
module load lang/Python/3.12.3-GCCcore-13.2.0

python3 bench.py -c $1 $1.out
