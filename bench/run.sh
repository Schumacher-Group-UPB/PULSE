mkdir output
module reset
module load tools/likwid/5.3.0-GCC-13.2.0
module load toolchain/foss/2023b
#module load compiler/Clang/17.0.6-GCCcore-13.2.0
#module load toolchain/intel/2023b
#make clean
if [[ "1" == "1" ]]; then
  rm main.o
  #make SFML=FALSE FP32=FALSE TETM=FALSE CPU=TRUE COMPILER=g++
  rm ./obj/fp32/cpu/cuda_solver/solver/iterator/solver_iterator_rk4.obj
  rm ./obj/fp32/cpu/cuda_solver/solver/solver_iterator.obj
  rm ./obj/fp32/cpu/cuda_solver/solver/iterator/solver_iterator.obj
  make SFML=FALSE FP32=TRUE TETM=FALSE CPU=TRUE COMPILER=g++ NO_CALCULATE_K=FALSE NO_INTERMEDIATE_SUM_K=FALSE NO_FINAL_SUM_K=FALSE NO_HALO_SYNC=FALSE AVX2=TRUE BENCH=TRUE LIKWID=TRUE
  #make SFML=FALSE FP32=TRUE TETM=FALSE CPU=TRUE COMPILER=clang
  #make SFML=FALSE FP32=TRUE TETM=FALSE CPU=TRUE COMPILER=icx
fi
export OMP_STACKSIZE=10g
export OMP_NUM_THREADS=$3
#$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

bx=$4
by=$5
d="run_size_${1}_by_${2}_threads_${3}_bx_${bx}_by_${by}"
echo $d

T=200
T2=201
seed=4343

#out="none"
out="psi_plus"
for g in DATA FLOPS_SP CACHE L2 L2CACHE L3 L3CACHE MEM1 MEM2; # FLOPS_DP;
#for g in DATA; # FLOPS_SP CACHE L2 L2CACHE L3 L3CACHE MEM1 MEM2; # FLOPS_DP;
do
  likwid-perfctr -m -g $g -C 0-`echo "$OMP_NUM_THREADS-1" | bc` -o runs/${d}_$g ./main.o --L 200 200 --tmax $T --gammaC 0.01 --gc 6E-6 --gr 12E-6 --initRandom 0.01 $seed 100 --outEvery $T2 --threads $OMP_NUM_THREADS --path output --N $1 $2 --output $out --fftEvery 100000000 --boundary zero zero --subgrids $bx $by > runs/${d}_${g}_out
  #./main.o --L 200 200 --tmax 2000 --gammaC 0.01 --gc 6E-6 --gr 12E-6 --initRandom 0.01 random 100 --outEvery 50000000 --threads $OMP_NUM_THREADS --path /dev/shm --N $1 $2 --output none --fftEvery 100000000 --boundary periodic periodic --subgrids $bx $by > runs/${d}_${g}_out
  grep "Total Runtime:" runs/${d}_${g}_out
  #./main.o --L 200 200 --tmax 200 --gammaC 0.01 --gc 6E-6 --gr 12E-6 --initRandom 0.01 random --tmax 100 --outEvery 500000 --threads $OMP_NUM_THREADS --path output --N $1 $1 --output none --fftEvery 100000 --boundary periodic periodic --subgrids $bx $by > ${d}_out
  sleep 0.1
done
egrep "Memory bandwidth \(channels 0-3\) \[MBytes/s\] STAT|Memory bandwidth \(channels 4-7\) \[MBytes/s\] STAT|L3 miss ratio STAT|L2 miss ratio STAT|L2 bandwidth \[MBytes/s\] STAT|Prefetch bandwidth \[MBytes/s\] STAT|L3 bandwidth \[MBytes/s\] STAT|Load to store ratio STAT|SP \[MFLOP/s\] STAT|Clock \[MHz\] STAT" runs/${d}_* | column -t -s ","

