lscpu
likwid-bench -v
if [[ "$1" == "5800X3D" ]];
then
  likwid-bench -t triad_avx_fma -W S0:64000kB:16
  likwid-bench -t triad_mem_avx_fma -W S0:6400000kB:16
  likwid-bench -t peakflops_avx_fma -W S0:64000kB:16
  likwid-bench -t peakflops_sp_avx_fma -W S0:64000kB:16
fi

if [[ "$1" == "7763" ]];
then
  likwid-bench -t triad_avx_fma -W S0:64000kB:64
  likwid-bench -t triad_mem_avx_fma -W S0:6400000kB:64
  likwid-bench -t peakflops_avx_fma -W S0:64000kB:64
  likwid-bench -t peakflops_sp_avx_fma -W S0:64000kB:64
fi
