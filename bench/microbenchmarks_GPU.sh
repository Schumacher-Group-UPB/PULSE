lscpu 
nvidia-smi
nvcc --version
git clone https://github.com/te42kyfo/gpu-benches
cd gpu-benches
git checkout 1c36d509ee717dca9215c27135fa1e611bb03ed1
cd gpu-l2-cache
make
./cuda-l2-cache
cd ..
cd gpu-stream
make
./cuda-stream

