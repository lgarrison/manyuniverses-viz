module load python/3.9
module load fftw/3
module load git
module load disBatch/2
module load ffmpeg

#module load intel-oneapi-compilers/2022.0
#module load intel-oneapi-mkl/2022.0
#module load intel-oneapi-tbb/2021.5

module load gcc/11

. $(dirname "$(readlink -f "$BASH_SOURCE")")/venv/bin/activate
