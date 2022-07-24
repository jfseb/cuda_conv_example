# cuda_conv_example
UTF-8 to UTC32 conversion with cuda


# this is a small example project for first steps with CUDA.

conversion of a utf-8 ecoded arrow style array
(data memory chunk + offsets array) into UCS32 is compared.


Comparison w.r.t. error handling follows the U_NEXT macros of ICU semantics.

for ~13MB of data,
Comparing a single thread conversion of a (randomise set of 4,3,2,1 byte sequences  + error sequences)
yields a speedup of 20-5 depending on avg length of strings ignoring the recombination work.


Using a stack buffer for output significantly enhances performance.

UTF conversion is interesting as with the naive macros warp divergence cannot be avoided.
(branch prediction significantly also shows drastic variations in on CPU performance for pure ascii/identical strings)


TODO: implemente the recombination parts in cuda.


# building

install Cuda (11.7.0)

install next to the cuda-samples directory

./cuda_samples
./cuda_conv_example

Set environment variables to point to CUDA samples headers

```
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
CUDA_PATH_V11_7=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
CUDA_SAMPLES_DIR=c:\progenv/cuda/cuda-samples
```

Open .sln in Visual Studio (2022)
Compile as Release X64 build.
dir bin\win64\Release\cuda_utf8_uchar32.exe



# some timings

Processor	Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz, 2304 Mhz, 8 Core(s), 16 Logical Processor(s)
NVIDIA Quadro T2000

AVG_LEN | NRBLOCKS  | len var   | div |  time CPU | time GPU1 | timeGPUStckBuffer | coalese  |NrStrings
4*64    | 32        |   0       |  1  |   50ms   |  10ms      |  2.9ms            |  6.2ms   |
64      | 32        |   0       |  1  |   50ms   |  4.6ms     |  2.6ms            |  6.6ms   |204800


Note: Timing exclude Device to host memcpy and allocations!

Note: The CPU is single threaded.