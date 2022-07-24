/* Copyright (c) 2022, gerd forstmann (modifications for utf-8 conversion example)
 * 
 * frame based on code (cuda-samples) provided by:
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.  (frame code)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This does a utf-8 to UCS32 conversion (with error detection)
 * on an arrow-format like buffer
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <thrust/scan.h>
#include <iomanip >

#include <helper_functions.h>
#include <helper_cuda.h>

#define UChar32 int32_t

///////////////////////////////////////////////////////////////////////////////
// Convert buffer src containing utf-8 described by srcOffsets or nrStrings+1 
// into a buffer dest described by destOffsets, replacing illegal UTF-8 sequences 
// by a unicode Replacement character
// 
// Note: This is an *unsafe* function w.r.t. buffer overruns, no checks
// on the validity of offsets is provided.
// srcOffsets[0,nrStrings+1] are assumed monotonously increasing offsets, with
// 
// dest is supposed to be at least srcOffsets[nrStrings+1] long.
// returns 0 if all data was converted, 
// return 1 if (any) illegal sequence was found
///////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Convert buffer src containing utf-8 described by srcOffsets or nrStrings+1 
/// into a buffer dest described by destOffsets, replacing illegal UTF-8 sequences 
/// by a (fixed) unicode Replacement character
/// 
/// Note: This is an *unsafe* function w.r.t. buffer overruns, no checks
/// on the validity of offsets is provided.
/// srcOffsets[0,nrStrings+1] are assumed monotonously increasing offsets, with
/// 
/// dest is supposed to be at least srcOffsets[nrStrings+1] long.
/// returns 0 if all data was converted, 
/// return 1 if (any) illegal sequence was found
/// </summary>
/// <param name="src"> input data, at least 0...srcOffsets[nrString] elements</param>
/// <param name="srcOffsets">input offsets,. nrStrings+1 values, srcOffsets[0] == 0, monotonusly increasing 
///   srcOffset[i] &lt;= srcOffsets[i+1]  </param>
/// <param name="nrStrings">Number of separate strings encoded</param>
/// <param name="dest">output data, space for srcOffsets[nrString]</param>
/// <param name="destOffsets">output offsets, space for nrStrings+1 values</param>
/// <returns> returns 0 if all data was converted, 
/// return 1 if (any) illegal sequence was found</returns>
extern "C" int convChunkUTF8_UCHAR32(uint8_t* src, int* srcOffsets, int nrStrings, UChar32* dest, int* destOffsets);

// GPU variant 
#include "conv_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

//                    blocks   
//optimum @len 64      32 :18.5            128 : 16

const int AVG_LEN = 4*64;
const int NRSTRINGS = 64 * 256 * 8 * 100 / AVG_LEN;
//const int NRSTRINGS = 64 * 256 * 8 *640 / AVG_LEN; with AVG_LEN = 3200 crash


#define ALL_STRINGS_SAME 0


#define ALLASCII 0
#if ALLASCII
const float plen = 0; // 1 = constant length, 0 = avg length
const int range[] = { 0,0,0,0 }; // promille  [err, 4-byte, 3-byte, 2-byte  [remainder to 1000 = 1byte[
#else 
const float plen = 0.5; // 1 = ragged length, 0.0 = constant length
const int range[] = { 1,100,200,300 }; // promille  [err, 4-byte, 3-byte, 2-byte  [remainder to 1000 = 1byte[
#endif 
const int perc[] = { range[0], range[0] + range[1], range[0] + range[1] + range[2], range[0] + range[1] + range[2] + range[3] };

#define NRBLOCK 32

// but strongly preferred to be a multiple of warp size
// to meet memory coalescing constraints



// length of source char type
#define src_char_t uint8_t
// length of dest char type
#define dst_char_t UChar32
// length of index type into the offset array
#define indexoff_t int
// type of the offset array
#define offsets__t int

bool isInRange(int tst, int upr) {
  return tst >= 0 && tst <= upr;
}

const bool verbose = false;


int genError(src_char_t* pos, int len, int x) {
  if (len < 4) {
    *pos++ = 0x80; // sole trail byte
    return 1;
  }
  switch (x % 7) {
  case 0:
    *pos++ = 0x80; // sole trail byte
    return 1;
  case 1:
    *pos++ = 0xF0;
    *pos++ = 0x9F;
    *pos++ = 0x98;
    return 3;
  case 2:
    *pos++ = 0xF0;
    *pos++ = 0x9F;
    return 2;
  case 3:
    *pos++ = 0xF0;
    return 1;
    // broken 3 byte
  case 4:
    *pos++ = 0xE2;
    *pos++ = 0x98;
    return 2;
  case 5:
    *pos++ = 0xE2;
    return 1;
    // broken 2 byte 
  default: 
    *pos++ = 0xc3;
    return 1;
  }
}

/// <summary>
/// add a next "random" char at x
/// </summary>
/// <param name="pos"></param>
/// <param name="len"></param>
/// <returns>nr of added bytes</returns>
int addNextRandom(src_char_t* pos, int len) {
  int x = rand();
  int prom = x % 1000;
  if ((prom < perc[0]) && len >= 4) {
    // genrate an error
    return genError(pos, x, len);
  }
  if ((prom < perc[1]) && len >= 4) { // 4 byte
      // 4 byte uc  
      // Smiling Face with Sunglasses Emoji 
      // U + 1F60E
      // F0 9F 98 8E  U+
    *pos++ = 0xF0;
    *pos++ = 0x9F;
    *pos++ = 0x98;
    *pos++ = 0x8E;
    return 4;
  }
  if ((prom < perc[2]) && len >= 3) {
    *pos++ = 0xE2;
    *pos++ = 0x98;
    *pos++ = 0x80;
    return 3;
  }
  if ((prom < perc[3]) && len >= 2) {
    *pos++ = 0xc3;
    *pos++ = 0x84;
    return 2;
  }
  *pos++ = prom % 0x7F;
  return 1;
}


template<typename T> void dumpStartT(T* data, int dlen, offsets__t* offs, int start, int end) {

  if (!verbose) {
    return;
  }
  for (int i = start; i < end; ++i) {
    std::stringstream sr;
    std::cout 
      << "[" << std::setfill(' ') << std::setw(5) << i << "]"
      << std::setfill('0') << std::setw(7) << offs[i] << "," 
      << std::setfill('0') << std::setw(7) << offs[i+1] << " ";
    if (isInRange(offs[i], dlen) 
      && isInRange(offs[i + 1], dlen) 
      && offs[i] <= offs[i + 1
  ]) {
      for (int k = offs[i]; k < offs[i + 1]; ++k) {
        int val = data[k];
        if (val >= '0' &&  val <= 0x7F) {
          std::cout << (char)data[k];
        }
        else {
          std::stringstream sr;
          sr << "U+" << std::hex << (int16_t) data[k] << " ";
          std::cout << sr.str();
        }
      }
    }
    else {
      std::cout << "invalid";
    }
    std::cout << " (" << std::setfill(' ') << std::setw(5) << (offs[i+1] - offs[i]) << ")"<< std::endl;
  }
}

void dumpStart(src_char_t* data, int dlen, offsets__t* offs, int start, int end) {
  dumpStartT<src_char_t>(data, dlen, offs, start, end);
}

void dumpStartD(dst_char_t* data, int dlen, offsets__t* offs, int start, int end) {
  dumpStartT<dst_char_t>(data, dlen, offs, start, end);
}

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  assert(sizeof(int) == 4);
  StopWatchInterface* hTimer = NULL;

  printf("%s Starting...\n\n", argv[0]);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char**)argv);

  sdkCreateTimer(&hTimer);


  (verbose > 3) && printf("Initializing data...\n");
  (verbose > 3) && printf("...allocating CPU memory.\n");

  // length of source char type
#define src_char_t uint8_t
// length of dest char type
#define dst_char_t UChar32
// length of index type into the offset array
#define indexoff_t int
// type of the offset array
#define offsets__t int

  // for utf-8 -> uchar32 
  // worst case all ascii 1:1

  indexoff_t WORST_PER_INPUT_EL = 1; // every input element yields

  indexoff_t max_target_len = NRSTRINGS * AVG_LEN;
  // for now we overallocate...
  indexoff_t exact_dst_len = max_target_len;

  // srcdata
  src_char_t* srcData;
  indexoff_t* srcOffs;
  // dstData
  dst_char_t* dstData;
  indexoff_t* dstOffs;

  // cpu 
  srcData = (uint8_t*)malloc(NRSTRINGS * AVG_LEN);
  srcOffs = (int*)malloc(sizeof(int) * (NRSTRINGS + 1));

  dstOffs = (int*)malloc(sizeof(int) * (NRSTRINGS + 1));
  dstData = (UChar32*)malloc(sizeof(UChar32) * NRSTRINGS * AVG_LEN);

  //
  dst_char_t* h_dstData_GPU = (UChar32*)malloc(sizeof(UChar32) * NRSTRINGS * AVG_LEN);
  indexoff_t* h_dstOffs_GPU = (int*)malloc(sizeof(int) * (NRSTRINGS + 1));
  int* h_flawed_GPU = (int*)malloc(sizeof(int) * 1);

  
  printf("...allocating GPU memory.\n");

  uint8_t* d_srcData;
  UChar32* d_dstData;
  int* d_srcOffs;
  int* d_dstOffs;
  int* d_flawed;

  checkCudaErrors(cudaMalloc((void**)&d_srcData, sizeof(src_char_t) * NRSTRINGS * AVG_LEN));
  checkCudaErrors(cudaMalloc((void**)&d_srcOffs, sizeof(indexoff_t) * (NRSTRINGS + 1)));

  checkCudaErrors(cudaMalloc((void**)&d_dstData, sizeof(dst_char_t) * NRSTRINGS * AVG_LEN));
  checkCudaErrors(cudaMalloc((void**)&d_dstOffs, sizeof(indexoff_t) * (NRSTRINGS + 1)));
  checkCudaErrors(cudaMalloc((void**)&d_flawed, sizeof(int)));

  printf("...generating input data in CPU mem.\n");
#define RAND 1
#ifdef RAND
  srand(123);
  int base = 0;
  int tgt = base;
  srcOffs[0] = 0;
  for (int i = 0; i < NRSTRINGS; ++i) {
#if ALL_STRINGS_SAME
    srand(123);
#endif
    int len =(int) ((1.0-plen) * AVG_LEN + 1.0*(plen) * (rand() % (3 * AVG_LEN / 2)));
    while (tgt < base + len && tgt < NRSTRINGS * AVG_LEN) {
      tgt += addNextRandom(&srcData[tgt], base + len - tgt);
    }
    srcOffs[i + 1] = tgt;
    base = tgt;
  }
#else
  {
    int base = 0;
    int tgt = base;
    srcOffs[0] = 0;
    for (int i = 0; i < NRSTRINGS; ++i) {
      int len = std::min((i % AVG_LEN) + 5, AVG_LEN);
      if (tgt + 10 < NRSTRINGS * AVG_LEN) {
        int filled = snprintf((char*)&srcData[tgt], 10, "%d", i);
        tgt += filled;
      }
      for (; tgt < base + len && tgt < NRSTRINGS * AVG_LEN; ++tgt) {
        int offs = tgt - base;
        srcData[tgt] = ((i % 2) == 0)  ? 'a' + (offs % 27) : 'A' + (offs % 27);
      }
      srcOffs[i + 1] = tgt;
      base = tgt;
    }
  }


  srcData[srcOffs[2] + 2] = 0xc3;
  srcData[srcOffs[2] + 3] = 0x84;

  srcData[srcOffs[4] + 3] = 0xc3;
  srcData[srcOffs[4] + 4] = 0x84;


#endif 
  float sumL = 0;
  float squL = 0;
  for (int i = 0; i < NRSTRINGS; ++i) {
    long len = srcOffs[i + 1] - srcOffs[i];
    sumL += len;
    squL += (len * len);
  }

  std::cout << " here sumL" << sumL << " " << squL << " " << sumL*sumL/NRSTRINGS << std::endl;

  dumpStart( srcData, NRSTRINGS * AVG_LEN, srcOffs, 0, 3);
  dumpStart( srcData, NRSTRINGS * AVG_LEN, srcOffs,  NRSTRINGS-3, NRSTRINGS);

  int flawed = 0;

  const int NR_OFFSETS_N = NRSTRINGS + 1;
  const int nrStrings = NRSTRINGS;

  printf("...copying input data to GPU mem.\n");

  checkCudaErrors(cudaMemcpy(d_srcData, srcData, sizeof(src_char_t) * NRSTRINGS*AVG_LEN, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_srcOffs, srcOffs, sizeof(indexoff_t) * (NR_OFFSETS_N), cudaMemcpyHostToDevice));
  // clear flawed
  checkCudaErrors(cudaMemcpy(d_flawed, &flawed, sizeof(int), cudaMemcpyHostToDevice));

  printf("Data init done.\n");

  printf("Executing GPU kernel...\n");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  float tm_ms_gpu = 0.0;
  // model1, each warp processes exactly one string.
  dim3 block(NRBLOCK,1);
  dim3 grid( ((nrStrings+block.x)-1) / block.x, 1);
  {
    convUTF8_UCHAR32_kernel1 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);

    getLastCudaError("convUTF8_UCHAR32_kernel1() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    tm_ms_gpu = sdkGetTimerValue(&hTimer);
    printf("GPU time: %f msecs.\n", tm_ms_gpu);
  }
  sdkResetTimer(&hTimer);
  {
    sdkStartTimer(&hTimer);
    convUTF8_UCHAR32_kernel1Stck << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);
   // convUTF8_UCHAR32_kernel1 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);

    getLastCudaError("convUTF8_UCHAR32_kernel1() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    tm_ms_gpu = sdkGetTimerValue(&hTimer);
    printf("GPU time: %f msecs. (convUTF8_UCHAR32_kernel1Stck)\n", tm_ms_gpu);
  }
#ifdef NEXT
  {
    sdkStartTimer(&hTimer);
    const int stringsPerWarp = 16;
    dim3 gridN(((nrStrings + block.x) - 1) /(stringsPerWarp* block.x), 1);
    convUTF8_UCHAR32_kernel1StckN << <gridN, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed, stringsPerWarp);
    // convUTF8_UCHAR32_kernel1 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);

    getLastCudaError("convUTF8_UCHAR32_kernel1() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    tm_ms_gpu = sdkGetTimerValue(&hTimer);
    printf("GPU time: %f msecs. (convUTF8_UCHAR32_kernel1Stck)\n", tm_ms_gpu);
  }
#endif

  printf("Reading back GPU result...\n");

  memset(h_dstOffs_GPU, 0, sizeof(offsets__t) * (NRSTRINGS + 1));
  // Read back GPU results to compare them to CPU results
  checkCudaErrors(cudaMemcpy(h_dstData_GPU, d_dstData, sizeof(UChar32)* NRSTRINGS*AVG_LEN, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_dstOffs_GPU, d_dstOffs, sizeof(int)* (NRSTRINGS+1), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_flawed_GPU, d_flawed, sizeof(int)* 1, cudaMemcpyDeviceToHost));

  int flawed_GPU = *h_flawed_GPU;


  h_dstOffs_GPU[0] = 0;
  printf("Checking GPU results...\n");
  printf("..running CPU conversion\n");
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  flawed = convChunkUTF8_UCHAR32(srcData, srcOffs, nrStrings, dstData, dstOffs);
  sdkStopTimer(&hTimer);
  float tm_ms_cpu = sdkGetTimerValue(&hTimer);
  printf("CPU time: %f msecs.\n", tm_ms_cpu);

  dumpStartD(dstData, NRSTRINGS*AVG_LEN, dstOffs, 0, 6);
  std::cout << std::endl;
  dumpStartT(h_dstData_GPU, NRSTRINGS * AVG_LEN, h_dstOffs_GPU, 0, 6);
  std::cout << std::endl;
  std::cout << std::endl;
  dumpStartD(dstData, NRSTRINGS * AVG_LEN, dstOffs, NRSTRINGS - 6, NRSTRINGS);
  std::cout << std::endl;
  dumpStartT(h_dstData_GPU, NRSTRINGS* AVG_LEN, h_dstOffs_GPU, NRSTRINGS - 6, NRSTRINGS);
  std::cout << std::endl;




  /// <summary>
  /// index array  [0, 5, 5, 6, 10]
  /// ABCDE|F|GHIJ
  /// 
  /// "ABCDE",null,"F","GHJI"
  /// 
  /// We omit the 0 in the GPU processed arrays as it can be computed.
  /// </summary>
  /// <param name="argc"></param>
  /// <param name="argv"></param>
  /// <returns></returns>
  
  dst_char_t* h_dstDataNoGap_GPU = (dst_char_t*)malloc(sizeof(dst_char_t) * exact_dst_len);
  offsets__t* h_dstOffsNoGap_GPU = (offsets__t*)malloc(sizeof(offsets__t) * (NRSTRINGS +1));

  dst_char_t* srcWGaps = h_dstData_GPU;
  dst_char_t* dstNGaps = h_dstDataNoGap_GPU;

  printf("coalescing GPU CPU conversion\n");
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  const int sPerWarp = 1;
  // combine GPU results into contiguous string buffer
  indexoff_t last = 0;
  h_dstOffsNoGap_GPU[0] = h_dstOffs_GPU[0];
  for (int blk = 0; blk < grid.x; ++blk) {
    for (int treadidx = 0; treadidx < block.x; ++treadidx) {
      // we are looking at offsets from 
      indexoff_t ix = (blk * block.x + treadidx) * sPerWarp;
      indexoff_t ixend = ix + sPerWarp;
      verbose &&  std::cout << " t=" << treadidx << " b=" << blk << " " << block.x << " " << grid.x << " indexes[" << ix << ", " << ixend << "[" << std::endl;
      offsets__t unshifted_dest = srcOffs[ix] * WORST_PER_INPUT_EL;
      offsets__t end_dest = h_dstOffs_GPU[ixend];
      offsets__t delta = unshifted_dest - last;
      int len = end_dest - unshifted_dest;

      verbose && std::cout << " delta : " << delta << " src [" << unshifted_dest << "- " << unshifted_dest + len << "[  => " << "[" << last << " - " << last + len << "[" << std::endl; 
      assert(delta >= 0);
      dst_char_t* dst = &dstNGaps[last];
      dst_char_t* src = &srcWGaps[unshifted_dest];
      memcpy( (void*) dst, (void*) src, len * sizeof(dst_char_t));
      for (indexoff_t idxOff = ix; idxOff < ixend; ++idxOff) {
        h_dstOffsNoGap_GPU[idxOff+1] = h_dstOffs_GPU[idxOff+1] - delta;
      }
      last += len; 
    }
  }
  sdkStopTimer(&hTimer);
  float tm_ms_coa = sdkGetTimerValue(&hTimer);
  printf("coalesce GPU result time: %f msecs.\n", tm_ms_coa);


  dumpStartT(h_dstData_GPU, NRSTRINGS * AVG_LEN, h_dstOffs_GPU, 0, 6);
  std::cout << "nogap " << std::endl;
  dumpStartT(h_dstDataNoGap_GPU, NRSTRINGS * AVG_LEN, h_dstOffsNoGap_GPU, 0, 6);
  std::cout << std::endl;
  std::cout << std::endl;
  dumpStartT(h_dstData_GPU, NRSTRINGS * AVG_LEN, h_dstOffs_GPU, NRSTRINGS - 6, NRSTRINGS);
  std::cout << "nogap " <<  std::endl;
  dumpStartT(h_dstDataNoGap_GPU, NRSTRINGS * AVG_LEN, h_dstOffsNoGap_GPU, NRSTRINGS - 6, NRSTRINGS);
  std::cout << std::endl;


  printf("...comparing the results\n");




  // Calculate max absolute difference and L1 distance
  // between CPU and GPU results
  int diff_idx = 0;
  int diff_char = 0;

  for (int i = 0; i < nrStrings + 1; i++) {
    if (h_dstOffsNoGap_GPU[i] != dstOffs[i]) {
      ++diff_idx;
    }
  }
  if (diff_idx == 0) {
    for (int i = 0; i < nrStrings; i++) {
      indexoff_t len = dstOffs[i + 1] - dstOffs[i];
      for (indexoff_t pos = 0; pos  < len; ++pos) {
        if (h_dstDataNoGap_GPU[pos] != dstData[pos]) {
          ++diff_char;
        }
      }
    }
  }
  offsets__t* dstCUM = (offsets__t*) malloc(sizeof(offsets__t) * NRSTRINGS+1);

  {
    thrust::plus<int> binary_op;

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    thrust::inclusive_scan(dstOffs, dstOffs + NRSTRINGS + 1, dstCUM, binary_op);
    sdkStopTimer(&hTimer);
    float tm_ms_cum = sdkGetTimerValue(&hTimer);
    printf("GPU time: cum %f msecs.\n", tm_ms_cum);
  }

  printf("Shutting down...\n");

  checkCudaErrors(cudaFree(d_flawed));
  checkCudaErrors(cudaFree(d_dstOffs));
  checkCudaErrors(cudaFree(d_dstData));

  checkCudaErrors(cudaFree(d_srcOffs));
  checkCudaErrors(cudaFree(d_srcData));

  free(h_dstData_GPU);
  free(h_dstOffs_GPU);
  free(h_dstDataNoGap_GPU);
  free(h_dstOffsNoGap_GPU);
  sdkDeleteTimer(&hTimer);

  printf(" factor=%f gpu=%5.2f msec cpu=%5.2f msec  diff idx: %d %d %d %d  AVG_LEN=%d  NRSTRING=%d SIZE=%d BLK=%d (e|4|3|2|1) (%d|%d|%d|%d|%d) %8.1f %8.3f\n",
    tm_ms_cpu / tm_ms_gpu, tm_ms_gpu, tm_ms_cpu, diff_idx, diff_char, flawed, flawed_GPU,
    (int)AVG_LEN, (int)NRSTRINGS, (int)NRSTRINGS * AVG_LEN,
    (int)NRBLOCK,
    perc[0], perc[1], perc[2], perc[3], 1000 - perc[3],
    ((float)sumL) / NRSTRINGS,
    ((float)squL - sumL * sumL/(1.0*NRSTRINGS))*NRSTRINGS / ((1.0*sumL*sumL))
    );
  exit((diff_idx == 0) && (diff_char == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}
