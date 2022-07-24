/* Copyright (c) 2022, gerd forstmann, All rights reserved.
 * 
 */

//#include <cooperative_groups.h>
#include "../icu/source/common/unicode/utf8.h"

//# namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
// On G80-class hardware 24-bit multiplication takes 4 clocks per warp
// (the same as for floating point  multiplication and addition),
// whereas full 32-bit multiplication takes 16 clocks per warp.
// So if integer multiplication operands are  guaranteed to fit into 24 bits
// (always lie within [-8M, 8M - 1] range in signed case),
// explicit 24-bit multiplication is preferred for performance.
///////////////////////////////////////////////////////////////////////////////
#define IMUL(a, b) __mul24(a, b)

///////////////////////////////////////////////////////////////////////////////
__global__ void convUTF8_UCHAR32_kernel1(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed) {
////////////////////////////////////////////////////////////////////////////
  const UChar32 UC_REPL = 0xFFFD;
//     blk= 1... bDx
//     th = 1....gDx                // bdx

  //assert(sizeof(int) == 4);
  //assert(sizeof(UChar32) == 4);

  const int strPerWarp = 1;
  int ix = (threadIdx.x + IMUL(blockIdx.x,blockDim.x)) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStringN) {
    return;
  }
  const int WORST_PER_EL = 1;

  int* destOffsets = dO + 1; // first one is 0!
  int startOffset = srcOffsets[ix] * WORST_PER_EL;
  UChar32* dest = &dest0[startOffset]; // worst case, every byte an UChar
  int32_t tgt = 0;
  for (int nrString = ix; nrString < ixLast && nrString < nrStringN; ++nrString) {
    // the string spans from 
    // the most naive implementation.
    int start = srcOffsets[nrString];
    int end = srcOffsets[nrString + 1];
    int length = end - start;
#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast, nrString, start, end, length);
#endif
    uint8_t* s = src + start;
    UChar32 c;
    int32_t i = 0;
    while (i < length) {
      U8_NEXT(s, i, length, c); // advances i
      if (c < 0) [[unlikely]] {
        dest[tgt++] = UC_REPL;
        *flawed = 1; // racy, but only transitions from 0 -> 1
      }
      else {
        dest[tgt++] = c;
      }
    }
    destOffsets[nrString] = tgt + startOffset;
#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) set dest[%d] = %d \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast,
      nrString, start, end, length, nrString, tgt + startOffset);
#endif
  }
  return;
}



///////////////////////////////////////////////////////////////////////////////

// a kernel which use stack memory as a buffer for WRITING
// (tests also buffering input buffer had not significant effect)
__global__ void convUTF8_UCHAR32_kernel1Stck(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed) {
  ////////////////////////////////////////////////////////////////////////////
  const UChar32 UC_REPL = 0xFFFD;
  //     blk= 1... bDx
  //     th = 1....gDx                // bdx

  //assert(sizeof(int) == 4);
  //assert(sizeof(UChar32) == 4);
  const int strPerWarp = 1;
  int ix = (threadIdx.x + IMUL(blockIdx.x, blockDim.x)) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStringN) {
    return;
  }
  const int WORST_PER_EL = 1;
  const int BUFLEN = 2*64;
#ifdef STACK_IN
  // using a local buffer for input
  uint8_t srcBuffer[BUFLEN];
#endif
  UChar32 Buffer[BUFLEN * WORST_PER_EL];
  UChar32* destBuffer = &Buffer[0];
  int* destOffsets = dO + 1; // first one is 0!
  int startOffset = srcOffsets[ix] * WORST_PER_EL;
  UChar32* dest = &dest0[startOffset]; // worst case, every byte an UChar
  int32_t tgt = 0;
  for (int nrString = ix; nrString < ixLast && nrString < nrStringN; ++nrString) {
    // the string spans from 
    // the most naive implementation.
    int start = srcOffsets[nrString];
    int end = srcOffsets[nrString + 1];
    int length = end - start;
#ifdef STACK_IN
    for (int i = 0; i < length; ++i) {
      srcBuffer[i] = src[start + i];
    }
#else
    uint8_t *srcBuffer = &src[start];
#endif

#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast, nrString, start, end, length);
#endif
    uint8_t* s = srcBuffer;
    UChar32 c;
    int32_t i = 0;
    while (i < length) {
      U8_NEXT(s, i, length, c); // advances i
      if (c < 0) [[unlikely]] {
        destBuffer[tgt++] = UC_REPL;
        *flawed = 1; // racy, but only transitions from 0 -> 1
      }
      else {
        destBuffer[tgt++] = c;
      }
      if (tgt % BUFLEN == 0) {
        for (int i = 0; i < BUFLEN; ++i) {
          dest[i] = destBuffer[i];
        }
        dest -= BUFLEN;
        destBuffer -= BUFLEN;
      }
    }
    // no real harm or benefit __syncthreads();
    destOffsets[nrString] = tgt + startOffset;
    // memcpy(dest, destBuffer, sizeof(UChar32) * tgt);
    for (int i = 0; i < (tgt % BUFLEN); ++i) {
      dest[i] = destBuffer[i];
    }
#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) set dest[%d] = %d \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast,
      nrString, start, end, length, nrString, tgt + startOffset);
#endif
  }
  return;
}

////////////////////////////////////////////////////////////////////////////
// this kernel processes strPerThread contiguous strings per thread
__global__ void convUTF8_UCHAR32_kernel2(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed, int strPerThread) {
////////////////////////////////////////////////////////////////////////////
  const UChar32 UC_REPL = 0xFFFD;
  //     blk= 1... bDx
  //     th = 1....gDx                // bdx

  assert(sizeof(int) == 4);
  assert(sizeof(UChar32) == 4);

  const int strPerWarp = strPerThread;
  int ix = (threadIdx.x + blockIdx.x * blockDim.x) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStringN) {
    return;
  }
  const int WORST_PER_EL = 1;

  int* destOffsets = dO + 1; // first one is 0!
  int startOffset = srcOffsets[ix] * WORST_PER_EL;
  UChar32* dest = &dest0[startOffset]; // worst case, every byte an UChar
  int32_t tgt = 0;
  for (int nrString = ix; nrString < ixLast && nrString < nrStringN; ++nrString) {
    // the string spans from 
      // the most naive implementation.
    int start = srcOffsets[nrString];
    int end = srcOffsets[nrString + 1];
    int length = end - start;
#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast, nrString, start, end, length);
#endif
    uint8_t* s = src + start;
    UChar32 c;
    int32_t i = 0;
    while (i < length) {
      U8_NEXT(s, i, length, c); // advances i
      if (c < 0) [[unlikely]] {
        dest[tgt++] = UC_REPL;
        *flawed = 1; // racy, but only transitions from 0 -> 1
      }
      else {
        dest[tgt++] = c;
      }
    }
    destOffsets[nrString] = tgt + startOffset;
#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) set dest[%d] = %d \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast,
      nrString, start, end, length, nrString, tgt + startOffset);
#endif
  }
  return;
}


// the shift is a special form of a scan. 
// if we store length, not offsets in the target array, we can use a simple scan to produce the final array
// but we need the source offsets to copy the data.
// 
// thrust::inclusive_scan(data, data + 6, data); // in-place scan 

// parallel prefix sum kernel
// https://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf
//Reading – Mark Harris, Parallel Prefix Sum with CUDA
// http ://developer.download.nvidia.com/compute/cuda/1_1/Website/
//projects / scan / doc / scan.pdf
  // todo accumulate flawed
  //return;
  //return flawed;
  /*
    interesting, as https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/ claims at least
       __syncthreads() (? identical to cg::sync(cta) must be used at "same position ( no divergence) otherwise deadlocks
    ////////////////////////////////////////////////////////////////////////
    // Perform tree-like reduction of accumulators' results.
    // ACCUM_N has to be power of two at this stage
    ////////////////////////////////////////////////////////////////////////
    for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);

      for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x)
        accumResult[iAccum] += accumResult[stride + iAccum];
    }

    cg::sync(cta);

    if (threadIdx.x == 0) d_C[vec] = accumResult[0];
    */
//}
