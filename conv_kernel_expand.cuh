/* Copyright (c) 2022, gerd forstmann, All rights reserved.
 *
 */
/*
#define NR 1
#define SINP 000
#define SOUT 256
#define SPERWARP 1
#define FCT U8_NEXT
#define IMPL(s,i,length,c, fl)  U8_NEXT(s, i, length, c, fl)
*/
#ifndef NR
#error "Please define nr "
#endif
#ifndef NEXT_CHAR
#error "Please define NEXT_CHAR "
#endif
#ifndef signature
#error "Please define FCT "
#endif
#ifndef STRPERTHREAD
#error "Please define STRPERTHREAD "
#endif

///////////////////////////////////////////////////////////////////////////////
// a kernel which use stack memory as a buffer for WRITING
// (tests also buffering input buffer had not significant effect)
// this kernel avoids branch switching
//
// sth like __global__ void convUTF8_UCHAR32_kernel2Stck(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed, int strPerThread = 1) {
signature
  ////////////////////////////////////////////////////////////////////////////
  // (gridDim.x * blockDim.x)
  //assert(sizeof(int) == 4);
  //assert(sizeof(UChar32) == 4);
  const UChar32 UC_REPL = 0xFFFD;
#if (STRPERTHREAD > 1)
  int strPerWarp = strPerThread;
#else
  const int strPerWarp = 1;
#endif
  int ix = (threadIdx.x + IMUL(blockIdx.x, blockDim.x)) * strPerWarp;
  //int ix = threadIdx.x * gridDim.x + blockIdx.x; //  IMUL(blockIdx.x, blockDim.x)) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStringN) {
    return;
  }
  const int WORST_PER_EL = 1;
  const int BUFLEN = 5 * 64;
#if SINP > 0
  // using a local buffer for input
  uint8_t srcBuffer[BUFLEN];
#endif
  int lastStringEnd = 0;
#ifdef NO_OFFSET
  int destOffset = dO[ix]; // we know the offset!
#else
  int* destOffsets = dO + 1; // first one is 0!
  int destOffset = srcOffsets[ix] * WORST_PER_EL;
#endif
#ifndef NO_WRITE
  UChar32* dest = &dest0[destOffset]; // worst case, every byte an UChar
#if SOUT > 0
  UChar32 Buffer[SOUT * WORST_PER_EL];
  UChar32* destBuffer = &Buffer[0];
  offsets__t done = 0;
#else
  UChar32* destBuffer = dest;
#endif
#endif
  int32_t tgt = 0;
  bool flaw = 0;
  for (int nrString = ix; nrString < ixLast && nrString < nrStringN; ++nrString) {
    // the string spans from
    // the most naive implementation.
    int start = srcOffsets[nrString];
    int end = srcOffsets[nrString + 1];
    int length = end - start;
#if SINP > 0
    for (int i = 0; i < std::min(length,SINP); ++i) {
      srcBuffer[i] = src[start + i];
    }
#else
    const uint8_t* srcBuffer = &src[start];
#endif

#ifdef VERBOSE
    printf(" t=%d b=%d /%d [%d,%d[  idx= %d  start-end[%d,%d] (%d) \n", (int)threadIdx.x, (int)blockIdx.x, (int)blockDim.x, (int)ix, (int)ixLast, nrString, start, end, length);
#endif
    const uint8_t* s = srcBuffer;
    UChar32 c;

    int32_t i = 0;
    while (i < length) {
      NEXT_CHAR(s, i, length, c, flaw); // advances i, may set flaw to 1
#ifdef NO_WRITE
      ++tgt;
#else
      destBuffer[tgt++] = c;
#if SOUT > 0
      if (tgt % SOUT == 0) {
        for (int ii = 0; ii < SOUT; ++ii) {
          dest[done+ii] = destBuffer[done+ii];
        }
        done = tgt;
        destBuffer -= SOUT;
      }
#endif
#endif
    }
    // set length for this string
#ifndef NO_OFFSET
    destOffsets[nrString] = tgt - lastStringEnd;/* + startOffset*/;
    lastStringEnd = tgt;
#endif
#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) set dest[%d] = %d \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast,
      nrString, start, end, length, nrString, tgt + startOffset);
#endif
  }
#ifndef NO_WRITE
#if SOUT > 0
  // write remaining
  for (int iabs = done; iabs < tgt; ++iabs) {
    dest[iabs] = destBuffer[iabs];
  }
#endif
#endif
#ifdef NO_WRITE
  if (flaw) {
    *flawed = 1;
  }
#endif
  return;
}
