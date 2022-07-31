/* Copyright (c) 2022, gerd forstmann, All rights reserved.
 *
 */

#define DEST_LEN 1


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
  offsets__t lasttgt = 0;
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
#ifdef DEST_LEN
    destOffsets[nrString] = tgt - lasttgt;/* + startOffset*/;
    lasttgt = tgt;
#else
    destOffsets[nrString] = tgt + startOffset;
#endif
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
  // (gridDim.x * blockDim.x)
  //assert(sizeof(int) == 4);
  //assert(sizeof(UChar32) == 4);
  const int strPerWarp = 1;
  int ix = (threadIdx.x + IMUL(blockIdx.x, blockDim.x)) * strPerWarp;
  //int ix = threadIdx.x * gridDim.x + blockIdx.x; //  IMUL(blockIdx.x, blockDim.x)) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStringN) {
    return;
  }
  offsets__t lasttgt = 0;
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
    printf(" t=%d b=%d /%d [%d,%d[  idx= %d  start-end[%d,%d] (%d) \n", (int)threadIdx.x, (int)blockIdx.x, (int) blockDim.x, (int)ix, (int)ixLast, nrString, start, end, length);
#endif
    uint8_t* s = srcBuffer;
    UChar32 c;
    int32_t i = 0;
    offsets__t done = 0;
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
          dest[done+i] = destBuffer[done+i];
        }
        done = tgt;
        //dest += BUFLEN; // ?
        destBuffer -= BUFLEN;
      }
    }
    // no real harm or benefit __syncthreads();
#ifdef DEST_LEN
    destOffsets[nrString] = tgt - lasttgt;/* + startOffset*/;
    lasttgt = tgt;
#else
    destOffsets[nrString] = tgt + startOffset;
#endif
    // memcpy(dest, destBuffer, sizeof(UChar32) * tgt);
    for (int i = done; i < tgt; ++i) {
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
__global__ void convUTF8_UCHAR32_kernel2(uint8_t* src, offsets__t src_len, offsets__t* srcOffsets, int nrStrings, UChar32* dest0, int* dO, int* flawed, int strPerThread) {
////////////////////////////////////////////////////////////////////////////
  const UChar32 UC_REPL = 0xFFFD;
  //     blk= 1... bDx
  //     th = 1....gDx                // bdx

  assert(sizeof(int) == 4);
  assert(sizeof(UChar32) == 4);
  offsets__t lasttgt = 0;
  const int strPerWarp = strPerThread;
  int ix = (threadIdx.x + blockIdx.x * blockDim.x) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStrings) {
    return;
  }
  const int WORST_PER_EL = 1;

  int* destOffsets = dO + 1; // first one is 0!
  int startOffset = srcOffsets[ix] * WORST_PER_EL;
  UChar32* dest = &dest0[startOffset]; // worst case, every byte an UChar
  int32_t tgt = 0;
  for (int nrString = ix; nrString < ixLast && nrString < nrStrings; ++nrString) {
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
#ifdef DEST_LEN
    destOffsets[nrString] = tgt - lasttgt;/* + startOffset*/;
    lasttgt = tgt;
#else
    destOffsets[nrString] = tgt + startOffset;
#endif
#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) set dest[%d] = %d \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast,
      nrString, start, end, length, nrString, tgt + startOffset);
#endif
  }
  return;
}


/// <summary>
///  copies back the fragmented chunks of data into a single coalesced
/// </summary>
/// <param name="mid"></param>
/// <param name="srcOffsets"></param>
/// <param name="nrStrings"></param>
/// <param name="dst"></param>
/// <param name="dstOffsets">the correclty computed dest offsets</param>
/// <returns></returns>
__global__ void memoryCoalesce(UChar32* mid, int* srcOffsets, int nrStrings, UChar32* dst, int* dstOffsets) {
  const int strPerWarp = 1;
  int ix = (threadIdx.x + blockIdx.x * blockDim.x) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStrings) {
    return;
  }
  const int WORST_PER_EL = 1;
  for (int nrString = ix; nrString < ixLast && nrString < nrStrings; ++nrString) {
    offsets__t from = srcOffsets[ix] * WORST_PER_EL;
    offsets__t to = dstOffsets[ix];
    offsets__t len = dstOffsets[ix + strPerWarp] - to;
    for (int i = 0; i < len; ++i) {
      dst[to + i] = mid[from + i];
    }
  }
}

void U8_NEXT_NB_i(const uint8_t* src, int& i, int length, UChar32& r, bool verbose) {
  const UChar32 UC_REPL = 0xFFFD;
  if (length - i < 4) {
    U8_NEXT(src, i, length, r);
    return;
  }
  uint8_t c0 = src[i];
  uint8_t c1 = src[i + 1];
  uint8_t c2 = src[i + 2];
  uint8_t c3 = src[i + 3];
  const uint8_t maskL6 = 0b11111110;
  const uint8_t mtchL6 = 0b11111100;
  const uint8_t maskL5 = 0b11111100;
  const uint8_t mtchL5 = 0b11111000;
  const uint8_t maskL4 = 0b11111000;
  const uint8_t mtchL4 = 0b11110000;
  const uint8_t chk1L4 = 0b00000111;
  const uint8_t chkZL4 = 0b00000111;
  const uint8_t maskL3 = 0b11110000;
  const uint8_t mtchL3 = 0b11100000;
  const uint8_t chk1L3 = 0b00001111;
  const uint8_t chkZL3 = 0b00000111;
  const uint8_t maskL2 = 0b11100000;
  const uint8_t mtchL2 = 0b11000000;
  const uint8_t chk1L2 = 0b00011110; // ! only 1 is non-minimal
  const uint8_t maskFl = 0b11000000;
  const uint8_t mtchFl = 0b10000000;
  const uint8_t maskL1 = 0b10000000;
  const uint8_t mtchL1 = 0b00000000;

  const uint8_t chk1Fl = 0b00100000; // mask 1 leading data bit in a follow sequence
  const uint8_t chk2Fl = 0b00110000; // mask 2 leading data bits in a follow sequence


#define AnyLeadDataBitSet(X,c) (!!(c & chk##X##Fl))
#define IsLeadZero(X,c) ( !(c& chkZ##X) )

#define Is(X,c)  !((c & mask##X) ^ mtch##X)
#define IsNz(X,c)  (!!(c & chk1##X))

// the bits in the mask must be exactly the value
#define IsMV(c, mask,value) (!( (c & mask ) - value))

  // h2-h6 header bits indicate 2-6 byte sequence

  bool h7 = IsMV(c0, 0xFE, 0xFE);
  bool h6 = Is(L6, c0);
  bool h5 = Is(L5, c0);
  bool h4 = Is(L4, c0);

  // 0xF4  < 0x8A  or
  //  0x20007 f0 a0 80
  // 0x10FFFF f4 a0
  // 0xf0 0xa0
  bool o4NonMinimal = (!(c0 & 0b00000111)) && !(c1 & 0b00110000);
  bool o4Overflow = (IsMV(c0, 0xFF, 0xF4) && !IsMV(c1, 0b00110000, 0b00000000))
    || IsMV(c0, 0xFF, 0xF5)
    || IsMV(c0, 0xFF, 0xF6)
    || IsMV(c0, 0xFF, 0xF7);
  bool o4 = !o4NonMinimal && !o4Overflow;
//  bool o4TooLarge = (c0 & 0b00000011) || ;// (10FFFF)
  bool i4 = Is(L4, c0) && Is(Fl, c1) && Is(Fl, c2) && Is(Fl, c3) && o4;
  bool h3 = Is(L3, c0);
  bool isSurrogate = /*ed a0  - ed bF surrogate pair*/ IsMV(c0, 0xFF, 0xED) && IsMV(c1, 0x20, 0x20);
  bool o3 = /* check for minimal */(IsNz(L3, c0) || (IsLeadZero(L3, c0) && AnyLeadDataBitSet(1, c1)))
        /* exclude surrogates */
        && !(isSurrogate);
  bool i3 = Is(L3, c0) && Is(Fl, c1) && Is(Fl, c2) && o3;
  bool h2 = Is(L2, c0);
  bool o2 = IsNz(L2, c0);
  bool i2 = Is(L2, c0) && Is(Fl, c1) && o2;
  bool i1 = Is(L1, c0);
  bool hf = Is(Fl, c0);

  bool isNonMinimal3 = !o3;
  // 0x11110000 0b1001

#define V4_0(c)    (((UChar32) c &0b00000111 ) <<  (18))
#define V4_1(c)    (((UChar32) c &0b00111111 ) <<  (12))
#define V4_2(c)    (((UChar32) c &0b00111111 ) <<  (6))
#define V4_3(c)    (((UChar32) c &0b00111111 ) <<  (0))

#define V3_0(c)    (((UChar32) c &0b00001111 ) <<  (12))
#define V3_1(c)    (((UChar32) c &0b00111111 ) <<  (6))
#define V3_2(c)    (((UChar32) c &0b00111111 ) <<  (0))

#define V2_0(c)    (((UChar32) c &0b00011111 ) <<  (6))
#define V2_1(c)    (((UChar32) c &0b00111111 ) <<  (0))
#define V1_0(c)    (((UChar32) c &0b01111111 ) <<  (0))
  // icu varies in error handling
  // on "skipping" bytes, for e2 82 *41 fx with * beeing an unexpected non tail
  // it skips 2 chars (e2 82), re adjusting at 41 together
  //
  //
  bool err = (h2 && !i2) || (h3 && !i3) || (h4 && !i4) || hf || h5 || h6 || h7 ;

  offsets__t advance = i1 * 1 + i2 * 2 + i3 * 3 + i4 * 4
    + err * 1
    + (err * (h3 && o3 && Is(L3, c0) && Is(Fl, c1) && !Is(Fl,c2))) // if everything up to 2nd byte in 3-byte seq looks ok
    + (err * (h4 && o4 && Is(L4, c0) && Is(Fl, c1)) * (1 + Is(Fl, c2)))
    //+ (err * (h4 && o4 && Is(L4, c0) && Is(Fl, c1) &&  !Is(Fl, c2)))
    //+ (2*err * (h4 && o4 && Is(L4, c0) && Is(Fl, c1) && Is(Fl, c2) && !Is(Fl,c3)))
  +0;
  /* ed a0 80  -> 0xd800 surrogate pair*/
  /* ed a0 80  -> 0xdFFF*/
  /* (s)[i] - 0x80) <= 0x3f */
  /*
    + 1 * (Is(L3, c0) && Is(Fl, c1) && !Is(Fl, c2)) * o3 * (
      ((!IsLeadZero(L3, c0) && !AnyLeadDataBitSet(1, c1))
        || (IsLeadZero(L3, c0) && AnyLeadDataBitSet(1, c1))
        || (!IsLeadZero(L3, c0) && AnyLeadDataBitSet(1, c1))))*/;

  if (verbose) {
    std::cout << " i=" << i << " err=" << err << " adv=" << advance << std::endl;
    std::cout << i1 << "/" << i2 << "/" << i3 << "/" << i4 << std::endl;
    std::cout << " i1/2/3/4     "
      << i1 << "/" << i2 << "/" << i3 << "/" << i4 << std::endl;
    std::cout << " h-/2/3/4/5/6 "
      << "-" << "/" << h2 << "/" << h3 << "/" << h4 <<"/" << h5 <<"/" << h6<< std::endl;
    std::cout << " o-/2/3/4     "
      << "-" << "/" << o2 << "/" << o3 << "/" << o4 << std::endl;
    std::cout << "o4NM" << o4NonMinimal <<" c1= " << c1 <<
      " IsMV(c1, 0b00111000, 0b00000000) " << IsMV(c1, 0b00111000, 0b00000000) << " ";
      std::cout << " c1 & " << (int) (c1 & 0b00111000) << " " << !(int)(c1 & 0b001111000) << std::endl;
std::cout << "\n overflow" << o4Overflow << " " <<
      (!IsMV(c1, 0b00110000, 0b00000000)) << std::endl;

    std::cout << " IS Surr" << isSurrogate << " ff ed " << IsMV(c0, 0xFF, 0xED) << " " <<  IsMV(c1, 0x20, 0x20) << std::endl;
    std::cout << " o4 = " << "IsMV(c1, 0b00011111, 0b00010000)" << IsMV(c1, 0b00011111, 0b00010000) << "!IsMV(c1, 0b00010000, 0b00000000)" << !IsMV(c1, 0b00010000, 0b00000000) << std::endl;
    std::cout << " (IsLeadZero(L3, c0) && AnyLeadDataBitSet(1,c1));" << IsLeadZero(L3, c0) << AnyLeadDataBitSet(1, c1) << std::endl;
    std::cout << " extraskip (Is(L3, c0) && Is(Fl, c1) && !Is(Fl, c2)) << "
      << Is(L3, c0) << " && " << Is(Fl, c1) << " && " << !Is(Fl, c2) << std::endl;
  }
  r = i1 * c0
    + i2 * (V2_0(c0) + V2_1(c1))
    + i3 * (V3_0(c0) + V3_1(c1) + V3_2(c2))
    + i4 * (V4_0(c0) + V4_1(c1) + V4_2(c2) + V4_3(c3))
    + err * (-1); // Neg = error;
  i += advance;
}

void U8_NEXT_NB(const uint8_t* src, int& i, int length, UChar32& r) {
  U8_NEXT_NB_i(src, i, length, r, false);
}
void U8_NEXT_NB_V(const uint8_t* src, int& i, int length, UChar32& r) {
  U8_NEXT_NB_i(src, i, length, r, true);
}


__device__ void convU8_NEXT_NOBRANCH(const uint8_t* src, int& i, int length, UChar32& r, bool& flawed) {
  const UChar32 UC_REPL = 0xFFFD;
  if (length - i < 4) {
    U8_NEXT(src, i, length, r);
    return;
  }
  uint8_t c0 = src[i];
  uint8_t c1 = src[i + 1];
  uint8_t c2 = src[i + 2];
  uint8_t c3 = src[i + 3];
  const uint8_t maskL6 = 0b11111110;
  const uint8_t mtchL6 = 0b11111100;
  const uint8_t maskL5 = 0b11111100;
  const uint8_t mtchL5 = 0b11111000;
  const uint8_t maskL4 = 0b11111000;
  const uint8_t mtchL4 = 0b11110000;
  const uint8_t chk1L4 = 0b00000111;
  const uint8_t chkZL4 = 0b00000111;
  const uint8_t maskL3 = 0b11110000;
  const uint8_t mtchL3 = 0b11100000;
  const uint8_t chk1L3 = 0b00001111;
  const uint8_t chkZL3 = 0b00000111;
  const uint8_t maskL2 = 0b11100000;
  const uint8_t mtchL2 = 0b11000000;
  const uint8_t chk1L2 = 0b00011110; // ! only 1 is non-minimal
  const uint8_t maskFl = 0b11000000;
  const uint8_t mtchFl = 0b10000000;
  const uint8_t maskL1 = 0b10000000;
  const uint8_t mtchL1 = 0b00000000;

  const uint8_t chk1Fl = 0b00100000; // mask 1 leading data bit in a follow sequence
  const uint8_t chk2Fl = 0b00110000; // mask 2 leading data bits in a follow sequence


#define AnyLeadDataBitSet(X,c) (!!(c & chk##X##Fl))
#define IsLeadZero(X,c) ( !(c& chkZ##X) )

#define Is(X,c)  !((c & mask##X) ^ mtch##X)
#define IsNz(X,c)  (!!(c & chk1##X))

// the bits in the mask must be exactly the value
#define IsMV(c, mask,value) (!( (c & mask ) - value))

  // h2-h6 header bits indicate 2-6 byte sequence

  bool h7 = IsMV(c0, 0xFE, 0xFE);
  bool h6 = Is(L6, c0);
  bool h5 = Is(L5, c0);
  bool h4 = Is(L4, c0);

  // 0xF4  < 0x8A  or
  //  0x20007 f0 a0 80
  // 0x10FFFF f4 a0
  // 0xf0 0xa0
  bool o4NonMinimal = (!(c0 & 0b00000111)) && !(c1 & 0b00110000);
  bool o4Overflow = (IsMV(c0, 0xFF, 0xF4) && !IsMV(c1, 0b00110000, 0b00000000))
    || IsMV(c0, 0xFF, 0xF5)
    || IsMV(c0, 0xFF, 0xF6)
    || IsMV(c0, 0xFF, 0xF7);
  bool o4 = !o4NonMinimal && !o4Overflow;
  //  bool o4TooLarge = (c0 & 0b00000011) || ;// (10FFFF)
  bool i4 = Is(L4, c0) && Is(Fl, c1) && Is(Fl, c2) && Is(Fl, c3) && o4;
  bool h3 = Is(L3, c0);
  bool isSurrogate = /*ed a0  - ed bF surrogate pair*/ IsMV(c0, 0xFF, 0xED) && IsMV(c1, 0x20, 0x20);
  bool o3 = /* check for minimal */(IsNz(L3, c0) || (IsLeadZero(L3, c0) && AnyLeadDataBitSet(1, c1)))
    /* exclude surrogates */
    && !(isSurrogate);
  bool i3 = Is(L3, c0) && Is(Fl, c1) && Is(Fl, c2) && o3;
  bool h2 = Is(L2, c0);
  bool o2 = IsNz(L2, c0);
  bool i2 = Is(L2, c0) && Is(Fl, c1) && o2;
  bool i1 = Is(L1, c0);
  bool hf = Is(Fl, c0);

  bool isNonMinimal3 = !o3;
  // 0x11110000 0b1001

#define V4_0(c)    (((UChar32) c &0b00000111 ) <<  (18))
#define V4_1(c)    (((UChar32) c &0b00111111 ) <<  (12))
#define V4_2(c)    (((UChar32) c &0b00111111 ) <<  (6))
#define V4_3(c)    (((UChar32) c &0b00111111 ) <<  (0))

#define V3_0(c)    (((UChar32) c &0b00001111 ) <<  (12))
#define V3_1(c)    (((UChar32) c &0b00111111 ) <<  (6))
#define V3_2(c)    (((UChar32) c &0b00111111 ) <<  (0))

#define V2_0(c)    (((UChar32) c &0b00011111 ) <<  (6))
#define V2_1(c)    (((UChar32) c &0b00111111 ) <<  (0))
#define V1_0(c)    (((UChar32) c &0b01111111 ) <<  (0))
  // icu varies in error handling
  // on "skipping" bytes, for e2 82 *41 fx with * beeing an unexpected non tail
  // it skips 2 chars (e2 82), re adjusting at 41 together
  //
  //
  bool err = (h2 && !i2) || (h3 && !i3) || (h4 && !i4) || hf || h5 || h6 || h7;
  offsets__t advance = i1 * 1 + i2 * 2 + i3 * 3 + i4 * 4
    + err * (1 +  (Is(L3, c0) && h3 && o3 && Is(Fl, c1) && !Is(Fl, c2)) // if everything up to 2nd byte in 3-byte seq looks ok
             +    (Is(L4, c0) && h4 && o4 && Is(Fl, c1)) * (1 + Is(Fl, c2)))
    //+ (err * (h4 && o4 && Is(L4, c0) && Is(Fl, c1) &&  !Is(Fl, c2)))
    //+ (2*err * (h4 && o4 && Is(L4, c0) && Is(Fl, c1) && Is(Fl, c2) && !Is(Fl,c3)))
    + 0;
   r = i1 * c0
          + IMUL(i2 , (V2_0(c0) + V2_1(c1)))
          + IMUL(i3 , (V3_0(c0) + V3_1(c1) + V3_2(c2)))
          + i4 * (V4_0(c0) + V4_1(c1) + V4_2(c2) + V4_3(c3))
          + IMUL(err,UC_REPL); // Neg = error;
   flawed |= err;
   i += advance;

#ifdef OLD



  const uint8_t maskL6 = 0b11111110;
  const uint8_t mtchL6 = 0b11111100;
  const uint8_t maskL5 = 0b11111100;
  const uint8_t mtchL5 = 0b11111000;
  const uint8_t maskL4 = 0b11111000;
  const uint8_t mtchL4 = 0b11110000;
  const uint8_t chk1L4 = 0b00000111;
  const uint8_t maskL3 = 0b11110000;
  const uint8_t mtchL3 = 0b11100000;
  const uint8_t chk1L3 = 0b00001111;
  const uint8_t maskL2 = 0b11100000;
  const uint8_t mtchL2 = 0b11000000;
  const uint8_t chk1L2 = 0b00011111;
  const uint8_t maskFl = 0b11000000;
  const uint8_t mtchFl = 0b10000000;
  const uint8_t maskL1 = 0b10000000;
  const uint8_t mtchL1 = 0b00000000;

#define Is(X,c)  !((c & mask##X) ^ mtch##X)
#define IsNz(X,c)  (!!(c & chk1##X))

  bool h6 = Is(L6, c0);
  bool h5 = Is(L5, c0);
  bool h4 = Is(L4, c0);
  bool o4 = IsNz(L4, c0);
  bool i4 = Is(L4, c0) && Is(Fl, c1) && Is(Fl, c2) && Is(Fl, c3) && o4;
  bool h3 = Is(L3, c0);
  bool o3 = IsNz(L3, c0);
  bool i3 = Is(L3, c0) && Is(Fl, c1) && Is(Fl, c2) && o3;
  bool h2 = Is(L2, c0);
  bool o2 = IsNz(L2, c0);
  bool i2 = Is(L2, c0) && Is(Fl, c1) && o2;
  bool i1 = Is(L1, c0);
  bool hf = Is(Fl, c0);

#define V4_0(c)    (((UChar32) c &0b00000111 ) <<  (18))
#define V4_1(c)    (((UChar32) c &0b00111111 ) <<  (12))
#define V4_2(c)    (((UChar32) c &0b00111111 ) <<  (6))
#define V4_3(c)    (((UChar32) c &0b00111111 ) <<  (0))

#define V3_0(c)    (((UChar32) c &0b00001111 ) <<  (12))
#define V3_1(c)    (((UChar32) c &0b00111111 ) <<  (6))
#define V3_2(c)    (((UChar32) c &0b00111111 ) <<  (0))

#define V2_0(c)    (((UChar32) c &0b00011111 ) <<  (6))
#define V2_1(c)    (((UChar32) c &0b00111111 ) <<  (0))

#define V1_0(c)    (((UChar32) c &0b01111111 ) <<  (0))
  // icu varies in error handling
  // on "skipping" bytes, for e2 82 *41 fx with * beeing an unexpected non tail
  // it skips 2 chars (e2 82), re adjusting at 41 together
  //
  //
  bool err = (h2 && !i2) || (h3 && !i3) || (h4 && !i4) || hf || h5 || h6;
  flawed |= err;
  offsets__t advance = i1 * 1 + i2 * 2 + i3 * 3 + i4 * 4
    + err * 1
    + 1 * (Is(L3, c0) && Is(Fl, c1) && !Is(Fl, c2));
  r = i1 * c0
    + i2 * (V2_0(c0) + V2_1(c1))
    + i3 * (V3_0(c0) + V3_1(c1) + V3_2(c2))
    + i4 * (V4_0(c0) + V4_1(c1) + V4_2(c2) + V4_3(c3))
    + err * UC_REPL; // Neg = error;
  i += advance;
#endif
}


///////////////////////////////////////////////////////////////////////////////
// a kernel which use stack memory as a buffer for WRITING
// (tests also buffering input buffer had not significant effect)
// this kernel avoids branch switching
//
__global__ void convUTF8_UCHAR32_kernel2Stck(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed) {
  ////////////////////////////////////////////////////////////////////////////
  // (gridDim.x * blockDim.x)
  //assert(sizeof(int) == 4);
  //assert(sizeof(UChar32) == 4);
  const int strPerWarp = 1;
  int ix = (threadIdx.x + IMUL(blockIdx.x, blockDim.x)) * strPerWarp;
  //int ix = threadIdx.x * gridDim.x + blockIdx.x; //  IMUL(blockIdx.x, blockDim.x)) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStringN) {
    return;
  }
  const int WORST_PER_EL = 1;
  const int BUFLEN = 5 * 64;
#undef STACK_IN
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
  bool flaw = 0;
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
    uint8_t* srcBuffer = &src[start];
#endif

#ifdef VERBOSE
    printf(" t=%d b=%d /%d [%d,%d[  idx= %d  start-end[%d,%d] (%d) \n", (int)threadIdx.x, (int)blockIdx.x, (int)blockDim.x, (int)ix, (int)ixLast, nrString, start, end, length);
#endif
    uint8_t* s = srcBuffer;
    UChar32 c;
    offsets__t done = 0;
    int32_t i = 0;
    while (i < length) {
      convU8_NEXT_NOBRANCH(s, i, length, c, flaw); // advances i
      destBuffer[tgt++] = c;
      if (tgt % BUFLEN == 0) {
        for (int i = 0; i < BUFLEN; ++i) {
          dest[done+i] = destBuffer[done+i];
        }
        //dest -= BUFLEN;
        destBuffer -= BUFLEN;
      }
    }
    // no real harm or benefit __syncthreads();
#ifdef DEST_LEN
    destOffsets[nrString] = tgt /* + startOffset*/;
#else
    destOffsets[nrString] = tgt + startOffset;
#endif
    // memcpy(dest, destBuffer, sizeof(UChar32) * tgt);
    for (int i = done; i < tgt; ++i) {
      dest[done+i] = destBuffer[done+i];
    }
#ifdef VERBOSE
    printf(" t=%d b=%d  [%d,%d[  idx= %d  start-end[%d,%d] (%d) set dest[%d] = %d \n", (int)threadIdx.x, (int)blockIdx.x, (int)ix, (int)ixLast,
      nrString, start, end, length, nrString, tgt + startOffset);
#endif
  }
  if (flaw) {
    *flawed = 1;
  }
  return;
}


///////////////////////////////////////////////////////////////////////////////
// this kernel avoids branch switching
// not using an input of output buffer is superiour
__global__ void convUTF8_UCHAR32_kernel2NB(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed) {
  ////////////////////////////////////////////////////////////////////////////
  const UChar32 UC_REPL = 0xFFFD;
  const int strPerWarp = 1;
  int ix = (threadIdx.x + IMUL(blockIdx.x, blockDim.x)) * strPerWarp;
  int ixLast = ix + strPerWarp;
  if (ix >= nrStringN) {
    return;
  }
  const int WORST_PER_EL = 1;
  const int xBUFLEN = 5 * 64;
  int* destOffsets = dO + 1; // first one is 0!
  int startOffset = srcOffsets[ix] * WORST_PER_EL;
  UChar32* dest = &dest0[startOffset]; // worst case, every byte an UChar
  UChar32* destBuffer = dest;
  int32_t tgt = 0;
  bool flaw = 0;
  for (int nrString = ix; nrString < ixLast && nrString < nrStringN; ++nrString) {
    // the string spans from
    // the most naive implementation.
    int start = srcOffsets[nrString];
    int end = srcOffsets[nrString + 1];
    int length = end - start;
    uint8_t* srcBuffer = &src[start];
    uint8_t* s = srcBuffer;
    UChar32 c;
    int32_t i = 0;
    while (i < length) {
      convU8_NEXT_NOBRANCH(s, i, length, c, flaw); // advances i
      destBuffer[tgt++] = c;
    }
    // no real harm or benefit __syncthreads();
#ifdef DEST_LEN
    destOffsets[nrString] = tgt /* + startOffset*/;
#else
    destOffsets[nrString] = tgt + startOffset;
#endif
  }
  if (flaw) {
    *flawed = 1;
  }
  return;
}

const char *kernel_1 = "GPU_001_U8_NEXT_000_000";
#define NR 1
#define SINP 000
#define SOUT 000
#define NEXT_CHAR(s,i,length,c, fl) U8_NEXT(s, i, length, c); if (c < 0) { fl = 1; c = UC_REPL; }
#define STRPERTHREAD 001
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_001_U8_NEXT_000_000(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"

#undef NR
#undef NEXT_CHAR
#undef SINP
#undef SOUT
#undef SIGNATURE
#undef STRPERTHREAD
const char *kernel_2 = "GPU_001_U8_NEXT_000_256";
#define NR 2
#define SINP 000
#define SOUT 256
#define NEXT_CHAR(s,i,length,c, fl) U8_NEXT(s, i, length, c); if (c < 0) { fl = 1; c = UC_REPL; }
#define STRPERTHREAD 001
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_001_U8_NEXT_000_256(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"


const char *kernel_3 = "GPU_001_U8_NOBR_000_000";
#define NR 3
#define SINP 000
#define SOUT 000
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 001
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_001_U8_NOBR_000_000(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"

const char *kernel_4 = "GPU_001_U8_NOBR_000_256";
#define NR 4
#define SINP 000
#define SOUT 256
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 001
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_001_U8_NOBR_000_256(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"

const char* kernel_5 = "GPU_016_U8_NEXT_000_000";
#define NR 5
#define SINP 000
#define SOUT 000
#define NEXT_CHAR(s,i,length,c, fl) U8_NEXT(s, i, length, c); if (c < 0) { fl = 1; c = UC_REPL; }
#define STRPERTHREAD 016
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_016_U8_NEXT_000_000(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed, int strPerThread) {
#include "conv_kernel_expand.cuh"

const char* kernel_6 = "GPU_016_U8_NEXT_000_256";
#define NR 6
#define SINP 000
#define SOUT 256
#define NEXT_CHAR(s,i,length,c, fl) U8_NEXT(s, i, length, c); if (c < 0) { fl = 1; c = UC_REPL; }
#define STRPERTHREAD 016
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_016_U8_NEXT_000_256(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed, int strPerThread) {
#include "conv_kernel_expand.cuh"

const char* kernel_7 = "GPU_016_U8_NOBR_000_000";
#define NR 5
#define SINP 000
#define SOUT 000
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 016
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_016_U8_NOBR_000_000(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed, int strPerThread) {
#include "conv_kernel_expand.cuh"

const char* kernel_8 = "GPU_016_U8_NOBR_000_256";
#define NR 6
#define SINP 000
#define SOUT 256
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 016
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_016_U8_NOBR_000_256(uint8_t* src, int* srcOffsets, int nrStringN, UChar32* dest0, int* dO, int* flawed, int strPerThread) {
#include "conv_kernel_expand.cuh"
#define NO_WRITE 1
#define signature __global__ void convUTF8_UCHAR32_kernelGPU_016_U8_NOBR_000_256NW(uint8_t* src, int* srcOffsets, int nrStringN, int* dO, int* flawed, int strPerThread) {
#include "conv_kernel_expand.cuh"
#undef NO_WRITE

// buffer lengths
const char* kernel_9 = "GP2_001_U8_NOBR_000_016";
#define NR 15
#define SINP 000
#define SOUT 016
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 001
#undef NO_WRITE
#define NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_016NO(const uint8_t* src, const int* srcOffsets, int nrStringN, UChar32* dest0, const int* dO) {
#include "conv_kernel_expand.cuh"
#define NO_WRITE
#undef NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_016NW(const uint8_t* src, const int* srcOffsets, int nrStringN, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"
#undef NO_WRITE
#undef NO_OFFSET

// buffer lengths
const char* kernel_10 = "GP2_001_U8_NOBR_000_032";
#define NR 15
#define SINP 000
#define SOUT 032
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 001
#undef NO_WRITE
#define NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_032NO(const uint8_t* src, const int* srcOffsets, int nrStringN, UChar32* dest0, const int* dO) {
#include "conv_kernel_expand.cuh"
#define NO_WRITE
#undef NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_032NW(const uint8_t* src, const int* srcOffsets, int nrStringN, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"
#undef NO_WRITE
#undef NO_OFFSET
// buffer lengths
const char* kernel_11 = "GP2_001_U8_NOBR_000_064";
#define NR 15
#define SINP 000
#define SOUT 064
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 001
#undef NO_WRITE
#define NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_064NO(const uint8_t* src, const int* srcOffsets, int nrStringN, UChar32* dest0, const int* dO) {
#include "conv_kernel_expand.cuh"
#define NO_WRITE
#undef NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_064NW(const uint8_t* src, const int* srcOffsets, int nrStringN, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"
#undef NO_WRITE
#undef NO_OFFSET

const char* kernel_12 = "GP2_001_U8_NEXT_000_256";
#define NR 12
#define SINP 000
#define SOUT 256
#define NEXT_CHAR(s,i,length,c, fl) U8_NEXT(s, i, length, c); if (c < 0) { fl = 1; c = UC_REPL; }
#define STRPERTHREAD 001
#undef NO_WRITE
#define NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_256NO(const uint8_t* src, const int* srcOffsets, int nrStringN, UChar32* dest0, const int* dO) {
#include "conv_kernel_expand.cuh"
#define NO_WRITE
#undef NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_256NW(const uint8_t* src, const int* srcOffsets, int nrStringN, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"
#undef NO_WRITE
#undef NO_OFFSET

const char* kernel_13 = "GP2_001_U8_NEXT_000_128";
#define NR 13
#define SINP 000
#define SOUT 128
#define NEXT_CHAR(s,i,length,c, fl) U8_NEXT(s, i, length, c); if (c < 0) { fl = 1; c = UC_REPL; }
#define STRPERTHREAD 001
#undef NO_WRITE
#define NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_128NO(const uint8_t* src, const int* srcOffsets, int nrStringN, UChar32* dest0, const int* dO) {
#include "conv_kernel_expand.cuh"
#define NO_WRITE
#undef NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_128NW(const uint8_t* src, const int* srcOffsets, int nrStringN, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"
#undef NO_WRITE
#undef NO_OFFSET


const char* kernel_14 = "GP2_001_U8_NOBR_000_256";
#define NR 14
#define SINP 000
#define SOUT 256
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 001
#undef NO_WRITE
#define NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_256NO(const uint8_t* src, const int* srcOffsets, int nrStringN, UChar32* dest0, const int* dO) {
#include "conv_kernel_expand.cuh"
#define NO_WRITE
#undef NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_256NW(const uint8_t* src, const int* srcOffsets, int nrStringN, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"
#undef NO_WRITE
#undef NO_OFFSET


const char* kernel_15 = "GP2_001_U8_NOBR_000_128";
#define NR 15
#define SINP 000
#define SOUT 128
#define NEXT_CHAR(s,i,length,c, fl) convU8_NEXT_NOBRANCH(s, i, length, c, fl)
#define STRPERTHREAD 001
#undef NO_WRITE
#define NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_128NO(const uint8_t* src, const int* srcOffsets, int nrStringN, UChar32* dest0, const int* dO) {
#include "conv_kernel_expand.cuh"
#define NO_WRITE
#undef NO_OFFSET
#define signature __global__ void convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_128NW(const uint8_t* src, const int* srcOffsets, int nrStringN, int* dO, int* flawed) {
#include "conv_kernel_expand.cuh"
#undef NO_WRITE
#undef NO_OFFSET

// the shift is a special form of a scan.
// if we store length, not offsets in the target array, we can use a simple scan to produce the final array
// but we need the source offsets to copy the data.
//
// thrust::inclusive_scan(data, data + 6, data); // in-place scan

// parallel prefix sum kernel
// https://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf
//Reading � Mark Harris, Parallel Prefix Sum with CUDA
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


