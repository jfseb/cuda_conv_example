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
#include <thrust/device_ptr.h>
#include <iomanip >
#include <vector>
#include <random>
#include <regex>

#include <helper_functions.h>
#include <helper_cuda.h>



#define UChar32 int32_t
#include "../icu/source/common/unicode/utf8.h"

#include "types.h"

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

// GPU variant(s)
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

// 2*4*64,  ASS = 0, AA = 0, SL=1 NRBLOCKS 32 -> error cuda!
// 4*64,    ASS = 0, AA = 0, SL=1 NRBLOCKS 32 -> error cuda!

const int xAVG_LEN = /*4*2*2* */4*64; // 8x64 crashes!?
const int xNRSTRINGS = 64 * 256 * 8 * 100 / xAVG_LEN;
//const int NRSTRINGS = 64 * 256 * 8 *640 / AVG_LEN; with AVG_LEN = 3200 crash

// 4x64 | 8*64*256*8*100/AVG_LEN / 32  0 0 1 cudaMemCpy error!
#define NRBLOCK 32
//#define NRBLOCK 64

#define ALL_STRINGS_SAME 0
#define ALLASCII 0
#define SAME_LEN 1


const int range[] = { 1,100,200,300 }; // promille  [err, 4-byte, 3-byte, 2-byte  [remainder to 1000 = 1byte[
const int perc1_100_200_300[] = { range[0], range[0] + range[1], range[0] + range[1] + range[2], range[0] + range[1] + range[2] + range[3] };
const int percAllAscii[] = { 0, 0, 0, 0 };

// Forwards 

void genData(src_char_t* data, offsets__t dataLen, offsets__t* offsets, int nrStrings, offsets__t avgLen, 
  float plenVariation, const int* perc, bool allSame, bool simple,
  double& sum, double& squ, int& maxL);

class Output;
std::string compareOutputs(const Output& ref, const Output& cmp);


void* smalloc(int len) {
  void* a = malloc(len);
  if (a == 0) {
    std::cout << "oom";
    exit(-1);
  }
  return a;
}

class InputData {
public:
  const int nrStrings;
  const int avgLen;

  bool allAscii = false;
  float lenVar = 0.0;
  bool allSame = false;

  src_char_t* data = NULL;
  offsets__t* offs = 0;

  double squL;
  double sumL;
  int maxL;
  float ActualMean() const {
    return sumL / nrStrings;
  }
  float ActualStdDev() const {
    double nrS = nrStrings;
    // sqrt ( sum (xi - m)^2 / N ) = (sum(xi^2) - 2* sum(xi)*m + N*(m^2) /N =  1/N ( squL  - 2 * N*m *m + N*m*m ) = 1/N (sqlL - N*m*m )
    return sqrt((1.0 / (nrS)) * (squL - sumL * sumL /nrS));
  }

  InputData(int nrStrings, int avgLen) : nrStrings(nrStrings), avgLen(avgLen) {
    data = (src_char_t*) smalloc(nrStrings * avgLen);
    offs = (offsets__t*) smalloc(sizeof(offsets__t) * (nrStrings + 1));
  }
  void fill() {
    const int* perc = (allAscii) ? percAllAscii : perc1_100_200_300;
    genData(data, nrStrings * avgLen, offs, nrStrings, avgLen, lenVar, perc, allSame, /*simple*/(nrStrings < 100),
      sumL,squL, maxL);
  }
  ~InputData() {
    free(offs);
    free(data);
  }
  std::string descriptionHead() const {
    std::stringstream ss;
    // asccii vs 
    ss << " CT";
    ss << "," << "DST";
    ss << "," << std::setw(5) << "ALEN";
    ss << "," << std::setw(7) << "NRSTRING";
    ss << "," << std::setw(5) << "MB";
    ss << "," << std::setw(6) << "MEAN";
    ss << "," << std::setw(8) << "STDDEF";
    ss << "," << std::setw(6) << "MAXL";
    return ss.str();
  }
  std::string description() const {
    std::stringstream ss;
    // ascii vs
    ss << " " << std::string((allSame) ? ((allAscii) ? "AS" : "FS")
                               : ((allAscii) ? "AV" : "FV"));
    ss << "," << std::string(((allSame) ? "FIX" : ((lenVar == 0.0) ? 
      "FIX" : ((lenVar > 0.0) ? "BOX" : 
         (lenVar < -1.1) ? "GM1" : "PS1"))));
    ss << "," << std::setw(5) << avgLen;
    ss << "," << std::setw(7) << nrStrings;
    ss << "," << std::setw(5) << std::setw(6) << std::fixed << std::setprecision(0) << (int)(ActualMean()*nrStrings/ (1024.0*1024.0));
    ss << "," << std::setw(6) << std::fixed << std::setprecision(0) << ActualMean();
    ss << "," << std::setw(8) << std::fixed << std::setprecision(2) << ActualStdDev();
    ss << "," << std::setw(6) << maxL;
    return ss.str();
  }

  bool filterReject(std::string& dataFilter) {
    std::string filter = std::regex_replace(dataFilter, std::regex(" "), "");
    std::string tst = std::regex_replace(description(), std::regex(" "), "");
    if (filter.length() <= tst.length()) {
      std::cout << filter << std::endl;
      std::cout << tst << std::endl;
      return !std::equal(filter.begin(), filter.end(), tst.begin());
    }
    return filter.length() != 0;
  }

};




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

//const bool verbose = false;

class Output {
public:
  const InputData* input = NULL;
  dst_char_t* data = NULL;
  offsets__t* offs = NULL;
  int rc = -2;
  int grid_x = 0;
  int block_x = 0;
  std::string name; // name of the example
  float tm_proc = 0.0; // time to process
  float tm_coal = 0.0; // time to coalesce, if relevant
  float tm_ref = 0.0; // reference time

  std::string summaryHead() const {
    std::stringstream ss;
    ss << std::setw(23) << "NAME"
      << "," << NamePartsHead()
      << "," << std::setw(10) << "MbpsGPU"
      << "," << std::setw(8) << "MbpsCPU"
      << "," << std::setw(5) << "FAKT"
      << "," << std::setw(6) << "T_PROC"
      << "," << std::setw(6) << "T_COAL"
      << "," << std::setw(7) << "T_CPU"
      << "," << std::setw(6) << input->descriptionHead()
      << "," << std::setw(6) << "THRDS"
      << "," << "GRID* BLK";
    return ss.str();
  }

  std::string summary() const {
    std::stringstream ss;
    float mb = (input->avgLen * input->nrStrings)/(1024.0*1024.0);
    float mbpsCPU = (tm_ref == 0.0) ? 0.0 : (1000.0 * mb / tm_ref);
    float mbpsGPU = (tm_coal + tm_proc == 0.0) ? 0.0 : (1000.0 * mb / (tm_coal + tm_proc));
    float fakt = (mbpsCPU == 0.0) ? 0 : (mbpsGPU / mbpsCPU);
    ss << std::setw(18) << Name();
      ss << "," << NameParts()
      << "," << std::setw(10) << std::fixed << std::setprecision(2) << mbpsGPU
      << "," << std::setw(8) << std::fixed << std::setprecision(2) << mbpsCPU
      << "," << std::setw(5) << std::fixed << std::setprecision(1) << fakt
      << "," << std::setw(6) << std::fixed << std::setprecision(2) << tm_proc
      << "," << std::setw(6) << std::fixed << std::setprecision(2) << tm_coal
      << "," << std::setw(7) << std::fixed << std::setprecision(2) << tm_ref
      << "," << input->description()
      << "," << std::setw(6) << grid_x * block_x << "," << std::setw(6) << grid_x << "*" << std::setw(3) << block_x;
    return ss.str();
  }

  Output() {
    input = NULL;
    data = NULL;
    offs = NULL;
    tm_proc = 0.0;
    tm_coal = 0.0;
    rc = -2;
    grid_x = 0;
    block_x = 0;
    name = "uninitialized";
  }
  Output(const char* name, const InputData* inp) :
    input(inp)
    , name(name)
    , data(NULL)
    , offs(NULL)
  {
    tm_proc = 0.0;
    tm_coal = 0.0;
  }

  std::string Name() const {
    return name;
  }

  std::string NameParts() const {
    std::stringstream ss;
    if (Name().size() < 0) {
      return "0,0,0,0";
    }
    ss << Name().substr(0, 3);
    ss << ", " << Name().substr(5, 2);
    ss << ", " << Name().substr(11, 4);
    ss << ", " << Name().substr(11+4+1+3+1, 3);
    return ss.str();
  }

  std::string NamePartsHead() const {
    std::stringstream ss;
    ss << " ALG, S, METH,SOUT";
    return ss.str();
  }

  Output& operator =(Output&& other) {
    free(data);
    free(offs);
    data = other.data; other.data = NULL;
    offs = other.offs; other.offs = NULL;
    input = other.input;
    name = other.name;
    grid_x = other.grid_x;
    block_x = other.block_x;
    tm_proc = (tm_proc == 0.0) ? other.tm_proc : std::min(tm_proc, other.tm_proc);
    tm_coal = (tm_coal == 0.0) ? other.tm_coal : std::min(tm_coal, other.tm_coal);
    other.clear();
    return *this;
  }

  Output(Output&& other) {
    free(data);
    free(offs);
    data = other.data;
    other.data = NULL;
    offs = other.offs;
    other.offs = NULL;
    input = other.input;
    name = other.name;
    grid_x = other.grid_x;
    block_x = other.block_x;
    tm_proc = (tm_proc == 0.0) ? other.tm_proc : std::min(tm_proc, other.tm_proc);
    tm_coal = (tm_coal == 0.0) ? other.tm_coal : std::min(tm_coal, other.tm_coal);
    other.clear();
  };
  void clear() {
    offs = NULL;
    data = NULL;
    input = NULL;
    name.clear();
    tm_proc = 0.0;
    tm_coal = 0.0;
    grid_x = 0;
    block_x = 0;
  }

  ~Output() {
    free(data);
    free(offs);
    // not input!
  }

  std::string difference;
  /// <returns> true if equal ( no difference)</returns>
  bool compare(const Output& ref) {
    difference = compareOutputs(ref, *this);
    tm_ref = (tm_ref == 0.0) ? ref.tm_proc : std::min(tm_ref, ref.tm_proc);
    (0 != difference.length()) && std::cout << " comparing " << Name() << " vs." << ref.Name() << " " << difference << std::endl;
    return difference.length() == 0;
  }
  void setThreads(dim3& grid, dim3& block) {
    grid_x = grid.x;
    block_x = block.x;
  }
};

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
int addNextRandom(src_char_t* pos, int len, const int * perc) {
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


int genLen(int avgLen, float plenVariation, std::default_random_engine & generator, 
  std::poisson_distribution<int>& distribution,
  std::gamma_distribution<float>& gdist, int nrStrings) {
  if (plenVariation == 0.0) {
    return avgLen;
  }
  if (plenVariation > 0.0) {
    return (int)(avgLen + (plenVariation * (rand() % (2 * avgLen) - avgLen)));  //  (-avgLen ... *avgLen-1)*p
  }
  if (plenVariation < -1.0) {
    float flt = gdist(generator);
    if (flt < (float)(avgLen * nrStrings / 2)) {
      return (int)flt;
    }
    return avgLen * nrStrings / 20.0;
  }
  int r = distribution(generator);
  return r;
}

void genData(src_char_t* data, offsets__t dataLen, offsets__t* offsets, int nrStrings, offsets__t avgLen, float plenVariation, const int* perc,
  bool allSame, bool simple,
  double& sumL, double& squL, int& maxL) {
  printf("...generating input data in CPU mem. %f\n",         plenVariation);
  std::default_random_engine generator;
  double lambda = 1.0 / avgLen;
  std::poisson_distribution<int> distribution(1.0*avgLen);
  //alpha small = > flat;
  // stddef = alpha/beta ^2;
  float alpha0 = 1.02;
  float beta0 = avgLen * 1.0/alpha0;
  std::gamma_distribution<float> gdist(alpha0, beta0);
  assert(nrStrings > 0);
  assert(dataLen >= nrStrings * avgLen);
  offsets[0] = 0;
  maxL = 0;
  if (!simple) {
    srand(123);
    int base = 0;
    int tgt = base;
    for (int i = 0; i < nrStrings; ++i) {
      if (allSame) {
        srand(123);
      }
      int len = genLen(avgLen, plenVariation, generator, distribution, gdist, nrStrings); // (int)(avgLen + (plenVariation * (rand() % (2 * avgLen) - avgLen)));  //  (-avgLen ... *avgLen-1)*p
      while (tgt < base + len && tgt + 1 < nrStrings * avgLen) {
        int length = std::min(base + len - tgt, nrStrings * avgLen - 1 - tgt);
        tgt += addNextRandom(&data[tgt], length, perc);
      }
      offsets[i + 1] = tgt;
      base = tgt;
    }
  }
  else {
    int base = 0;
    int tgt = base;
    for (int i = 0; i < nrStrings; ++i) {
      int len = std::min((i % avgLen) + 5, avgLen);
      if (tgt + 10 < nrStrings * avgLen) {
        int filled = snprintf((char*)&data[tgt], 10, "%d", i);
        tgt += filled;
      }
      for (; tgt < base + len && tgt < nrStrings * avgLen; ++tgt) {
        int offs = tgt - base;
        data[tgt] = ((i % 2) == 0) ? 'a' + (offs % 27) : 'A' + (offs % 27);
      }
      offsets[i + 1] = tgt;
      base = tgt;
    }
    data[offsets[2] + 2] = 0xc3;
    data[offsets[2] + 3] = 0x84;
    data[offsets[4] + 3] = 0xc3;
    data[offsets[4] + 4] = 0x84;
  }

  sumL = 0.0;
  squL = 0.0;
  maxL = 0;
  for (int i = 0; i < nrStrings; ++i) {
    long len = offsets[i + 1] - offsets[i];
    maxL = std::max(len,(long) maxL);
    sumL += len;
    double lenf = len;
    squL += (lenf * lenf);
  }
}

template<typename T> void dumpStartT(T* data, int dlen, offsets__t* offs, int start, int end) {
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

/// compact an data / offsets pair with gaps

void compactString(offsets__t* srcOffsets, // original offsets,
                   offsets__t* spreadOffsets,
            /*out*/offsets__t* compactOffsets,
                   indexoff_t nrStrings,
                   dst_char_t* spreadData,
            /*out*/dst_char_t* compactData,
                   dim3 block, dim3 grid,
                   int stringsPerThread,
                   int verbose) {
  assert(block.y == 1 && block.z == 1);
  assert(grid.y == 1 && grid.z == 1);
  assert(block.x * grid.x *stringsPerThread >= nrStrings);
  assert(srcOffsets[0] == 0);
  // combine GPU results into contiguous string buffer
  indexoff_t last = 0;
  compactOffsets[0] = 0;
  for (int blk = 0; blk < grid.x; ++blk) {
    for (int treadidx = 0; treadidx < block.x; ++treadidx) {
      // we can combine the stringsPerThread as they are already contiguos
      indexoff_t ix = (blk * block.x + treadidx) * stringsPerThread;
      if (ix > nrStrings) {
        continue;
      }
      indexoff_t ixend =std::min(nrStrings, ix + stringsPerThread);
      (verbose > 0) && std::cout << " t=" << treadidx << " b=" << blk << " " << block.x << " " << grid.x << " indexes[" << ix << ", " << ixend << "[" << std::endl;
      offsets__t unshifted_dest = srcOffsets[ix] * WORST_PER_INPUT_EL;
      offsets__t delta = unshifted_dest - last;
      offsets__t end_dest = spreadOffsets[ixend];
      int len = end_dest - unshifted_dest;
      (verbose > 0) && std::cout << " delta : " << delta << " src [" << unshifted_dest << "- " << unshifted_dest + len << "[  => " << "[" << last << " - " << last + len << "[" << std::endl;
      assert(delta >= 0);
      dst_char_t* dst = &compactData[last];
      dst_char_t* src = &spreadData[unshifted_dest];
      memcpy((void*)dst, (void*)src, len * sizeof(dst_char_t));
      for (indexoff_t idxOff = ix; idxOff < ixend; ++idxOff) {
        compactOffsets[idxOff + 1] = spreadOffsets[idxOff + 1] - delta;
      }
      last += len;
    }
  }
}

// coalesce strings in spreadData 
// spanning
//  ix = (blk * block.x + treadidx) * stringsPerThread;
//  len = dstOffsets[ix + stringsPerThread] - dstOffsets[ix];
// [ srcOffsets[ix] * WORST_PER_EL , srcOffsets[ix] * WORST_PER_EL + len [
// to 
//  compactData[dstOffsets[ix]];
void compactString2(offsets__t* srcOffsets, // original offsets,
  offsets__t* dstOffsets,
  indexoff_t nrStrings,
  dst_char_t* spreadData,
  /*out*/dst_char_t* compactData,
  dim3 block, dim3 grid,
  int stringsPerThread) {
  assert(block.y == 1 && block.z == 1);
  assert(grid.y == 1 && grid.z == 1);
  assert(block.x * grid.x * stringsPerThread >= nrStrings);
  assert(srcOffsets[0] == 0);
  // combine GPU results into contiguous string buffer
  indexoff_t last = 0;
  assert(dstOffsets[0] == 0);
  for (int blk = 0; blk < grid.x; ++blk) {
    for (int treadidx = 0; treadidx < block.x; ++treadidx) {
      // we can combine the stringsPerThread as they are already contiguos
      indexoff_t ix = (blk * block.x + treadidx) * stringsPerThread;
      int ixLast = ix + stringsPerThread;
      if (ix > nrStrings) {
        continue;
      }
      if (ix >= nrStrings) {
        return;
      }
      const int WORST_PER_EL = 1;
      indexoff_t ixend = std::min(nrStrings, ix + stringsPerThread);
      offsets__t from = srcOffsets[ix] * WORST_PER_EL;
      offsets__t to = dstOffsets[ix];
      offsets__t len = dstOffsets[ixend] - to;
      dst_char_t* dst_start = compactData + to;
      dst_char_t* src = spreadData + from;
      memcpy((void*)dst_start, (void*)src, len * sizeof(dst_char_t));
    }
  }
}


void checkU8_NEXT() {
  const uint8_t s[]{
    'A', 0xF0,
    0x7F, 0xFF,
    0x80, 0x00, // smalles follow
    0xBF, 0xFF, // largest follow
    0xC0, 0x00, // non-minimal lwr
    0xC1, 0xFF, // non-minimal upr
     0xC2, 0x00, // no follow lwr
     0xC2, 0x7F, // no follow upr
    0xC2, 0x80, // --interleaved 
       // surrogate low D800-DB7F DC00-DFFF
       // surrogate upr
    0xE0, 0x80, // 3 
    0xEF, 0xAF, // 3 byte utc
    0xF0, 0x90, // largest 4 byte 
    0xF0, 0x91, // too large 
    0xF1, 0x00, // too large
    0xDF, 0x81, // -- 3 byte 
    0xE2, 0x82, 0xAC ,
    0xFC, 0xA0, 0xA1, 0xA1,
    0xC1, 0x80, // non-minimal
    0x80, // single follow
    0xE2, 'A',             // err 3-1
    0xE0, 0x80, 0x00, 'A', // invalid non-minimal, skip only 1
    0xE0, 0xA0, 0x00, 'A',  // skip 2   (11 1 1 1)
    0xE0, 0xA0, 0x80, 'A', // valid 
    0xE1, 0x80, 0x00, 'A', //  skip 2   (00 1 1 1)  0/1/0 | 1/1/0
    0xE2, 0x82, 'A',       // err 3-2  skip 2
    0xE2, 0xE2, 0x82, 0xAD,
    0xE8, 0x80, 0x00, 'A', // skip2     (10,1,1,1)  0/1/0 | 1/1/0
    0xED, 0xA0, 0x00, 'A', // err skip 1 (01 1,1,1) 
    0xFC, 0xA0, 'B',  'A', // err-4-2
    0xFC, 0xA0, 0xA1, 'C', // err-4-3
    0xEF, 0xA0, 0x80, 'A', // ok f800
    0xF0, 0x82, 0x82, 0xAC, // overlong 3-byte
    0xF0, 0x82, 0x82, 0xAC, // overlong 3-byte
    0xF0, 0x88, 0x98, 0x80, // non-minimal
    0xf0, 0xa0, 0x80, 0x84, // ok
    0xf0, 0x90, 0x80, 0x00,
    0xf0, 0x90, 0x00, 'A', // skip 2
    0xF2, 0x9F, 0x98, 0x80,
    0xF4, 0x9F, 0x98, 0x80, // 11F600
    0xf4, 0x9f, 0x98, 0x80, // overflow
    0b11100000, 0b10111111, 
    0b10111111, // minimal!   (zero & leadc2) || (nonzero
    0b11000001, 0b10000000, // non-minimal! -> error, same as 0x01000000

    0xf0, 0x9f, 0x98, 0x80, // this is *not* non-minimal!
    0xf4, 0x9f, 0x98, 0x80,
    0x00, 0x00, 0x00, 0x00
  };
  // three 0x00 indicate end
  int len = 0;
  for (; (s[len] != 0) || (s[len + 1] != 0) || (s[len + 2] != 0); ++len) {
    //std::cout << (int) s[len] << (int) s[len + 1] << " " << (int) s[len + 2] << " " << len << std::endl;
  }
  std::cout << "len=" << len << std::endl;
  for (int i = 0; i < len; ++i) {
    int i1 = i;
    UChar32 c1 = 0;
    U8_NEXT(s, i1, len, c1);
    int i2 = i;
    UChar32 c2 = 0;
    U8_NEXT_NB_V2(s, i2, len, c2);
    if (i1 != i2 || c2 != c1) {
      std::ios_base::fmtflags fmtflags(std::cout.flags());
      {
        std::cout << "Processing ..." << std::endl;
        int i3 = i;
        UChar32 c3 = 0;
        //U8_NEXT_NB_V(s, i3, len, c3);
        U8_NEXT_NB_V2_V(s, i3, len, c3);
      }
      int b0 = s[i]; int b1 = s[i + 1]; int b2 = s[i + 2]; int b3 = s[i + 3];
      std::cout << "diff for " << std::hex << b0 << " " << b1 << " " << b2 << " " << b3 << std::endl;
      std::cout << std::resetiosflags(fmtflags);
      std::cout << "U8_NEXT " << (i1-i) << " " << std::hex << c1 << std::endl;
      std::cout << std::resetiosflags(fmtflags);
      std::cout << "U8_NGXT " << (i2-i) << " " << std::hex << c2 << std::endl;
    }
  }
  std::cout.flush();
  bool full_verify = true;
  if (full_verify) {
    // there are only 0x10FFFF million uc characters (and 256^4 4 byte sequences) just test them all :-)
    // 
    uint8_t sall[] { 'A', 'B', 'C', 'D' };
    for (int b0 = 0x7F; b0 <= 0xFF; ++b0) {
      int err = 0;
      for (int b1 = 0; b1 <= 0xFF; ++b1) {
        for (int b2 = 0; b2 <= 0xFF; ++b2) {
          for (int b3 = 0; b3 <= 0xFF; ++b3) {
            len = 4;
            sall[0] = b0;
            sall[1] = b1;
            sall[2] = b2;
            sall[3] = b3;
            int i1 = 0;
            UChar32 c1 = 0;
            U8_NEXT(sall, i1, len, c1);
            int i2 = 0;
            UChar32 c2 = 0;
            U8_NEXT_NB_V2(sall, i2, len, c2);
            if ( i1 != i2 || c2 != c1) {
              ++err; 
              i2 = 0;
              c2 = 0;
              std::ios_base::fmtflags fmtflags(std::cout.flags());
              U8_NEXT_NB_V2_V(sall, i2, len, c2);
              std::cout << "diff for " << std::hex << b0 << " " <<  b1 << " " << b2 << " " << b3 << std::endl;
              std::cout << std::resetiosflags(fmtflags);
              std::cout << "U8_NEXT " << i1 << " " << std::hex << c1 << std::endl;
              std::cout << std::resetiosflags(fmtflags);
              std::cout << "U8_NGXT " << i2 << " " << std::hex << c2 << std::endl;
            }
            if (err > 10)
              exit(-1);
          }
        }
      }
    }
  }
}

Output convCPU(const InputData& input, int verbose, StopWatchInterface* hTimer) {
  // srcdata
  src_char_t* srcData = input.data;
  indexoff_t* srcOffs = input.offs;
  // dstData
  Output res("CPU", &input);

  const int nrStrings = input.nrStrings;
  const int avgLen = input.avgLen;

  dst_char_t* dstData = (dst_char_t*)smalloc(sizeof(dst_char_t) * nrStrings * avgLen);
  offsets__t* dstOffs = (offsets__t*)smalloc(sizeof(offsets__t) * (nrStrings + 1));
  if (verbose) {
    dumpStart(srcData, nrStrings * avgLen, srcOffs, 0, 3);
    dumpStart(srcData, nrStrings * avgLen, srcOffs, nrStrings - 3, nrStrings);
  }
  (verbose > 0) && printf("..running CPU conversion\n");
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  res.rc = convChunkUTF8_UCHAR32(srcData, srcOffs, nrStrings, dstData, dstOffs);
  sdkStopTimer(&hTimer);
  res.tm_proc = sdkGetTimerValue(&hTimer);
  res.data = dstData;
  res.offs = dstOffs;
  (verbose > 0) && printf("CPU time: %f msecs.\n", res.tm_proc);
  if (verbose > 1) {
    dumpStartD(dstData, nrStrings * avgLen, dstOffs, 0, 6);
    std::cout << std::endl;
    dumpStartD(dstData, nrStrings * avgLen, dstOffs, nrStrings - 3, nrStrings);
    std::cout << std::endl;
  }
  return std::move(res);
}

/// <returns> 0-length string if equal, otherwise  "1,0,false" or similar</returns>
std::string compareOutputs(const Output& ref, const Output& cmp) {
  int nrStrings = ref.input->nrStrings;
  // Calculate max absolute difference and L1 distance
  // between CPU and GPU results
  int nr_offsets_diff = 0;
  int nr_char_diff = 0;
  int first_offset_diff_idx = -1;
  int first_offset_char_diff = -1; 
  for (int i = 0; i < nrStrings + 1; i++) {
    if (cmp.offs[i] != ref.offs[i]) {
      if (first_offset_diff_idx == -1) {
        first_offset_diff_idx = i;
      }
      ++nr_offsets_diff;
    }
  }
  if (nr_offsets_diff == 0) {
    for (int i = 0; i < nrStrings; i++) {
      indexoff_t len = ref.offs[i + 1] - ref.offs[i];
      for (indexoff_t pos = 0; pos < len; ++pos) {
        if (cmp.data[pos] != ref.data[pos]) {
          if (first_offset_char_diff == -1) {
            first_offset_char_diff = pos;
          }
          ++nr_char_diff;
        }
      }
    }
  }
  if ((nr_offsets_diff) > 0 || (nr_char_diff > 0) || (ref.rc != cmp.rc)) {
    std::cout << "difference !!" << std::endl;
    if (ref.rc != cmp.rc) {
      std::cout << "  rc diff cmp=" << cmp.rc << " ref=" << ref.rc << cmp.Name() << " vs " << ref.Name() << std::endl;
    }
    if (nr_offsets_diff > 0) {
      std::cout << "  offsets " << nr_offsets_diff << " offs[" << first_offset_diff_idx << "] cmp=" << cmp.offs[first_offset_diff_idx] << " ref=" << ref.offs[first_offset_diff_idx] << ";" << ref.rc << " vs " << cmp.rc << cmp.Name() << " vs " << ref.Name() << std::endl;

      int start = first_offset_diff_idx - 1;
      std::cout << " cmp " << std::endl;
      dumpStartT(cmp.data, cmp.input->nrStrings * cmp.input->avgLen, cmp.offs, start, first_offset_diff_idx);
      std::cout << " ref " << std::endl;
      dumpStartT(ref.data, ref.input->nrStrings * ref.input->avgLen, ref.offs, start, first_offset_diff_idx);
      // source 
      std::cout << "src" << std::endl;
      dumpStartT(cmp.input->data, cmp.input->nrStrings * cmp.input->avgLen, cmp.input->offs, start, first_offset_diff_idx);
      std::cout << std::endl;
    }
    if (nr_char_diff > 0) {
      std::cout << "  data " << nr_char_diff << "  data[" << first_offset_char_diff << "] " << cmp.data[first_offset_char_diff] << " " << ref.data[first_offset_char_diff] << ";" << ref.rc << " vs " << cmp.rc << cmp.Name() << " vs " << ref.Name() << std::endl;
    }
  }
  std::stringstream ss;
  if ((nr_offsets_diff > 0) || (nr_char_diff > 0) || (ref.rc != cmp.rc)) {
    ss << nr_offsets_diff << "," << nr_char_diff << "," << (ref.rc != cmp.rc);
  }
  return ss.str();
}

Output convGPU(const InputData& input, int verbose, StopWatchInterface* hTimer, int nrBlocks, int kernelNr) {
  /// CONV_GPU 

  Output res((const char*)"GPU1", &input);
  const int nrStrings = input.nrStrings;
  const int avgLen = input.avgLen;

  res.data = (dst_char_t*)smalloc(sizeof(dst_char_t) * nrStrings * avgLen);
  res.offs = (offsets__t*)smalloc(sizeof(offsets__t) * (nrStrings + 1));

  int* h_flawed_GPU = (int*)smalloc(sizeof(int) * 1);

  (verbose > 3) && printf("...allocating GPU memory.\n");

  uint8_t* d_srcData;
  UChar32* d_dstData;
  offsets__t* d_srcOffs;
  offsets__t* d_dstOffs;
  int* d_flawed;
  offsets__t* d_dstOffsC;
  UChar32* d_dstDataC;

  checkCudaErrors(cudaMalloc((void**)&d_srcData, sizeof(src_char_t) * nrStrings * avgLen));
  checkCudaErrors(cudaMalloc((void**)&d_srcOffs, sizeof(offsets__t) * (nrStrings + 1)));
  // result of kernel1
  checkCudaErrors(cudaMalloc((void**)&d_dstData, sizeof(dst_char_t) * nrStrings * avgLen));
  checkCudaErrors(cudaMalloc((void**)&d_dstOffs, sizeof(offsets__t) * (nrStrings + 1)));
  checkCudaErrors(cudaMalloc((void**)&d_flawed, sizeof(int)));


  checkCudaErrors(cudaMalloc((void**)&d_dstDataC, sizeof(dst_char_t) * nrStrings * avgLen));
  checkCudaErrors(cudaMalloc((void**)&d_dstOffsC, sizeof(offsets__t) * (nrStrings + 1)));
 

  (verbose > 3) && printf("...copying input data to GPU mem.\n");


  if (verbose > 1) {
    for (int i = 0; i < nrStrings+1; ++i) {
      if (input.offs[i] >= nrStrings * avgLen) {
        std::cout << "offset is too large @i" << i << " " << input.offs[i] << " " << nrStrings * avgLen << std::endl;
      }
    }
    for (int i = 0; i < nrStrings ; ++i) {
      if (input.offs[i] > input.offs[i+1]) {
        std::cout << "offsets not monotonous @i" << i << " "  << input.offs[i] << " " << input.offs[i + 1] << std::endl;
      }
    }
  }

  checkCudaErrors(cudaMemcpy(d_srcData, input.data, sizeof(src_char_t) * nrStrings * avgLen, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_srcOffs, input.offs, sizeof(indexoff_t) * (nrStrings + 1), cudaMemcpyHostToDevice));
  // clear flawed
  int flawed = 0;
  checkCudaErrors(cudaMemcpy(d_flawed, &flawed, sizeof(int), cudaMemcpyHostToDevice));

  (verbose > 3) && printf("Data init done.\n");

  (verbose > 0 ) && printf("Executing GPU kernel %d ...\n", kernelNr);
  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  // model1, each warp processes exactly one string.
  dim3 block(nrBlocks, 1);
  dim3 grid(((nrStrings + block.x) - 1) / block.x, 1);
  int stringsPerThread = 1;

  switch (kernelNr) {
  case 1:
    res.name = std::string(kernel_1);
    convUTF8_UCHAR32_kernelGPU_001_U8_NEXT_000_000 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_001_U8_NEXT_000_000() execution failed\n");
    break;
  case 2:
    res.name = std::string(kernel_2);
    convUTF8_UCHAR32_kernelGPU_001_U8_NEXT_000_256 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_001_U8_NEXT_000_256() execution failed\n");
    break;
  case 3:
    res.name = std::string(kernel_3);
    convUTF8_UCHAR32_kernelGPU_001_U8_NOBR_000_000 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_001_U8_NOBR_000_000() execution failed\n");
    break;
  case 4:
    res.name = std::string(kernel_4);
    convUTF8_UCHAR32_kernelGPU_001_U8_NOBR_000_256 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_001_U8_NOBR_000_256() execution failed\n");
    break;
  case 5:
    res.name = std::string(kernel_5);
    grid.x = ((nrStrings / 16 + block.x) - 1) / block.x;
    stringsPerThread = 16;
    convUTF8_UCHAR32_kernelGPU_016_U8_NEXT_000_000 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed, 16);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_016_U8_NEXT_000_000() execution failed\n");
    break;
  case 6:
    res.name = std::string(kernel_6);
    grid.x = ((nrStrings / 16 + block.x) - 1) / block.x;
    stringsPerThread = 16;

    convUTF8_UCHAR32_kernelGPU_016_U8_NEXT_000_256 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed, 16);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_016_U8_NEXT_000_256() execution failed\n");
    break;
  case 7:
    res.name = std::string(kernel_7);
    grid.x = ((nrStrings/16 + block.x) - 1) / block.x;
    stringsPerThread = 16;
    convUTF8_UCHAR32_kernelGPU_016_U8_NOBR_000_000 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed,16);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_016_U8_NOBR_000_000() execution failed\n");
    break;
  case 8:
    res.name = std::string(kernel_8);
    grid.x = ((nrStrings / 16 + block.x) - 1) / block.x;
    stringsPerThread = 16;
    convUTF8_UCHAR32_kernelGPU_016_U8_NOBR_000_256 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed, 16);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_016_U8_NOBR_000_256() execution failed\n");
    break;
  case 9:
    res.name = std::string(kernel_9);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_000NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_000NW() execution failed\n");
    break;
  case 10:
    res.name = std::string(kernel_10);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_004NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_004NW() execution failed\n");
    break;
  case 11:
    res.name = std::string(kernel_11);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_008NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_008NW() execution failed\n");
    break;
  case 12:
    res.name = std::string(kernel_12);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_016NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_016NW() execution failed\n");
    break;
  case 13:
    res.name = std::string(kernel_13);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_032NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_032NW() execution failed\n");
    break;
  case 14:
    res.name = std::string(kernel_14);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_064NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_064NW() execution failed\n");
    break;
  case 15:
    res.name = std::string(kernel_15);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_128NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_256NW() execution failed\n");
   break;
  case 16:
    res.name = std::string(kernel_16);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_256NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_256NW() execution failed\n");
    break;
    break;
  case 17:
    res.name = std::string(kernel_17);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_256NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_256NW() execution failed\n");
    break;
  case 18:
    res.name = std::string(kernel_18);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_128NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_256NW() execution failed\n");
    break;
  case 19:
    res.name = std::string(kernel_19);
    // produces only the offset array
    convUTF8_UCHAR32_kernelGP2_001_U8_NOB2_000_032NW << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGP2_001_U8_NOB2_000_032NW() execution failed\n");
    break;
  default:
    res.name = std::string(kernel_4);
    convUTF8_UCHAR32_kernelGPU_001_U8_NOBR_000_256 << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffs, d_flawed);
    getLastCudaError("convUTF8_UCHAR32_kernelGPU_001_U8_NOBR_000_256() execution failed\n");
  }
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  res.tm_proc = sdkGetTimerValue(&hTimer);
  (verbose > 0) && printf("GPU time: %f msecs.\n", res.tm_proc);
  res.setThreads(grid, block);
  sdkResetTimer(&hTimer);

  // coalescing on the device with this kernel is extremely slow
  // also thrust on the device ptr appears slower than on the host pointer !? 

#define DEVICE_COALESCE 0

  if (kernelNr >= 9 
    && kernelNr <= 19)
  {
    thrust::plus<int> binary_op;
    thrust::device_ptr<offsets__t> dptr_dstOffs(d_dstOffs);
    thrust::device_ptr<offsets__t> dptr_dstOffsEnd(d_dstOffs + nrStrings + 1);
    thrust::device_ptr<offsets__t> dptr_OffsC(d_dstOffsC);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    if (verbose > 1) {
      checkCudaErrors(cudaMemcpy(res.offs, d_dstOffsC, sizeof(int) * (nrStrings + 1), cudaMemcpyDeviceToHost));
      std::cout << "len only " << std::endl;
      dumpStartT(input.data, nrStrings * avgLen, res.offs, 0, 6);
      std::cout << std::endl;
      std::cout << std::endl;
      dumpStartT(input.data, nrStrings * avgLen, res.offs, nrStrings - 6, nrStrings);
      std::cout << std::endl;
    }
    // cumulative sum on a device ptr
    // 0 1 0 4  7 ... =>
    // 0 1 1 5 12 ...
    thrust::inclusive_scan(dptr_dstOffs, dptr_dstOffsEnd, dptr_OffsC, binary_op);
    float tm_ms_cum = 0.0;
    sdkStopTimer(&hTimer);
    {
      tm_ms_cum = sdkGetTimerValue(&hTimer);
      (verbose > 0) && printf("incl_scan GPU time: cum %f msecs.\n", tm_ms_cum);
    }
    getLastCudaError("memoryCoalesce() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    offsets__t* d_dstOffsC2 = dptr_OffsC.get();
    if (verbose > 1) {
      checkCudaErrors(cudaMemcpy(res.offs, d_dstOffsC2, sizeof(int)* (nrStrings + 1), cudaMemcpyDeviceToHost));
      std::cout << "len cum only " << std::endl;
      dumpStartT(input.data, nrStrings * avgLen, res.offs, 0, 6);
      std::cout << std::endl;
      std::cout << std::endl;
      dumpStartT(input.data, nrStrings * avgLen, res.offs, nrStrings - 6, nrStrings);
      std::cout << std::endl;
    }
    switch (kernelNr) {
    case  9:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_000NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case  10:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_004NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case  11:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_008NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case 12:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_016NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case 13:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_032NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case 14:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_064NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case 15:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_128NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case 16:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOBR_000_256NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case 17:
      convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_256NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case 18:
      convUTF8_UCHAR32_kernelGP2_001_U8_NEXT_000_128NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    case 19:
      convUTF8_UCHAR32_kernelGP2_001_U8_NOB2_000_032NO << <grid, block >> > (d_srcData, d_srcOffs, nrStrings, d_dstData, d_dstOffsC2);
      break;
    default:
      exit(-1);
    }
    //memoryCoalesce << < grid, block >> > (d_dstData, d_srcOffs, nrStrings, d_dstDataC, d_dstOffsC2);
    getLastCudaError("memoryCoalesce() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    //#include <xstddef>
    sdkStopTimer(&hTimer);
    res.tm_coal = sdkGetTimerValue(&hTimer) + tm_ms_cum;
    (verbose > 0 ) && printf("Coalesce GPU time+cum %f msecs. (%s)\n", res.tm_coal, res.name.c_str());

    (verbose > 0) && printf("Reading back GPU result...\n");
    memset(res.offs, 0, sizeof(offsets__t) * (nrStrings + 1));
    // Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(res.data, d_dstData, sizeof(UChar32) * nrStrings * avgLen, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(res.offs, d_dstOffsC2, sizeof(int) * (nrStrings + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_flawed_GPU, d_flawed, sizeof(int) * 1, cudaMemcpyDeviceToHost));

   
  } else if (DEVICE_COALESCE) {
    thrust::plus<int> binary_op;
    thrust::device_ptr<offsets__t> dptr_dstOffs(d_dstOffs);
    thrust::device_ptr<offsets__t> dptr_dstOffsEnd(d_dstOffs + nrStrings + 1);
    thrust::device_ptr<offsets__t> dptr_OffsC(d_dstOffsC);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    // cumulative sum on a device ptr
    // 0 1 0 4  7 ... =>
    // 0 1 1 5 12 ...
    thrust::inclusive_scan(dptr_dstOffs, dptr_dstOffsEnd, dptr_OffsC, binary_op);
    sdkStopTimer(&hTimer);
    {
      float tm_ms_cum = sdkGetTimerValue(&hTimer);
      (verbose > 0) && printf("incl_scan GPU time: cum %f msecs.\n", tm_ms_cum);
    }
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    offsets__t* d_dstOffsC2 = dptr_OffsC.get();
    memoryCoalesce << < grid, block >> > (d_dstData, d_srcOffs, nrStrings, d_dstDataC, d_dstOffsC2);
    getLastCudaError("memoryCoalesce() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    //#include <xstddef>
    std::swap(d_dstData, d_dstDataC);
    sdkStopTimer(&hTimer);
    float tm_ms_cum = sdkGetTimerValue(&hTimer);
    (verbose > 0) && printf("Coalesce GPU time: cum %f msecs.\n", tm_ms_cum);

    (verbose > 0) && printf("Reading back GPU result...\n");
    memset(res.offs, 0, sizeof(offsets__t) * (nrStrings + 1));
    // Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(res.data, d_dstData, sizeof(UChar32) * nrStrings * avgLen, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(res.offs, d_dstOffsC, sizeof(int) * (nrStrings + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_flawed_GPU, d_flawed, sizeof(int) * 1, cudaMemcpyDeviceToHost));

  }
  else {
    (verbose > 0) && printf("Reading back GPU result...\n");
    memset(res.offs, 0, sizeof(offsets__t) * (nrStrings + 1));
    // Read back GPU results to compare them to CPU results
    offsets__t* h_dstLen_GPU = (offsets__t*)smalloc(sizeof(offsets__t) * (nrStrings + 1));
    dst_char_t* h_dstDataGap_GPU = (dst_char_t*)smalloc(sizeof(dst_char_t) * nrStrings * avgLen);
    checkCudaErrors(cudaMemcpy(h_dstDataGap_GPU, d_dstData, sizeof(UChar32) * nrStrings * avgLen, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_dstLen_GPU, d_dstOffs, sizeof(int) * (nrStrings + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_flawed_GPU, d_flawed, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    res.rc = *h_flawed_GPU;
    if (res.rc < 0) {
      return std::move(res);
    }
    res.offs[0] = 0;
    offsets__t* h_dstOffsNoGap_GPU = (offsets__t*)smalloc(sizeof(offsets__t) * (nrStrings + 1));
    /* sum up length into offsets */
    {
      thrust::plus<int> binary_op;
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
      thrust::inclusive_scan(h_dstLen_GPU, h_dstLen_GPU + nrStrings + 1, res.offs, binary_op);
      sdkStopTimer(&hTimer);
      float tm_ms_cum = sdkGetTimerValue(&hTimer);
      (verbose > 0) && printf("GPU time: cum %f msecs.\n", tm_ms_cum);
    }

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    compactString2(input.offs, res.offs, nrStrings, h_dstDataGap_GPU, res.data, block, grid, stringsPerThread);
    sdkStopTimer(&hTimer);
    res.tm_coal = sdkGetTimerValue(&hTimer);
    (verbose > 0) && printf("CPU time:compact %f msecs.\n", res.tm_coal);

    free(h_dstLen_GPU);
    free(h_dstDataGap_GPU);
  }
  res.rc = *h_flawed_GPU;
  if (res.rc < 0) {
    return std::move(res);
  }
  res.offs[0] = 0;
  // done here.. 
  if (verbose > 1) {
    dumpStartT(res.data, nrStrings * avgLen, res.offs, 0, 6);
    std::cout << std::endl;
    std::cout << std::endl;
    dumpStartT(res.data, nrStrings * avgLen, res.offs, nrStrings - 6, nrStrings);
    std::cout << std::endl;
  }

  (verbose > 0) && printf("free GPU memory...\n");

  // free everything else 
  checkCudaErrors(cudaFree(d_flawed));
  checkCudaErrors(cudaFree(d_dstOffs));
  checkCudaErrors(cudaFree(d_dstOffsC));
  checkCudaErrors(cudaFree(d_dstData));
  checkCudaErrors(cudaFree(d_dstDataC));

  checkCudaErrors(cudaFree(d_srcOffs));
  checkCudaErrors(cudaFree(d_srcData));

  free(h_flawed_GPU);
  return std::move(res);
}

void printDist() {
  std::default_random_engine generator;
  int avgLen = 100;
  // variance = beta^2*alpha;
  // mean = alpha * beta
  float alpha0 = 1.1;
  float beta0 = avgLen*1.0/alpha0;
  std::gamma_distribution<float> gdist(alpha0, beta0);
  for (int i = 0; i < 100; ++i) {
    float val = gdist(generator);
    std::cout << val << " " << (int) val << std::endl;
  }
}

void printHistogram(offsets__t* offs, int nrStrings) {
  float hst[31];
  for (int i = 0; i < 31; ++i) {
    hst[i] = 0;
  }
  for (int i = 0; i < nrStrings; ++i) {
    int len = offs[i + 1] - offs[i];
    int bucket = 0;
    while (len > 0) {
      ++bucket;
      len >>= 1;
    }
    hst[bucket]++;
  }
  for (int i = 0; i < 31; ++i) {
    hst[i] /= 1.0*nrStrings;
  }
  for (int i = 0; i < 31; ++i) {
    std::cout << "[" << (0x1 << i) << "-" << (0x2 << i) << "]  " << hst[i] << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  int verbose = 0;
  assert(sizeof(int) == 4);
  checkU8_NEXT();
  StopWatchInterface* hTimer = NULL;
  printf("%s Starting...\n\n", argv[0]);
  printDist();
  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char**)argv);
  std::string dataFilter;
  for (int i = 0; i < argc - 1; ++i) {
    if (0 == strcmp(argv[i], "-str")) {
      dataFilter = std::string(argv[i + 1]);
      std::cout << " dataFilter:" << dataFilter << std::endl;
    }
  }
  sdkCreateTimer(&hTimer);

  (verbose > 3) && printf("Initializing data...\n");
  (verbose > 3) && printf("...allocating CPU memory.\n");

  if (dataFilter.length() == 0) {
    // create a small input data set
    InputData inputSimple(64, 32);
    inputSimple.fill();
    std::cout << " simple " << inputSimple.ActualMean() << " " << inputSimple.ActualStdDev() << std::endl;

#define NR_KERNELS 20
    for (int krn = 0; krn < NR_KERNELS; ++krn)
    {
      Output refCPU = convCPU(inputSimple, true, hTimer);
      Output resGPU = convGPU(inputSimple, true, hTimer, 2, krn);
      resGPU.compare(refCPU);
    }

#define NR_INP  1
    // test with a small size (sanity check)
    for (int krn = 0; krn < NR_KERNELS; ++krn) {
      Output ref = convCPU(inputSimple, false, hTimer);
      Output ref1 = convGPU(inputSimple, false, hTimer, 2, krn);
      if (!ref1.compare(ref)) {
        // run again in verbose mode ...
        std::cout << " diff in kernel " << ref1.name << std::endl;
        Output refCPU = convCPU(inputSimple, true, hTimer);
        Output resGPU = convGPU(inputSimple, true, hTimer, 2, krn);
        resGPU.compare(refCPU);
        exit(-1);
      }
      std::cout << ref1.summaryHead() << std::endl;
      std::cout << ref1.summary() << std::endl;
    }
  }
  else {
    std::cout << "Data Filter, no sanity check" << std::endl;
  }
  bool once = false;
  std::vector<InputData*> largeInputs;
  bool lean = true;

  for(int k = 0; k < 2; ++k)
  for (int fill = 0; fill < 4+1; ++fill) {
    if (lean && (fill != 2 && fill != 0)) {
      continue;
    }
    for (int avgLenB = 2 << 13; avgLenB > 2; avgLenB >>= 1) {
      if (lean && (avgLenB != 16 && avgLenB != 256 && avgLenB != 8192)) {
        continue;
      }
      for (int i = 0; i < 2; ++i) {
        if (lean && (i == 1)) {
          continue;
        }
        int avgLen = avgLenB + i * ((avgLenB / 2) + 1);
        //printf("...generating input data in CPU mem # %d. %d %d \n", (int) largeInputs.size(), avgLen, fill);
        const int nrStrings = 64 * 256 * 8 * 100 / avgLen;
        InputData* largeInput = new InputData(nrStrings, avgLen);
        largeInput->allAscii = !!k; //  ALLASCII;
        largeInput->allSame = (fill > 3); // ALL_STRINGS_SAME;
        switch(fill % 4 ) {
        case 0: largeInput->lenVar = 0.0; break;
        case 1: largeInput->lenVar = 1.0; break;
        case 2: largeInput->lenVar = -2.0; break; // gamma
        case 3: largeInput->lenVar = -1.0; break; // poisson
        }
        if (largeInput->filterReject(dataFilter)) {
          std::cout << " rejecting " << largeInput->description() << std::endl;
          delete largeInput;
        }
        else {
          largeInput->fill();
          if (!once) {
            std::cout << " filling  " << std::setw(3) << i << " " << largeInput->descriptionHead() << std::endl;
            once = true;
          }
          std::cout << " filling  " << std::setw(3) << i << " " << largeInput->description() << std::endl;
          if (dataFilter.length() > 0) {
            printHistogram(largeInput->offs, largeInput->nrStrings);
          }
          largeInputs.push_back(largeInput);
        }
      }
    }
  }


#define NR_RUN 25

  verbose = (dataFilter.length() > 0) ? 3 : 0;

  std::ofstream myfile;
  myfile.open((dataFilter.length()== 0) ? "../../../output/result.csv" : "../../../output/result.df.csv");  
  once = false;

  for (int i = 0; i < largeInputs.size(); ++i) {
    Output rx[NR_KERNELS];
    Output r0;
    // take min of NR_RUN
    InputData* id = largeInputs[i];
    std::cout << i << "/" << largeInputs.size() << " " << id->descriptionHead() << std::endl;
    std::cout << i << "/" << largeInputs.size() << " " << id->description() << std::endl;
    for (int k = 0; k < NR_RUN; ++k) {
      r0 = convCPU(*id, false, hTimer);
      for (int krn = 0; krn < NR_KERNELS; ++krn) {
        rx[krn] = convGPU(*id, verbose, hTimer, NRBLOCK, krn);
        bool res = rx[krn].compare(r0);
        if (!res)
          exit(-1);
      }
      (verbose > 1) && printf("%s\n", r0.summary().c_str());
      for (int krn = 0; krn < NR_KERNELS; ++krn) {
        (verbose > 1) && printf("%s\n", rx[krn].summary().c_str());
      }
    }
    printf("%s\n", r0.summaryHead().c_str());
    if (!once) {
      myfile << r0.summaryHead() << std::endl;
      once = true;
    }
    for (int krn = 0; krn < NR_KERNELS; ++krn) {
      printf("%s\n", rx[krn].summary().c_str());
      myfile << rx[krn].summary() << std::endl;
    }
    myfile.flush();
  }
  myfile.close();
  sdkDeleteTimer(&hTimer);
#ifdef OLD
  printf(" factor=%f gpu=%5.2f msec cpu=%5.2f msec  diff idx: %s AVG_LEN=%d  NRSTRING=%d SIZE=%d BLK=%d (e|4|3|2|1) (%d|%d|%d|%d|%d) %8.1f %8.3f\n",
    r0.tm_proc / rx[2].tm_proc, rx[2].tm_proc, r0.tm_proc, rx[2].difference.c_str() /*risky*/,
    (int)r0.input->avgLen, (int)r0.input->nrStrings, (int)r0.input->nrStrings * r0.input->avgLen,
    (int)NRBLOCK,
    perc1_100_200_300[0], perc1_100_200_300[1], perc1_100_200_300[2], perc1_100_200_300[3], 1000 - perc1_100_200_300[3],
    r0.input->ActualMean(),
    r0.input->ActualStdDev()
    //  ((float)squL - sumL * sumL/(1.0*NRSTRINGS))*NRSTRINGS / ((1.0*sumL*sumL))
  );
#endif
  for (int i = 0; i < largeInputs.size(); ++i) {
    printf("...deleting input data in CPU mem # %d.\n", i);
    delete largeInputs[i];
  }

  exit(0); // (diff_idx == 0) && (diff_char == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
}
