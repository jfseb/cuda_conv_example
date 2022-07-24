/*
 * (c) ICU code, see respective license
 * Extensions and wrapping Copyright (c) 2022, Gerd Forstmann. All rights reserved.
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "../icu/source/common/unicode/utf8.h"

#define int32_t int
#define UChar32 int // int32 ! signed!
////////////////////////////////////////////////////////////////////////////
// Convert the buffer starting at src, described by srcOffsets, assuming to be utf8
// content into a corresponding buffer with ucs32 encoding.
//
// The three strings "AB\0C", "", "HI" are represented
//
// note: Strings may contain 0 bytes.
//
// src -> [A] [B] [\0] [C]|[H ] [I ]|
// offset {0, 4, 4, 6}
////////////////////////////////////////////////////////////////////////////
extern "C" int convChunkUTF8_UCHAR32(uint8_t* src, int* srcOffsets, int nrStrings, UChar32* dest, int* destOffsets) {
  const UChar32 UC_REPL = 0xFFFD;
  const int nrOffsetsN = nrStrings + 1;
  bool flawed = false;
  assert(sizeof(int) == 4);
  assert(sizeof(UChar32) == 4);
  destOffsets[0] = 0;
  int32_t tgt = 0;
  // the most naive implementation.
  for (int nrString = 1; nrString < nrOffsetsN; ++nrString) {
    int32_t length;
    length = srcOffsets[nrString] - srcOffsets[nrString - 1];
    const uint8_t* s = src + srcOffsets[nrString - 1];
    UChar32 c;
    int32_t i = 0;
    while (i < length) {
      U8_NEXT(s, i, length, c); // advances i
      if (c < 0) {
        dest[tgt++] = UC_REPL;
        flawed = true;
      }
      else {
        dest[tgt++] = c;
      }
    }
    destOffsets[nrString] = tgt;
  }
  return flawed;
}
