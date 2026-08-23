/* avr/pgmspace.h — stub per compile-check HOST (solo offline).
 * Il toolchain AVR reale di PlatformIO fornisce la versione vera. */
#pragma once
#include <string.h>
#include <stdint.h>
#ifndef PROGMEM
#define PROGMEM
#endif
#ifndef pgm_read_word
#define pgm_read_word(addr) (*(const short*)(addr))
#endif
#ifndef pgm_read_byte
#define pgm_read_byte(addr) (*(const unsigned char*)(addr))
#endif
#ifndef pgm_read_dword
#define pgm_read_dword(addr) (*(const uint32_t*)(addr))
#endif
#ifndef memcpy_P
#define memcpy_P(dest, src, n) memcpy((dest), (src), (n))
#endif
