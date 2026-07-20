/* avr/pgmspace.h — stub per compile-check HOST (solo offline).
 * Il toolchain AVR reale di PlatformIO fornisce la versione vera. */
#pragma once
#ifndef PROGMEM
#define PROGMEM
#endif
#ifndef pgm_read_word
#define pgm_read_word(addr) (*(const short*)(addr))
#endif
#ifndef pgm_read_byte
#define pgm_read_byte(addr) (*(const unsigned char*)(addr))
#endif
