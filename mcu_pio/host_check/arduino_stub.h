/*
 * arduino_stub.h — stub minimale di Arduino.h per compile-check su HOST
 * ---------------------------------------------------------------------
 * Permette `g++ -fsyntax-only` (e anche un link vero) di src/main.cpp
 * per la verifica offline quando i toolchain PlatformIO non sono disponibili.
 * Fornisce: micros(), Serial finto, F()/PROGMEM no-op, pgm_read_word,
 * delay(), setup()/loop() chiamati da un main() opzionale.
 *
 * NON e' un runtime: serve solo a validare che il codice compili pulito
 * per entrambe le varianti #ifdef (AVR ed ESP32), simulate via -D.
 */
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <ctime>

/* ---- macro PROGMEM / F() ---- */
#ifndef PROGMEM
#define PROGMEM
#endif
#ifndef F
#define F(x) (x)
#endif
typedef const char* __FlashStringHelper_stub;

/* ---- pgm_read (AVR) ---- */
#ifndef pgm_read_word
#define pgm_read_word(addr) (*(const int16_t*)(addr))
#endif
#ifndef pgm_read_byte
#define pgm_read_byte(addr) (*(const uint8_t*)(addr))
#endif
#ifndef pgm_read_dword
#define pgm_read_dword(addr) (*(const uint32_t*)(addr))
#endif
#ifndef memcpy_P
#define memcpy_P(dest, src, n) memcpy((dest), (src), (n))
#endif

/* ---- timing ---- */
static inline unsigned long micros() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (unsigned long)(ts.tv_sec * 1000000UL + ts.tv_nsec / 1000UL);
}
static inline unsigned long millis() { return micros() / 1000UL; }
static inline void delay(unsigned long) {}
static inline void delayMicroseconds(unsigned int) {}

/* ---- ESP32: esp_timer_get_time / esp_get_free_heap_size ---- */
#ifdef ARDUINO_ARCH_ESP32
static inline uint64_t esp_timer_get_time() { return (uint64_t)micros(); }
static inline uint32_t esp_get_free_heap_size() { return 200000u; }
#endif

/* ---- I2C (Wire) stub, per il blocco INA219 ---- */
struct WireStub {
  void begin() {}
  void beginTransmission(uint8_t) {}
  void write(uint8_t) {}
  uint8_t endTransmission() { return 0; }
  uint8_t requestFrom(uint8_t, uint8_t) { return 0; }
  int available() { return 0; }
  int read() { return 0; }
};
static WireStub Wire;

/* ---- Serial finto ---- */
#define DEC 10
#define HEX 16
struct SerialStub {
  void begin(unsigned long) {}
  operator bool() const { return true; }
  void print(const char* s) { std::fputs(s, stdout); }
  void print(int v) { std::printf("%d", v); }
  void print(unsigned int v) { std::printf("%u", v); }
  void print(long v) { std::printf("%ld", v); }
  void print(unsigned long v) { std::printf("%lu", v); }
  void print(int32_t* ) {}
  void print(double v) { std::printf("%f", v); }
  void print(double v, int d) { std::printf("%.*f", d, v); }
  void print(float v, int d) { std::printf("%.*f", d, (double)v); }
  void println() { std::putchar('\n'); }
  void println(const char* s) { std::fputs(s, stdout); std::putchar('\n'); }
  void println(int v) { std::printf("%d\n", v); }
  void println(unsigned long v) { std::printf("%lu\n", v); }
  void println(long v) { std::printf("%ld\n", v); }
  void println(double v, int d) { std::printf("%.*f\n", d, v); }
  void println(float v, int d) { std::printf("%.*f\n", d, (double)v); }
  void flush() { std::fflush(stdout); }
};
static SerialStub Serial;

/* setup()/loop() dichiarati dallo sketch */
void setup();
void loop();
