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

/* ---- GPIO finto ----
   Serve a src/main_energy.cpp, che marca le finestre di misura alzando e
   abbassando un pin per il trigger dello strumento. Qui non fa nulla: il
   compile-check verifica solo che il firmware compili per entrambe le
   varianti #ifdef senza i toolchain PlatformIO. */
#ifndef HIGH
#define HIGH 1
#endif
#ifndef LOW
#define LOW 0
#endif
#ifndef OUTPUT
#define OUTPUT 1
#endif
#ifndef INPUT
#define INPUT 0
#endif
static inline void pinMode(int, int) {}
static inline void digitalWrite(int, int) {}
static inline int  digitalRead(int) { return 0; }

/* ---- Serial finto ---- */
#define DEC 10
#define HEX 16
struct SerialStub {
  void begin(unsigned long) {}
  operator bool() const { return true; }
  void print(const char* s) { std::fputs(s, stdout); }
  /* senza questa, Serial.print(',') finiva su print(int) e stampava "44":
     l'output CSV del compile-check era illeggibile, e su hardware vero no */
  void print(char c) { std::putchar(c); }
  void println(char c) { std::putchar(c); std::putchar('\n'); }
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
