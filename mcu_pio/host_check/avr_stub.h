/*
 * avr_stub.h — stub minimale del core Arduino PER IL TARGET AVR
 * -------------------------------------------------------------
 * Non e' il gemello di arduino_stub.h. Quello serve a compilare i firmware
 * sull'HOST (x86, libstdc++, Serial che stampa davvero) per eseguirli e
 * confrontarne le predizioni. Questo serve a compilarli per ATmega2560 e
 * LEGGERE L'ASSEMBLY EMESSO, che e' l'unico posto dove si vede che cosa il
 * processore esegue davvero.
 *
 * Perche' serve un secondo stub. avr-g++ non ha libstdc++: <cstdint>,
 * <cstdio> e `std::printf` non esistono, e arduino_stub.h non compila per
 * AVR nemmeno con -fsyntax-only. E installare il core Arduino completo per
 * far girare un test sarebbe un requisito enorme per una verifica di tre
 * righe di assembly.
 *
 * Che cosa e' vero e che cosa e' finto, perche' la differenza decide che
 * cosa il test puo' concludere:
 *
 *   VERO   PROGMEM, pgm_read_byte/word/dword, memcpy_P: sono quelli di
 *          avr-libc, inclusi da <avr/pgmspace.h>. Gli accessi alla Flash
 *          nell'assembly sono quindi le LPM/ELPM reali, con il costo reale.
 *   VERO   ogni tipo intero, ogni promozione, ogni helper di libgcc che il
 *          compilatore decide di chiamare: e' esattamente il codice che
 *          finisce sul microcontrollore.
 *   FINTO  Serial (non fa nulla), micros() (contatore fittizio), pinMode e
 *          digitalWrite (vuoti).
 *
 * Le parti finte stanno tutte FUORI dalla finestra misurata — il firmware e'
 * costruito perche' ci stiano — quindi l'assembly della finestra e' quello
 * vero. Questo stub non serve a eseguire niente: un binario prodotto cosi'
 * non stampa e non misura. Serve a guardare il codice generato.
 */
#pragma once
#include <stdint.h>
#include <string.h>
#include <avr/pgmspace.h>

#ifndef F
#define F(x) (x)
#endif

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

/* Contatore fittizio: avanza a ogni chiamata perche' i cicli di calibrazione
 * che aspettano un tempo minimo devono terminare invece di girare in eterno
 * se qualcuno prova davvero a eseguire questo binario. Non e' un tempo. */
static inline unsigned long micros(void) {
  static unsigned long t = 0;
  t += 100000UL;
  return t;
}
static inline unsigned long millis(void) { return micros() / 1000UL; }
static inline void delay(unsigned long) {}
static inline void delayMicroseconds(unsigned int) {}

static inline void pinMode(int, int) {}
static inline void digitalWrite(int, int) {}
static inline int  digitalRead(int) { return 0; }

/* I template coprono tutti gli overload di Serial senza elencarli: qui
 * nessuna stampa deve avvenire, quindi il corpo e' vuoto per costruzione. */
struct SerialStubAvr {
  void begin(unsigned long) {}
  void flush() {}
  void println() {}
  template <class T> void print(T) {}
  template <class T> void print(T, int) {}
  template <class T> void println(T) {}
  template <class T> void println(T, int) {}
};
static SerialStubAvr Serial;

void setup();
void loop();
