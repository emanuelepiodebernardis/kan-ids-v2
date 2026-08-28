/* BENCHMARK DI ENERGIA — una finestra lunga di sole inferenze, niente altro.
 *
 * Perche' serve un firmware separato
 * ----------------------------------
 * I nove firmware di latenza cronometrano UNA inferenza per volta e fra una
 * misura e la successiva stampano 5-9 valori su Serial. Per la latenza va
 * bene: fra t0 e t1 c'e' solo la chiamata al kernel. Per l'energia no: uno
 * strumento misura la corrente assorbita nel tempo, e in quel tempo la UART
 * a 115200 baud e — dove era abilitato l'hook INA219 — anche il bus I2C
 * consumano molto piu' dell'inferenza. L'integratore che c'era in main.cpp
 * accumulava proprio su quegli intervalli sporchi e chiamava l'I2C dentro il
 * conteggio: misurava l'energia della UART, non quella del modello.
 *
 * Come e' fatto qui (richiesta del Prof. Kuznetsov, punto 4)
 * ----------------------------------------------------------
 * Ogni ripetizione produce due finestre della STESSA durata, adiacenti:
 *
 *     EB_PIN alto      finestra ATTIVA: EB_BATCH inferenze consecutive, e
 *                      nulla altro. Nessuna Serial, nessun Wire, nessun
 *                      delay, nessun accesso a PROGMEM: i vettori di
 *                      ingresso sono gia' in RAM, e l'indice avanza con un
 *                      confronto (niente divisioni nel ciclo misurato).
 *     EB_PIN_REF alto  finestra di RIFERIMENTO: la CPU gira a vuoto su `nop`
 *                      per un numero di giri calibrato in modo da durare
 *                      quanto la finestra attiva. Stesso clock, stesse
 *                      periferiche, stessa durata, nessuna inferenza.
 *
 * I due marcatori sono pin DISTINTI (rc3): con un pin solo il livello basso
 * significava sia "finestra di riferimento" sia "tutto il resto" — setup,
 * intervalli fra le ripetizioni, stampe finali — e chi integra la corrente
 * doveva fidarsi dell'ordine invece di leggerlo dalla traccia. Adesso ogni
 * finestra ha il suo fronte, e nessun campione entra nell'integrale
 * sbagliato. L'energia per inferenza si ricava come
 *
 *     E_inf = (P_attiva - P_riferimento) * T_finestra / EB_BATCH
 *
 * Tutte le stampe sono PRIMA della prima finestra e DOPO l'ultima. Fra le
 * ripetizioni non viene eseguita una sola istruzione di I/O.
 *
 * Sulla linea di base: la CPU sveglia che gira su `nop` consuma piu' di una
 * CPU in sleep e meno di una che esegue il kernel. Sottraendola si ottiene
 * il costo MARGINALE dell'inferenza rispetto a un processore acceso e
 * inattivo, che e' la quantita' confrontabile fra i sette modelli. Il
 * consumo assoluto del sistema e' la sola finestra attiva, ed e' comunque
 * misurabile perche' le due finestre sono separate sul pin.
 *
 * Contro l'eliminazione del codice morto
 * --------------------------------------
 * Nei firmware di latenza l'unica cosa che impediva al compilatore di
 * cancellare le inferenze era la Serial.print del risultato. Qui la Serial
 * non c'e': i kernel sono `static inline` in header, le costanti stanno in
 * Flash e senza precauzioni `-O2` potrebbe eliminare l'intero ciclo. Il
 * risultato di ogni inferenza viene quindi accumulato in un `volatile`, e
 * alla fine la somma viene CONFRONTATA con quella attesa, calcolata dai
 * golden vector. Se il confronto fallisce il firmware lo dichiara: una
 * finestra vuota non puo' essere scambiata per una finestra veloce.
 *
 * Uso
 * ---
 *   pio run -e megaatmega2560_energy -t upload      (variante di default)
 *   pio run -e esp32c3_energy -t upload
 *
 * Varianti selezionabili a compilazione, una sola per volta:
 *   -DEB_COEFF (default)  KAN single-layer a coefficienti, 254 B
 *   -DEB_MLCOEFF          KAN multi-layer, 5.244 B
 *   -DEB_MC               KAN multiclasse 10 classi, 8.268 B
 *   -DEB_E2E              catena integer end-to-end binaria, 1.334 B
 *   -DEB_DT5              albero di decisione profondo 5, 285 B
 *   -DEB_MLP              MLP piccolo, 16 nascosti ReLU, 760 B
 *   -DEB_LUT14            KAN single-layer campionata (sampled-LUT), 5.194 B
 *
 * Altri parametri: -DEB_BATCH=n (inferenze per finestra), -DEB_REPS=n
 * (ripetizioni), -DEB_PIN=n e -DEB_PIN_REF=n (i due marcatori),
 * -DEB_TOLL_PERMILLE=n (quanto le due finestre possono discostarsi),
 * -DEB_NO_PIN (nessun pin).
 *
 * I pin di marcatura vanno collegati SOLO agli ingressi di trigger dello
 * strumento, che sono ad alta impedenza. Non usare il pin del LED di bordo:
 * il LED assorbe corrente e finirebbe dentro la misura. Per questo il
 * default non e' LED_BUILTIN.
 *
 * Che cosa controllare nell'output prima di fidarsi di una misura:
 *   checksum_ok=1        le inferenze sono avvenute (nessuna finestra vuota)
 *   calibration_ok=1     la calibrazione del ciclo di riferimento e' riuscita
 *   windows_ok=1         le due finestre durano lo stesso entro tolleranza
 * Le due durate misurate e il loro scarto in parti per mille sono su ogni
 * riga: sono numeri, non promesse.
 */
#ifdef HOST_CHECK
  #include "arduino_stub.h"
#else
  #include <Arduino.h>
#endif
#include <stdint.h>
#include <string.h>

/* ── selezione della variante ─────────────────────────────────────── */
#if !defined(EB_COEFF) && !defined(EB_MLCOEFF) && !defined(EB_MC) && \
    !defined(EB_E2E) && !defined(EB_DT5) && !defined(EB_MLP) && \
    !defined(EB_LUT14)
  #define EB_COEFF
#endif

#if defined(EB_COEFF)
  #include "kan14_coeff_infer.h"
  #include "kan14_test_vectors.h"
  #define EB_NAME       "coeff_int8"
  #define EB_MODEL_BYTES 254
#elif defined(EB_MLCOEFF)
  #include "kan14_ml_coeff_infer.h"
  #include "kan14_ml_test_vectors.h"
  #define EB_NAME       "ml_coeff_int8"
  #define EB_MODEL_BYTES 5244
#elif defined(EB_MC)
  #include "kan14_mc_coeff_infer.h"
  #include "kan14_mc_test_vectors.h"
  #define EB_NAME       "mc_coeff_int8"
  #define EB_MODEL_BYTES 8268
#elif defined(EB_E2E)
  #include "kan_e2e_int.h"
  #include "kan_e2e_infer.h"
  #define EB_NAME       "e2e_int"
  #define EB_MODEL_BYTES 1334
#elif defined(EB_DT5)
  #include "dt5_model.h"
  #define EB_NAME       "dt5"
  #define EB_MODEL_BYTES 285
#elif defined(EB_MLP)
  #include "mlp16_infer.h"
  #include "mlp16_test_vectors.h"
  #define EB_NAME       "mlp16_int8"
  #define EB_MODEL_BYTES 760
#elif defined(EB_LUT14)
  /* Stessa KAN single-layer di EB_COEFF, funzioni campionate invece che a
   * coefficienti: gli ingressi e le predizioni attese sono gli stessi, quindi
   * la differenza misurata e' quella della sola rappresentazione. */
  #include "kan14_lut_infer.h"
  #include "kan14_test_vectors.h"
  #define EB_NAME       "lut_int16"
  #define EB_MODEL_BYTES 5194
#endif

/* ── parametri ────────────────────────────────────────────────────── */
#ifndef EB_BATCH
  #define EB_BATCH 2000            /* inferenze per finestra attiva */
#endif
#ifndef EB_REPS
  #define EB_REPS 5                /* finestre ripetute */
#endif
#ifndef EB_CACHE
  #define EB_CACHE 20              /* vettori tenuti in RAM (10 + 10) */
#endif
#ifndef EB_PIN
  #if defined(__AVR__)
    #define EB_PIN 22              /* pin digitale libero sul Mega, NON il LED */
  #else
    #define EB_PIN 3               /* GPIO libero su ESP32-C3 DevKitM-1 */
  #endif
#endif
/* Marcatore della finestra di RIFERIMENTO, separato da quello della finestra
 * attiva. Con un pin solo il livello basso significa due cose diverse — la
 * finestra di riferimento E tutto il resto (setup, intervalli fra le
 * ripetizioni, stampe finali) — e chi integra la corrente deve fidarsi
 * dell'ordine invece di leggerlo. Con due pin ogni finestra ha il suo
 * marcatore alto, e le due integrazioni si ritagliano dalla traccia senza
 * ambiguita'. */
#ifndef EB_PIN_REF
  #if defined(__AVR__)
    #define EB_PIN_REF 24          /* adiacente al 22 sullo stesso header */
  #else
    #define EB_PIN_REF 4           /* GPIO libero accanto al 3 */
  #endif
#endif

#if defined(__AVR__)
  #define EB_RD16(p) ((int16_t)pgm_read_word(&(p)))
  #define EB_RD8(p)  ((uint8_t)pgm_read_byte(&(p)))
#else
  #define EB_RD16(p) (p)
  #define EB_RD8(p)  (p)
#endif

/* ── vettori di ingresso, copiati in RAM prima delle finestre ─────── */
#if defined(EB_E2E)
static int32_t  eb_sb[EB_CACHE], eb_db[EB_CACHE], eb_sp[EB_CACHE], eb_dp[EB_CACHE];
static int32_t  eb_dur[EB_CACHE];
#elif defined(EB_DT5)
static int16_t  eb_x[EB_CACHE][DT5_NFEAT];
#else
static int16_t  eb_x[EB_CACHE][10];
static uint8_t  eb_c[EB_CACHE][4];
#endif
static uint8_t  eb_expected[EB_CACHE];

/* Meta' dei vettori dal blocco attacco, meta' dal blocco normale: il ciclo
 * non deve girare su un ingresso degenere, perche' i rami presi cambiano il
 * consumo. Gli indici seguono la stessa convenzione dei firmware di latenza
 * (prima meta' attacco, seconda meta' normale). */
static void eb_load(void) {
  for (uint8_t i = 0; i < EB_CACHE; i++) {
    const uint8_t half = EB_CACHE / 2;
#if defined(EB_E2E)
    const uint16_t k = (i < half) ? i : (E2E_N_GOLDEN / 2 + (i - half));
    e2e_golden_t g;
    memcpy_P(&g, &E2E_GOLDEN[k], sizeof(g));
    eb_sb[i] = g.sb; eb_db[i] = g.db; eb_sp[i] = g.sp; eb_dp[i] = g.dp;
    eb_dur[i] = g.dur_us;
    eb_expected[i] = g.dec;
#elif defined(EB_DT5)
    const uint16_t k = (i < half) ? i : (DT5_N_GOLDEN / 2 + (i - half));
    dt5_golden_t g;
    memcpy_P(&g, &DT5_GOLDEN[k], sizeof(g));
    memcpy(eb_x[i], g.x, sizeof(g.x));
    eb_expected[i] = g.pred;
#else
  #if defined(EB_COEFF)
    #define EB_TVX KTV_X
    #define EB_TVC KTV_CAT
    #define EB_TVE KTV_EXPECTED
    #define EB_TVN KTV_N
  #elif defined(EB_MLCOEFF)
    #define EB_TVX KMLTV_X
    #define EB_TVC KMLTV_CAT
    #define EB_TVE KMLTV_EXPECTED
    #define EB_TVN KMLTV_N
  #elif defined(EB_LUT14)
    #define EB_TVX KTV_X
    #define EB_TVC KTV_CAT
    #define EB_TVE KTV_EXPECTED
    #define EB_TVN KTV_N
  #elif defined(EB_MLP)
    #define EB_TVX MLPTV_X
    #define EB_TVC MLPTV_CAT
    #define EB_TVE MLPTV_EXPECTED
    #define EB_TVN MLPTV_N
  #else
    #define EB_TVX KMCTV_X
    #define EB_TVC KMCTV_CAT
    #define EB_TVE KMCTV_EXPECTED
    #define EB_TVN KMCTV_N
  #endif
    const uint16_t k = (i < half) ? i : (EB_TVN / 2 + (i - half));
    for (uint8_t j = 0; j < 10; j++) eb_x[i][j] = EB_RD16(EB_TVX[k][j]);
    for (uint8_t j = 0; j < 4;  j++) eb_c[i][j] = EB_RD8(EB_TVC[k][j]);
    eb_expected[i] = EB_RD8(EB_TVE[k]);
#endif
  }
}

/* Una inferenza sul vettore i, gia' in RAM. Questo e' TUTTO quello che gira
 * dentro la finestra misurata. */
static inline uint8_t eb_one(uint8_t i) {
#if defined(EB_COEFF)
  return kan14_coeff_predict(eb_x[i], eb_c[i]);
#elif defined(EB_MLCOEFF)
  return kan14_ml_predict(eb_x[i], eb_c[i]);
#elif defined(EB_MC)
  return kan14_mc_predict(eb_x[i], eb_c[i]);
#elif defined(EB_E2E)
  return e2e_predict(eb_sb[i], eb_db[i], eb_sp[i], eb_dp[i], eb_dur[i]);
#elif defined(EB_DT5)
  return dt5_predict(eb_x[i]);
#elif defined(EB_MLP)
  return mlp16_predict(eb_x[i], eb_c[i]);
#elif defined(EB_LUT14)
  return kan14_lut_predict(eb_x[i], eb_c[i]);
#endif
}

/* ── accumulatore volatile: senza, il ciclo puo' sparire ──────────── */
static volatile uint32_t eb_acc = 0;

/* LA FINESTRA MISURATA, e nient'altro (richiesta del Prof. Kuznetsov, rc3).
 *
 * La versione precedente scriveva
 *
 *     for (uint32_t k = 0; k < EB_BATCH; k++) eb_acc += eb_one(k % EB_CACHE);
 *
 * e dentro la finestra pagava due costi estranei al modello. Il primo e'
 * `k % EB_CACHE`: k e' un uint32 e EB_CACHE non e' una potenza di due,
 * quindi su AVR ogni giro chiamava __udivmodsi4 — una routine di libgcc da
 * qualche centinaio di cicli, contro le poche centinaia dell'inferenza
 * intera che si vuole misurare. Il secondo e' `eb_acc +=` su un volatile:
 * quattro byte riletti dalla RAM, sommati e riscritti a ogni giro, e
 * l'accumulatore escluso dai registri per definizione.
 *
 * Adesso l'indice avanza con un confronto e un azzeramento (la sequenza
 * degli ingressi e' identica: 0,1,...,EB_CACHE-1,0,1,...), la somma vive in
 * un registro, e il volatile viene scritto UNA volta alla fine del batch.
 * Scriverlo alla fine basta a impedire l'eliminazione del ciclo, perche' il
 * valore osservabile dipende da tutte le inferenze.
 *
 * Il ciclo resta piatto di proposito. Scriverlo annidato — EB_BATCH/EB_CACHE
 * giri esterni su EB_CACHE vettori — avrebbe tolto anche il confronto, ma il
 * ciclo interno sarebbe stato invariante rispetto a quello esterno: un
 * compilatore autorizzato a sollevarlo eseguirebbe EB_CACHE inferenze e
 * moltiplicherebbe per il resto, e il checksum tornerebbe lo stesso. Sarebbe
 * la finestra vuota che questo firmware esiste per rendere impossibile.
 *
 * E' una funzione a se' perche' cosi' la si puo' ispezionare nell'assembly
 * emesso per ATmega2560: un test pretende che il suo corpo non contenga
 * NESSUNA chiamata a libgcc (niente __udivmodsi4, niente soft-float, niente
 * helper a 64 bit). La finestra misurata e' un oggetto con un nome. */
#if defined(__GNUC__)
  #define EB_NOINLINE __attribute__((noinline))
#else
  #define EB_NOINLINE
#endif

/* `noinline` di proposito: cinque chiamate in tutto (una per ripetizione,
 * fuori dal conteggio delle inferenze) in cambio di una funzione che esiste
 * nell'assembly con un nome e dei confini. Senza, il compilatore la fonde
 * dentro setup() e la finestra misurata non e' piu' un oggetto che si possa
 * ispezionare: il test che pretende zero chiamate a libgcc nel ciclo non
 * saprebbe dove guardare. */
static EB_NOINLINE uint32_t eb_finestra_attiva(uint32_t giri) {
  uint32_t acc = 0;
  uint8_t  i   = 0;
  while (giri--) {
    acc += eb_one(i);
    if (++i == (uint8_t)EB_CACHE) i = 0;
  }
  eb_acc = acc;                    /* unico accesso al volatile del batch */
  return acc;
}

/* Quanti giri del ciclo di riferimento stanno in un microsecondo, in Q8.
 *
 * PERCHE' IN Q8 E PERCHE' COL MEDESIMO CICLO.
 * La prima versione cronometrava un ciclo diverso da quello poi usato:
 *
 *     while ((uint32_t)(micros() - t0) < 20000UL) { nop; n++; }
 *     eb_nop_per_us = n / 20000UL;
 *
 * Ogni giro pagava una chiamata a micros(), che su AVR costa ~3,4 us fra
 * lettura del timer e disabilitazione degli interrupt. In 20 ms si fanno
 * cosi' ~5.900 giri invece dei ~320.000 di un ciclo di soli nop, e la
 * divisione intera 5900/20000 da' ZERO, poi forzato a 1 dal guard. La
 * finestra di riferimento girava quindi a 1 giro per microsecondo dove il
 * ciclo vero ne fa sedici: durava piu' di un ordine di grandezza meno di
 * quella attiva, mentre E = (P_alta - P_bassa) * T / N presuppone che le
 * due durate coincidano.
 *
 * Adesso si cronometra ESATTAMENTE la funzione che verra' usata, con
 * micros() chiamato due volte in tutto e fuori dal ciclo, e il risultato
 * si tiene in virgola fissa Q8 perche' su un core veloce il valore intero
 * arrotonderebbe di nuovo male. */
static uint32_t eb_nop_per_us_q8 = 0;
static uint8_t  eb_calibrazione_ok = 0;

/* L'unico ciclo di riferimento del firmware: quello calibrato e quello
 * eseguito devono essere lo stesso codice, altrimenti si calibra una cosa
 * e se ne misura un'altra. `volatile` impedisce all'ottimizzatore di
 * cancellarlo. */
static void eb_nop_loop(uint32_t giri) {
  volatile uint32_t n = giri;
  while (n--) { __asm__ __volatile__("nop"); }
}

static void eb_calibra(void) {
  /* Venti raddoppi, non dodici. Dodici bastano su un AVR a 16 MHz, dove il
   * ciclo di riferimento e' lento; su un core veloce — ESP32-C3, o l'host su
   * cui gira la verifica — 20000<<11 giri stanno sotto i 50 ms e la
   * calibrazione finiva i tentativi, ripiegando su un valore arbitrario e
   * dichiarando calibration_ok=0. Il ripiego funzionava come previsto, ma
   * una calibrazione che non converge sulla macchina piu' comune non e' una
   * calibrazione. Il limite superiore evita di far traboccare `giri`. */
  uint32_t giri = 20000UL;
  for (uint8_t tent = 0; tent < 20 && giri < (1UL << 31); tent++) {
    const uint32_t t0 = micros();
    eb_nop_loop(giri);
    const uint32_t dt = (uint32_t)(micros() - t0);
    if (dt >= 50000UL) {                       /* almeno 50 ms di finestra */
      eb_nop_per_us_q8 = (uint32_t)(((uint64_t)giri << 8) / dt);
      if (eb_nop_per_us_q8 == 0) eb_nop_per_us_q8 = 1;
      eb_calibrazione_ok = 1;
      return;
    }
    giri <<= 1;                                /* troppo corta: raddoppia */
  }
  /* Nessun tentativo ha raggiunto 50 ms: succede se micros() non avanza
   * (host stub) o su un core cosi' veloce da esaurire i raddoppi. Il valore
   * di ripiego e' arbitrario e il firmware lo DICE: una finestra di
   * riferimento sbagliata in silenzio e' il difetto che questa funzione
   * esiste per non ripetere. */
  eb_nop_per_us_q8 = 1 << 8;
}

/* Finestra di riferimento: stessa durata della finestra attiva, CPU sveglia,
 * nessuna inferenza. Restituisce la durata MISURATA, che il firmware stampa:
 * "stessa durata" e' una promessa, e chi legge i numeri deve poterla
 * verificare invece di crederci. */
static uint32_t eb_riferimento(uint32_t durata_us) {
  const uint32_t giri =
      (uint32_t)(((uint64_t)eb_nop_per_us_q8 * durata_us) >> 8);
#ifndef EB_NO_PIN
  digitalWrite(EB_PIN_REF, HIGH);
#endif
  const uint32_t t0 = micros();
  eb_nop_loop(giri);
  const uint32_t dt = (uint32_t)(micros() - t0);
#ifndef EB_NO_PIN
  digitalWrite(EB_PIN_REF, LOW);
#endif
  return dt;
}

/* Scarto fra le due finestre in parti per mille, con segno. E' il numero che
 * rende verificabile la frase "stessa durata": se non e' piccolo, sottrarre
 * la potenza di riferimento da quella attiva non produce un'energia. */
static int32_t eb_permille(uint32_t attiva, uint32_t riferimento) {
  if (attiva == 0) return 0;
  return (int32_t)(((int64_t)riferimento - (int64_t)attiva) * 1000
                   / (int64_t)attiva);
}

/* Quanto puo' discostarsi la finestra di riferimento prima che la misura non
 * valga piu'. Il 5% e' largo per una calibrazione in Q8 su un ciclo di soli
 * nop e stretto abbastanza da intercettare il difetto vero trovato prima
 * (finestra di riferimento un SEDICESIMO di quella attiva, cioe' -937). */
#ifndef EB_TOLL_PERMILLE
  #define EB_TOLL_PERMILLE 50
#endif

void setup() {
  Serial.begin(115200);
  delay(1500);

  eb_load();
  eb_calibra();

  /* somma attesa delle predizioni sulla finestra: e' la prova che le
   * inferenze sono state eseguite davvero e non ottimizzate via */
  uint32_t attesa_per_batch = 0;
  for (uint32_t r = 0; r < (uint32_t)EB_BATCH; r++)
    attesa_per_batch += eb_expected[r % EB_CACHE];

  /* riscaldamento fuori da ogni finestra */
  for (uint16_t w = 0; w < 64; w++) eb_acc += eb_one(w % EB_CACHE);
  eb_acc = 0;

#ifndef EB_NO_PIN
  pinMode(EB_PIN, OUTPUT);
  digitalWrite(EB_PIN, LOW);
  pinMode(EB_PIN_REF, OUTPUT);
  digitalWrite(EB_PIN_REF, LOW);
#endif

  Serial.print(F("# energy benchmark variant=")); Serial.print(F(EB_NAME));
  Serial.print(F(" model_bytes=")); Serial.print(EB_MODEL_BYTES);
  Serial.print(F(" batch=")); Serial.print((uint32_t)EB_BATCH);
  Serial.print(F(" reps="));  Serial.print((uint32_t)EB_REPS);
  Serial.print(F(" vectors_in_ram=")); Serial.print((uint32_t)EB_CACHE);
#ifndef EB_NO_PIN
  Serial.print(F(" marker_pin_active=")); Serial.print((uint32_t)EB_PIN);
  Serial.print(F(" marker_pin_ref="));    Serial.print((uint32_t)EB_PIN_REF);
#else
  Serial.print(F(" marker_pin=none"));
#endif
  Serial.println();
  Serial.println(F("# due marcatori distinti: ALTO su marker_pin_active = "
                   "finestra di inferenze, ALTO su marker_pin_ref = finestra "
                   "di riferimento; nessun I/O dentro nessuna delle due"));
  Serial.println(F("# E_totale per inferenza  = P_alta * T_alta / batch"));
  Serial.println(F("# E_dinamica per inferenza = (P_alta - P_bassa) * T_alta / batch"));
  Serial.println(F("# la prima include il consumo statico del core sveglio, la "
                   "seconda e' il solo costo del calcolo"));
  Serial.flush();
  delay(200);                     /* la UART deve essere ferma prima di iniziare */

  uint32_t durata[EB_REPS];       /* finestra ATTIVA, misurata */
  uint32_t durata_rif[EB_REPS];   /* finestra di RIFERIMENTO, misurata */
  uint32_t somma[EB_REPS];

  for (uint8_t rep = 0; rep < EB_REPS; rep++) {
    eb_acc = 0;

    /* ---------- finestra ATTIVA ---------- */
#ifndef EB_NO_PIN
    digitalWrite(EB_PIN, HIGH);
#endif
    const uint32_t t0 = micros();
    const uint32_t somma_rep = eb_finestra_attiva((uint32_t)EB_BATCH);
    const uint32_t t1 = micros();
#ifndef EB_NO_PIN
    digitalWrite(EB_PIN, LOW);
#endif
    /* ---------- fine finestra ATTIVA ---------- */

    durata[rep] = t1 - t0;
    somma[rep]  = somma_rep;

    durata_rif[rep] = eb_riferimento(durata[rep]);   /* finestra BASSA */
  }

  /* Da qui in poi si puo' tornare a parlare. */
  uint8_t tutte_ok = 1, finestre_ok = 1;
  Serial.println(F("variant,rep,batch,window_us,ref_us,ref_vs_active_permille,"
                   "windows_match,ns_per_inference,checksum,expected,ok"));
  for (uint8_t rep = 0; rep < EB_REPS; rep++) {
    const uint8_t ok = (somma[rep] == attesa_per_batch);
    if (!ok) tutte_ok = 0;
    const int32_t pm = eb_permille(durata[rep], durata_rif[rep]);
    const uint8_t pari = (pm <= (int32_t)EB_TOLL_PERMILLE &&
                          pm >= -(int32_t)EB_TOLL_PERMILLE);
    if (!pari) finestre_ok = 0;
    Serial.print(F(EB_NAME));     Serial.print(',');
    Serial.print(rep);            Serial.print(',');
    Serial.print((uint32_t)EB_BATCH); Serial.print(',');
    Serial.print(durata[rep]);    Serial.print(',');
    Serial.print(durata_rif[rep]); Serial.print(',');
    Serial.print(pm);             Serial.print(',');
    Serial.print(pari ? 1 : 0);   Serial.print(',');
    /* niente virgola mobile nemmeno qui: su AVR tirerebbe dentro le
     * routine soft-float di libgcc in un firmware che esiste per
     * misurare un modello integer-only. Nanosecondi in interi. */
    Serial.print((uint32_t)((durata[rep] * 1000UL) / (uint32_t)EB_BATCH));
    Serial.print(',');
    Serial.print(somma[rep]);     Serial.print(',');
    Serial.print(attesa_per_batch); Serial.print(',');
    Serial.println(ok ? 1 : 0);
  }

  uint32_t tot = 0;
  for (uint8_t rep = 0; rep < EB_REPS; rep++) tot += durata[rep];
  Serial.print(F("SUMMARY variant=")); Serial.print(F(EB_NAME));
  Serial.print(F(" model_bytes=")); Serial.print(EB_MODEL_BYTES);
  Serial.print(F(" mean_window_us=")); Serial.print(tot / EB_REPS);
  Serial.print(F(" mean_ns_per_inference="));
  Serial.print((uint32_t)((tot * 1000UL) / ((uint32_t)EB_REPS * (uint32_t)EB_BATCH)));
  uint32_t tot_rif = 0;
  for (uint8_t rep = 0; rep < EB_REPS; rep++) tot_rif += durata_rif[rep];
  Serial.print(F(" mean_ref_us=")); Serial.print(tot_rif / EB_REPS);
  /* scarto fra le due finestre in parti per mille: se non e' piccolo, la
   * sottrazione del baseline non ha senso e va detto qui, non scoperto dopo. */
  Serial.print(F(" ref_vs_active_permille="));
  Serial.print(eb_permille(tot, tot_rif));
  Serial.print(F(" nop_per_us_q8=")); Serial.print(eb_nop_per_us_q8);
  Serial.print(F(" calibration_ok=")); Serial.print(eb_calibrazione_ok ? 1 : 0);
  Serial.print(F(" windows_ok=")); Serial.print(finestre_ok ? 1 : 0);
  Serial.print(F(" tolerance_permille=")); Serial.print((int32_t)EB_TOLL_PERMILLE);
  Serial.print(F(" checksum_ok=")); Serial.print(tutte_ok ? 1 : 0);
  Serial.println();
  if (!finestre_ok) {
    Serial.println(F("ATTENZIONE: le due finestre non hanno la stessa durata "
                     "entro la tolleranza. La differenza fra le due potenze "
                     "NON e' l'energia dinamica dell'inferenza: usare la sola "
                     "finestra attiva, oppure ricalibrare."));
  }
  if (!tutte_ok) {
    Serial.println(F("ATTENZIONE: il checksum non torna. Le inferenze nella "
                     "finestra potrebbero essere state eliminate dal "
                     "compilatore: la misura di energia NON e' valida."));
  }
}

void loop() {}
