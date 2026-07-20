/*
 * main_lgb_wokwi.cpp — LightGBM multiclass (20 alberi/classe) su ESP32-C3
 * -------------------------------------------------------------------------
 * Modello: LightGBM(n_estimators=20, 10 classi), F1=0.9480
 * Export: m2cgen → codice C (if/else) con softmax
 * Input: feature grezze (stesso ordine del progetto, seed=42)
 * Preprocessing: NESSUNO (LGB è invariante a trasf. monotone)
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>

// Codice generato da m2cgen (LGB 20 alberi/classe)
// Inserire qui il contenuto di lgb20_m2cgen.c
// (incluso via #include nella versione Wokwi)
#include "lgb20_m2cgen.h"
#include "dt_test_vectors.h"

static int lgb_predict(const float *x) {
    double xd[DT_N_FEATURES];
    for (int j = 0; j < DT_N_FEATURES; j++) xd[j] = (double)x[j];
    double scores[10];
    lgb_score(xd, scores);
    int best = 0;
    for (int c = 1; c < 10; c++) if (scores[c] > scores[best]) best = c;
    return best;
}

static unsigned long micros_now() {
#ifdef ESP_PLATFORM
    return (unsigned long)(esp_timer_get_time());
#else
    return micros();
#endif
}

void setup() {
    Serial.begin(115200);
    while (!Serial) {}

    Serial.println(F("=== LightGBM multiclass (20 alberi/classe) ==="));
    Serial.print(F("F1=0.9480  classi=10  feature=10"));
    Serial.println();
    Serial.println(F("idx,label,pred,match,latency_us"));

    int correct = 0;
    unsigned long tot = 0, tmin = 0xFFFFFFFF, tmax = 0;

    for (int i = 0; i < N_TEST_DT; i++) {
        unsigned long t0 = micros_now();
        int pred = lgb_predict(TEST_RAW_DT[i]);
        unsigned long us = micros_now() - t0;

        if (pred == TEST_LABEL_DT[i]) correct++;
        tot += us;
        if (us < tmin) tmin = us;
        if (us > tmax) tmax = us;

        Serial.print(i);                       Serial.print(F(","));
        Serial.print(TEST_LABEL_DT[i]);        Serial.print(F(","));
        Serial.print(pred);                    Serial.print(F(","));
        Serial.print(pred==TEST_LABEL_DT[i]?'Y':'N'); Serial.print(F(","));
        Serial.println(us);
    }

    Serial.println(F("--- riepilogo ---"));
    Serial.print(F("accuratezza: "));
    Serial.print(100.0f * correct / N_TEST_DT, 1);
    Serial.println(F("%"));
    Serial.print(F("latenza media (us): "));
    Serial.println((float)tot / N_TEST_DT, 1);
    Serial.print(F("min/max (us): "));
    Serial.print(tmin); Serial.print(F(" / ")); Serial.println(tmax);
}

void loop() {}
