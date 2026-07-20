/*
 * main_dt_wokwi.cpp — Decision Tree multiclass su ESP32-C3 (Wokwi)
 * ------------------------------------------------------------------
 * Modello: DecisionTree(max_depth=15), 10 classi, 10 feature
 * F1 (macro): 0.9438  Nodi: 1055  Flash: ~10 KB
 *
 * Input: feature grezze (stessa pipeline del progetto, seed=42)
 * Preprocessing: NESSUNO (i tree sono invarianti a trasf. monotone)
 *
 * Inferenza: traversal array nodi in O(depth) = O(15) confronti
 */

#include <Arduino.h>
#include "dt_mc_nodes.h"
#include "dt_test_vectors.h"   // 40 vettori di test (feature grezze)

/* ── Inferenza ── */
static int dt_predict(const float *x) {
    int node = 0;
    while (DT_FEATURE[node] >= 0) {           // nodo interno
        if (x[DT_FEATURE[node]] <= DT_THRESHOLD[node])
            node = DT_LEFT[node];
        else
            node = DT_RIGHT[node];
    }
    return DT_LEAF_CLASS[node];               // foglia → classe
}

/* ── Wokwi timing ── */
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

    Serial.println(F("=== Decision Tree multiclass (depth=15) ==="));
    Serial.print(F("nodi=")); Serial.print(DT_N_NODES);
    Serial.print(F("  classi=")); Serial.print(DT_N_CLASSES);
    Serial.print(F("  feature=")); Serial.println(DT_N_FEATURES);
    Serial.println(F("idx,label,pred,match,latency_us"));

    int correct = 0;
    unsigned long tot = 0, tmin = 0xFFFFFFFF, tmax = 0;

    for (int i = 0; i < N_TEST_DT; i++) {
        unsigned long t0 = micros_now();
        int pred = dt_predict(TEST_RAW_DT[i]);
        unsigned long us = micros_now() - t0;

        if (pred == TEST_LABEL_DT[i]) correct++;
        tot += us;
        if (us < tmin) tmin = us;
        if (us > tmax) tmax = us;

        Serial.print(i);                      Serial.print(F(","));
        Serial.print(TEST_LABEL_DT[i]);       Serial.print(F(","));
        Serial.print(pred);                   Serial.print(F(","));
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
