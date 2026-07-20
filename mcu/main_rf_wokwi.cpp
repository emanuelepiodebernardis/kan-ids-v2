/*
 * main_rf_wokwi.cpp — Random Forest multiclass su ESP32-C3 (Wokwi)
 * -----------------------------------------------------------------
 * Modello: RandomForest(100 alberi, max_depth=10), 10 classi, 10 feature
 * F1 (macro): 0.9446   Nodi: 48.776   Flash: ~667 KB
 *
 * Inferenza: traversal loop su array flat in flash,
 *            voto a maggioranza tra i 100 alberi.
 * Latenza: costante per costruzione (depth fissa = 10).
 */

#include <Arduino.h>
#define DT_N_FEATURES 10
#include "rf_mc_nodes.h"
#include "dt_test_vectors.h"

static int rf_predict(const float *x) {
    int votes[RF_N_CLASSES] = {};
    for (int t = 0; t < RF_N_TREES; t++) {
        int node = RF_TREE_OFFSET[t];
        while (RF_FEATURE[node] >= 0) {
            if (x[RF_FEATURE[node]] <= RF_THRESHOLD[node])
                node = RF_LEFT[node];
            else
                node = RF_RIGHT[node];
        }
        votes[(int)RF_LEAF_CLASS[node]]++;
    }
    int best = 0;
    for (int c = 1; c < RF_N_CLASSES; c++)
        if (votes[c] > votes[best]) best = c;
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

    Serial.println(F("=== Random Forest multiclass (100 alberi, depth=10) ==="));
    Serial.print(F("nodi=")); Serial.print(RF_N_NODES);
    Serial.print(F("  alberi=")); Serial.print(RF_N_TREES);
    Serial.print(F("  classi=")); Serial.println(RF_N_CLASSES);
    Serial.println(F("idx,label,pred,match,latency_us"));

    int correct = 0;
    unsigned long tot = 0, tmin = 0xFFFFFFFF, tmax = 0;

    for (int i = 0; i < N_TEST_DT; i++) {
        unsigned long t0 = micros_now();
        int pred = rf_predict(TEST_RAW_DT[i]);
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
