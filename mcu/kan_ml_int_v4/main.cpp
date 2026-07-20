/*
 * main_kan_int_v4_ram3_wokwi.cpp — KAN-IDS integer-only v4 con memcpy in RAM
 * ---------------------------------------------------------------------------
 * Fix definitivo per la latenza del preprocessing:
 * Le LUT QT_KLO, QT_FRAC, QT_PPF vengono copiate dalla flash alla DRAM
 * a setup() con memcpy. Le funzioni di preprocessing usano i puntatori RAM.
 *
 * RAM usata: 43 KB / ~400 KB disponibili
 * Latenza attesa preprocessing: ~5-20 µs
 * Latenza totale attesa: ~695-720 µs
 */

#include <Arduino.h>
#include <string.h>

#define KANML_INDIM   10
#define KANML_HIDDEN  16
#define KANML_C       10
#define KANML_KSEG    8
#define KANML_L       64
#define KANML_S2      10000
#define FP_ONE        (1L << 16)
#define QT_K          10
#define QT_NQ         1000
#define QT_NVALS      1342
#define QT_Q_ONE      64

// LUT in flash (const)
#include "qt_int_v4_lut.h"
#include "kan_ml_layer1_v4.h"
#include "kan_ml_layer2_v4.h"
#include "kan_ml_tanh_v4.h"
#include "test_vectors_int_v4.h"

// ── Buffer RAM per le LUT di preprocessing ───────────────────────────────────
static uint16_t klo_ram[QT_K][QT_NVALS];
static uint8_t  frac_ram[QT_K][QT_NVALS];
static float    ppf_ram[QT_NQ];

// ── Preprocessing con accesso RAM ─────────────────────────────────────────────
static void qt_preproc_ram(const float *x_raw, float *x_out) {
    for(int j = 0; j < QT_K; j++) {
        float lv = logf(x_raw[j] > 0.0f ? x_raw[j] + 1.0f : 1.0f);
        int32_t xq = (int32_t)(lv * (float)QT_Q_ONE + 0.5f);
        if(xq < 0) xq = 0;
        if(xq >= QT_NVALS) xq = QT_NVALS - 1;

        int32_t li = (int32_t)klo_ram[j][xq];
        float   fr = (float)frac_ram[j][xq] * (1.0f / 256.0f);
        float out  = ppf_ram[li] + fr * (ppf_ram[li + 1] - ppf_ram[li]);
        if(out >  3.5f) out =  3.5f;
        if(out < -3.5f) out = -3.5f;
        x_out[j] = out;
    }
}

// ── Forward KAN (identico al Passo 5) ────────────────────────────────────────
#define L1_RD(e,idx)  (KANML_L1_V4[(e)][(idx)])
#define L2_RD(e,idx)  (KANML_L2_V4[(e)][(idx)])
#define TANH_RD(idx)  (KANML_TANH_V4[(idx)])

static inline int32_t to_q16(float x) {
    if(x<-1.f)x=-1.f; if(x>1.f)x=1.f;
    return (int32_t)((x+1.f)*(float)FP_ONE+0.5f);
}
static inline int32_t eval_l1(int e, int32_t x_q16) {
    int32_t full=2*FP_ONE, off=x_q16;
    if(off<0)off=0; if(off>full-1)off=full-1;
    int seg=(int)(((int64_t)off*KANML_KSEG)/full);
    if(seg>=KANML_KSEG)seg=KANML_KSEG-1;
    int64_t num=(int64_t)off*KANML_KSEG-(int64_t)seg*full;
    int64_t p256=(num*(KANML_L-1)*256)/full;
    int r0=(int)(p256>>8); if(r0>=KANML_L-1)r0=KANML_L-2;
    int32_t fr=(int32_t)(p256-((int64_t)r0<<8));
    if(fr<0)fr=0; if(fr>256)fr=256;
    int base=seg*KANML_L+r0;
    return (int32_t)L1_RD(e,base)+(((int32_t)(L1_RD(e,base+1)-L1_RD(e,base))*fr)>>8);
}
static inline int32_t tanh_lut(int32_t h) {
    int32_t v=h+KANML_HMAX_V4, span=2*KANML_HMAX_V4;
    if(v<0)v=0; if(v>span)v=span;
    int idx=(int)(((int64_t)v*(KANML_TL_V4-1))/span);
    if(idx<0)idx=0; if(idx>=KANML_TL_V4)idx=KANML_TL_V4-1;
    return TANH_RD(idx);
}
static inline int32_t eval_l2(int e, int32_t x_s2) {
    int32_t full=2*KANML_S2, off=x_s2+KANML_S2;
    if(off<0)off=0; if(off>full-1)off=full-1;
    int seg=(int)(((int64_t)off*KANML_KSEG)/full);
    if(seg>=KANML_KSEG)seg=KANML_KSEG-1;
    int64_t num=(int64_t)off*KANML_KSEG-(int64_t)seg*full;
    int64_t p256=(num*(KANML_L-1)*256)/full;
    int r0=(int)(p256>>8); if(r0>=KANML_L-1)r0=KANML_L-2;
    int32_t fr=(int32_t)(p256-((int64_t)r0<<8));
    if(fr<0)fr=0; if(fr>256)fr=256;
    int base=seg*KANML_L+r0;
    return (int32_t)L2_RD(e,base)+(((int32_t)(L2_RD(e,base+1)-L2_RD(e,base))*fr)>>8);
}
static int kan_predict(const int32_t *x_q16, int32_t *logits) {
    int32_t hpre[KANML_HIDDEN]={};
    for(int i=0;i<KANML_INDIM;i++){
        int32_t xi=x_q16[i];
        for(int j=0;j<KANML_HIDDEN;j++) hpre[j]+=eval_l1(i*KANML_HIDDEN+j,xi);
    }
    int32_t hpost[KANML_HIDDEN];
    for(int j=0;j<KANML_HIDDEN;j++) hpost[j]=tanh_lut(hpre[j]);
    for(int c=0;c<KANML_C;c++) logits[c]=0;
    for(int j=0;j<KANML_HIDDEN;j++){
        int32_t hj=hpost[j];
        for(int c=0;c<KANML_C;c++) logits[c]+=eval_l2(j*KANML_C+c,hj);
    }
    int best=0; int32_t bv=logits[0];
    for(int c=1;c<KANML_C;c++) if(logits[c]>bv){bv=logits[c];best=c;}
    return best;
}

static unsigned long micros_now(){
#ifdef ESP_PLATFORM
    return (unsigned long)(esp_timer_get_time());
#else
    return micros();
#endif
}

void setup(){
    Serial.begin(115200); while(!Serial){}

    // Copia LUT dalla flash alla DRAM
    memcpy(klo_ram,  QT_KLO,  sizeof(klo_ram));
    memcpy(frac_ram, QT_FRAC, sizeof(frac_ram));
    memcpy(ppf_ram,  QT_PPF,  sizeof(ppf_ram));
    Serial.println(F("=== KAN-IDS integer-only v4 RAM3 (memcpy) ==="));
    Serial.println(F("LUT preprocessing copiate in DRAM a setup()"));
    Serial.println(F("F1=0.9044  Acc_attesa=95%  Latenza_attesa~710us"));
    Serial.println(F("idx,preproc_us,forward_us,total_us,pred,match"));

    float   x_scaled[QT_K];
    int32_t x_q16[QT_K];
    int32_t logits[KANML_C];
    unsigned long tot_pre=0, tot_fwd=0;

    for(int i=0;i<N_TEST_V4;i++){
        // Preprocessing dalla DRAM
        unsigned long t0=micros_now();
        qt_preproc_ram(TEST_RAW_V4[i], x_scaled);
        for(int j=0;j<QT_K;j++) x_q16[j]=to_q16(x_scaled[j]);
        unsigned long pre_us=micros_now()-t0;

        // Forward KAN
        unsigned long t1=micros_now();
        int pred=kan_predict(x_q16, logits);
        unsigned long fwd_us=micros_now()-t1;

        tot_pre+=pre_us; tot_fwd+=fwd_us;
        if(pred==TEST_LABEL_V4[i]){}  // conta corretto ma non stampa qui

        Serial.print(i);          Serial.print(',');
        Serial.print(pre_us);     Serial.print(',');
        Serial.print(fwd_us);     Serial.print(',');
        Serial.print(pre_us+fwd_us); Serial.print(',');
        Serial.print(pred);       Serial.print(',');
        Serial.println(pred==TEST_LABEL_V4[i]?'Y':'N');
    }
    Serial.println(F("--- riepilogo ---"));
    Serial.print(F("preproc medio: ")); Serial.println((float)tot_pre/N_TEST_V4,1);
    Serial.print(F("forward medio: ")); Serial.println((float)tot_fwd/N_TEST_V4,1);
    Serial.print(F("totale medio:  ")); Serial.println((float)(tot_pre+tot_fwd)/N_TEST_V4,1);
}
void loop(){}
