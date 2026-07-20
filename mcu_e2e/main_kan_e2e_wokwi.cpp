/*
 * main_kan_e2e_wokwi.cpp  — KAN-IDS end-to-end con preprocessing on-chip
 * -------------------------------------------------------------------------
 * Catena completa (dati grezzi → predizione):
 *   raw feature  →  log1p (SKEW)  →  QT bidirezionale (sklearn-compatible)
 *   →  clip ±3.5  →  clip→[-1,1]  →  eval_l1 (Q16.16)  →  tanh_lut
 *   →  eval_l2 (scala S2)  →  argmax  →  classe
 *
 * Headers necessari (in mcu/ e mcu_e2e/):
 *   kan_ml_layer1.h, kan_ml_layer2.h, kan_ml_tanh.h  — LUT modello
 *   kan_ml_prep.h                                      — knot QT + refs
 *   test_vectors_e2e.h                                 — 40 vettori raw
 *
 * Target: ESP32-C3 (4MB flash, 400KB SRAM). Wokwi-ready.
 * Macro-F1 atteso: 0.9118 (su 40 vettori di test bilanciati).
 */

#include <Arduino.h>
#include "kan_ml_layer1.h"
#include "kan_ml_layer2.h"
#include "kan_ml_tanh.h"
#include "kan_ml_prep.h"
#include "test_vectors_e2e.h"

#ifdef __AVR__
  #define L1_RD(e,i) ((int16_t)pgm_read_word(&KANML_L1[(e)][(i)]))
  #define L2_RD(e,i) ((int16_t)pgm_read_word(&KANML_L2[(e)][(i)]))
  #define TANH_RD(i) ((int16_t)pgm_read_word(&KANML_TANH[(i)]))
#else
  #define L1_RD(e,i) (KANML_L1[(e)][(i)])
  #define L2_RD(e,i) (KANML_L2[(e)][(i)])
  #define TANH_RD(i) (KANML_TANH[(i)])
#endif

#define FP_ONE (1L<<16)

/* ── norm.ppf  (Acklam rational approx, max err ~5e-5) ── */
static double norm_ppf(double p) {
  static const double a[]={-3.969683028665376e+01,2.209460984245205e+02,
    -2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,
     2.506628277459239e+00};
  static const double b[]={-5.447609879822406e+01,1.615858368580409e+02,
    -1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01};
  static const double c[]={-7.784894002430293e-03,-3.223964580411365e-01,
    -2.400758277161838e+00,-2.549732539343734e+00, 4.374664141464968e+00,
     2.938163982698783e+00};
  static const double d[]={7.784695709041462e-03,3.224671290700398e-01,
     2.445134137142996e+00,3.754408661907416e+00};
  if (p<=0.0) return -7.0;
  if (p>=1.0) return  7.0;
  double q,r;
  if (p<0.02425) {
    q=sqrt(-2.0*log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
  } else if (p<=0.97575) {
    q=p-0.5; r=q*q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
  } else {
    q=sqrt(-2.0*log(1.0-p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
  }
}

/* Interpolazione lineare (np.interp equivalente, xp crescente) */
#define INTERP_EPS 1e-14
static double interp_fwd(double x, const double *xp, const double *fp, int n) {
  if (x<=xp[0])   return fp[0];
  if (x>=xp[n-1]) return fp[n-1];
  int lo=0,hi=n-1;
  while (hi-lo>1) {
    int mid=(lo+hi)>>1;
    if (xp[mid]<=x+INTERP_EPS) lo=mid; else hi=mid;
  }
  double t=(xp[hi]!=xp[lo])?(x-xp[lo])/(xp[hi]-xp[lo]):0.0;
  return fp[lo]+t*(fp[hi]-fp[lo]);
}

/* np.interp(-x, -xp[::-1], -fp[::-1]) */
static double interp_rev(double x, const double *xp, const double *fp, int n) {
  double nx=-x;
  if (nx<=-xp[n-1]) return -fp[n-1];
  if (nx>=-xp[0])   return -fp[0];
  int lo=0,hi=n-1;
  while (hi-lo>1) {
    int mid=(lo+hi)>>1;
    if (-xp[n-1-mid]<=nx) lo=mid; else hi=mid;
  }
  double x0=-xp[n-1-lo], x1=-xp[n-1-hi];
  double f0=-fp[n-1-lo],  f1=-fp[n-1-hi];
  return f0+(nx-x0)/(x1-x0)*(f1-f0);
}

/* Preprocessing: raw → QT (sklearn-compatible) → clip ±PREP_CLIP */
static void prep_one(const double *raw, double *out) {
  for (int j=0;j<PREP_K;j++) {
    double v=raw[j];
    if (PREP_SKEW[j]) v=log1p(v<0.0?0.0:v);
    double lb=PREP_KNOTS_X[j][0], ub=PREP_KNOTS_X[j][PREP_NKNOTS-1];
    double qt;
    if (v-PREP_BOUNDS_THR<lb)      qt=-PREP_CLIP;
    else if (v+PREP_BOUNDS_THR>ub) qt= PREP_CLIP;
    else {
      double rf=interp_fwd(v, PREP_KNOTS_X[j], PREP_REFS, PREP_NKNOTS);
      double rr=interp_rev(v, PREP_KNOTS_X[j], PREP_REFS, PREP_NKNOTS);
      qt=norm_ppf(0.5*(rf-rr));
    }
    if (qt> PREP_CLIP) qt= PREP_CLIP;
    if (qt<-PREP_CLIP) qt=-PREP_CLIP;
    out[j]=qt;   /* in [-3.5, 3.5]; il layer1 clippa a [-1,1] */
  }
}

/* to_q16: clip a [-1,1] → Q16.16 = (x+1)*2^16 */
static int32_t to_q16(double x) {
  if (x<-1.0) x=-1.0;
  if (x> 1.0) x= 1.0;
  return (int32_t)((x+1.0)*(double)FP_ONE+0.5);
}

/* ── Layer1 ── */
static inline int32_t eval_l1(int e, int32_t xq) {
  int32_t full=2*(int32_t)FP_ONE;
  int32_t off=xq<0?0:(xq>full-1?full-1:xq);
  int seg=(int)(((int64_t)off*KANML_KSEG)/full);
  if(seg>=KANML_KSEG) seg=KANML_KSEG-1;
  int64_t num=(int64_t)off*KANML_KSEG-(int64_t)seg*full;
  int64_t p256=(num*(KANML_L-1)*256)/full;
  int r0=(int)(p256>>8); if(r0>=KANML_L-1) r0=KANML_L-2;
  int32_t fr=(int32_t)(p256-((int64_t)r0<<8));
  if(fr<0)fr=0; if(fr>256)fr=256;
  int base=seg*KANML_L+r0;
  int16_t v0=L1_RD(e,base), v1=L1_RD(e,base+1);
  return (int32_t)v0+(((int32_t)(v1-v0)*fr)>>8);
}

/* ── tanh-LUT ── */
static inline int32_t tanh_lut(int32_t h) {
  int32_t v=h+KANML_HMAX, span=2*KANML_HMAX;
  if(v<0)v=0; if(v>span)v=span;
  int idx=(int)(((int64_t)v*(KANML_TL-1))/span);
  if(idx>=KANML_TL) idx=KANML_TL-1;
  return TANH_RD(idx);
}

/* ── Layer2 ── */
static inline int32_t eval_l2(int e, int32_t xs) {
  int32_t full=2*KANML_S2;
  int32_t off=xs+KANML_S2; if(off<0)off=0; if(off>full-1)off=full-1;
  int seg=(int)(((int64_t)off*KANML_KSEG)/full);
  if(seg>=KANML_KSEG) seg=KANML_KSEG-1;
  int64_t num=(int64_t)off*KANML_KSEG-(int64_t)seg*full;
  int64_t p256=(num*(KANML_L-1)*256)/full;
  int r0=(int)(p256>>8); if(r0>=KANML_L-1) r0=KANML_L-2;
  int32_t fr=(int32_t)(p256-((int64_t)r0<<8));
  if(fr<0)fr=0; if(fr>256)fr=256;
  int base=seg*KANML_L+r0;
  int16_t v0=L2_RD(e,base), v1=L2_RD(e,base+1);
  return (int32_t)v0+(((int32_t)(v1-v0)*fr)>>8);
}

/* ── Predizione end-to-end ── */
static int kan_e2e_predict(const double *raw_feat, int32_t *logits) {
  /* 1. Preprocessing */
  double qt[PREP_K];
  prep_one(raw_feat, qt);

  /* 2. to_q16 */
  int32_t xq[KANML_INDIM];
  for (int j=0;j<KANML_INDIM;j++) xq[j]=to_q16(qt[j]);

  /* 3. Layer 1 */
  int32_t hp[KANML_HIDDEN];
  for (int j=0;j<KANML_HIDDEN;j++) hp[j]=0;
  for (int i=0;i<KANML_INDIM;i++)
    for (int j=0;j<KANML_HIDDEN;j++)
      hp[j]+=eval_l1(i*KANML_HIDDEN+j, xq[i]);

  /* 4. tanh */
  int32_t hpost[KANML_HIDDEN];
  for (int j=0;j<KANML_HIDDEN;j++) hpost[j]=tanh_lut(hp[j]);

  /* 5. Layer 2 */
  for (int c=0;c<KANML_C;c++) logits[c]=0;
  for (int j=0;j<KANML_HIDDEN;j++) {
    int32_t hj=hpost[j];
    if(hj<-KANML_S2) hj=-KANML_S2;
    if(hj> KANML_S2) hj= KANML_S2;
    for (int c=0;c<KANML_C;c++)
      logits[c]+=eval_l2(j*KANML_C+c, hj);
  }

  /* 6. argmax */
  int best=0;
  for (int c=1;c<KANML_C;c++) if(logits[c]>logits[best]) best=c;
  return best;
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

  Serial.println(F("=== KAN-IDS END-TO-END (preprocessing on-chip) ==="));
  Serial.print(F("features="));  Serial.print(PREP_K);
  Serial.print(F(" hidden="));   Serial.print(KANML_HIDDEN);
  Serial.print(F(" classes="));  Serial.println(KANML_C);
  Serial.println(F("idx,label,pred,match,latency_us"));

  int correct=0;
  unsigned long tot=0, tmin=0xFFFFFFFF, tmax=0;
  int32_t logits[KANML_C];

  for (int i=0;i<N_TEST_E2E;i++) {
    unsigned long t0=micros_now();
    int pred=kan_e2e_predict(TEST_RAW_E2E[i], logits);
    unsigned long us=micros_now()-t0;

    if (pred==TEST_LABEL_E2E[i]) correct++;
    tot+=us; if(us<tmin)tmin=us; if(us>tmax)tmax=us;

    Serial.print(i);             Serial.print(F(","));
    Serial.print(TEST_LABEL_E2E[i]); Serial.print(F(","));
    Serial.print(pred);          Serial.print(F(","));
    Serial.print(pred==TEST_LABEL_E2E[i]?'Y':'N'); Serial.print(F(","));
    Serial.println(us);
  }

  Serial.println(F("--- riepilogo ---"));
  Serial.print(F("accuratezza: "));
  Serial.print(100.0f*correct/N_TEST_E2E, 1);
  Serial.println(F("%"));
  Serial.print(F("latenza media (us): "));
  Serial.println((float)tot/N_TEST_E2E, 1);
  Serial.print(F("min/max (us): "));
  Serial.print(tmin); Serial.print(F(" / ")); Serial.println(tmax);
}

void loop() {}
