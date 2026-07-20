/*
 * main_harness_12k.cpp — harness host, valutazione end-to-end 12k sample
 * Preprocessing: log1p(SKEW) + QT sklearn (formula bidirezionale + norm.ppf Acklam) + clip ±3.5
 * Forward: LUT int16 pubblicate (eval_l1, tanh_lut, eval_l2) con input clippato a [-1,1]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../mcu/kan_ml_layer1.h"
#include "../mcu/kan_ml_layer2.h"
#include "../mcu/kan_ml_tanh.h"
#include "kan_ml_prep.h"

#define L1_RD(e,i) (KANML_L1[(e)][(i)])
#define L2_RD(e,i) (KANML_L2[(e)][(i)])
#define TANH_RD(i) (KANML_TANH[(i)])
#define FP_ONE (1L<<16)

/* ── norm.ppf  (Acklam rational approx, max err ~5e-5) ── */
static double norm_ppf(double p) {
    static const double a[]={-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00};
    static const double b[]={-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01};
    static const double c[]={-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
    static const double d[]={7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00};
    if(p<=0.0) return -7.0;
    if(p>=1.0) return  7.0;
    double q,r;
    if(p<0.02425){
        q=sqrt(-2.0*log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if(p<=0.97575){
        q=p-0.5; r=q*q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q=sqrt(-2.0*log(1.0-p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

/* ── interpolazione lineare ── */
static double interp_fwd(double x, const double *xp, const double *fp, int n) {
    if(x<=xp[0]) return fp[0];
    if(x>=xp[n-1]) return fp[n-1];
    int lo=0,hi=n-1;
    while(hi-lo>1){ int mid=(lo+hi)>>1; if(xp[mid]<=x+1e-14)lo=mid; else hi=mid; }
    double t=(x-xp[lo])/(xp[hi]-xp[lo]);
    return fp[lo]+t*(fp[hi]-fp[lo]);
}
/* np.interp(-x, -xp[::-1], -fp[::-1]) */
static double interp_rev(double x, const double *xp, const double *fp, int n) {
    double nx=-x;
    if(nx<=-xp[n-1]) return -fp[n-1];
    if(nx>=-xp[0])   return -fp[0];
    int lo=0,hi=n-1;
    while(hi-lo>1){ int mid=(lo+hi)>>1; if(-xp[n-1-mid]<=nx)lo=mid; else hi=mid; }
    double x0=-xp[n-1-lo], x1=-xp[n-1-hi];
    double f0=-fp[n-1-lo], f1=-fp[n-1-hi];
    return f0+(nx-x0)/(x1-x0)*(f1-f0);
}

/* ── preprocessing: raw -> QT (sklearn exact) -> clip ±3.5 ── */
static void prep_one(const double *raw, double *out) {
    for(int j=0;j<PREP_K;j++){
        double v=raw[j];
        if(PREP_SKEW[j]) v=log1p(v<0.0?0.0:v);
        double lb=PREP_KNOTS_X[j][0], ub=PREP_KNOTS_X[j][PREP_NKNOTS-1];
        double qt;
        if(v-PREP_BOUNDS_THR<lb)      qt=-PREP_CLIP;
        else if(v+PREP_BOUNDS_THR>ub) qt= PREP_CLIP;
        else {
            double rf=interp_fwd(v, PREP_KNOTS_X[j], PREP_REFS, PREP_NKNOTS);
            double rr=interp_rev(v, PREP_KNOTS_X[j], PREP_REFS, PREP_NKNOTS);
            qt=norm_ppf(0.5*(rf-rr));
        }
        if(qt> PREP_CLIP) qt= PREP_CLIP;
        if(qt<-PREP_CLIP) qt=-PREP_CLIP;
        out[j]=qt;   /* in [-3.5, 3.5] — il layer1 clippa a [-1,1] */
    }
}

/* ── eval layer1: input Q16.16 = (x+1)*2^16 con x=clip(qt,-1,1) ── */
static inline int32_t eval_l1(int e, int32_t xq) {
    int32_t full=2*FP_ONE, off=xq<0?0:xq>full-1?full-1:xq;
    int seg=(int)(((int64_t)off*KANML_KSEG)/full); if(seg>=KANML_KSEG)seg=KANML_KSEG-1;
    int64_t num=(int64_t)off*KANML_KSEG-(int64_t)seg*full;
    int64_t p256=(num*(KANML_L-1)*256)/full;
    int r0=(int)(p256>>8); if(r0>=KANML_L-1)r0=KANML_L-2;
    int32_t fr=(int32_t)(p256-(r0<<8)); if(fr>256)fr=256;
    int base=seg*KANML_L+r0;
    return (int32_t)L1_RD(e,base)+(((int32_t)(L1_RD(e,base+1)-L1_RD(e,base))*fr)>>8);
}
static inline int32_t eval_l2(int e, int32_t xs) {
    int32_t full=2*KANML_S2, off=xs+KANML_S2; if(off<0)off=0; if(off>full-1)off=full-1;
    int seg=(int)(((int64_t)off*KANML_KSEG)/full); if(seg>=KANML_KSEG)seg=KANML_KSEG-1;
    int64_t num=(int64_t)off*KANML_KSEG-(int64_t)seg*full;
    int64_t p256=(num*(KANML_L-1)*256)/full;
    int r0=(int)(p256>>8); if(r0>=KANML_L-1)r0=KANML_L-2;
    int32_t fr=(int32_t)(p256-(r0<<8)); if(fr>256)fr=256;
    int base=seg*KANML_L+r0;
    return (int32_t)L2_RD(e,base)+(((int32_t)(L2_RD(e,base+1)-L2_RD(e,base))*fr)>>8);
}
static inline int32_t tanh_lut(int32_t h) {
    int32_t v=h+KANML_HMAX, span=2*KANML_HMAX;
    if(v<0)v=0; if(v>span)v=span;
    int idx=(int)(((int64_t)v*(KANML_TL-1))/span); if(idx>=KANML_TL)idx=KANML_TL-1;
    return TANH_RD(idx);
}
static int kan_ml_predict(const int32_t *xq, int32_t *logits) {
    int32_t hp[KANML_HIDDEN]={};
    for(int i=0;i<KANML_INDIM;i++) for(int j=0;j<KANML_HIDDEN;j++) hp[j]+=eval_l1(i*KANML_HIDDEN+j,xq[i]);
    int32_t hpost[KANML_HIDDEN]; for(int j=0;j<KANML_HIDDEN;j++) hpost[j]=tanh_lut(hp[j]);
    for(int c=0;c<KANML_C;c++) logits[c]=0;
    for(int j=0;j<KANML_HIDDEN;j++){
        int32_t hj=hpost[j]; if(hj<-KANML_S2)hj=-KANML_S2; if(hj>KANML_S2)hj=KANML_S2;
        for(int c=0;c<KANML_C;c++) logits[c]+=eval_l2(j*KANML_C+c,hj);
    }
    int best=0; for(int c=1;c<KANML_C;c++) if(logits[c]>logits[best]) best=c;
    return best;
}

/* clip qt (in [-3.5,3.5]) a [-1,1] -> Q16.16 */
static int32_t to_q16(double qt) {
    double x = qt < -1.0 ? -1.0 : qt > 1.0 ? 1.0 : qt;
    return (int32_t)((x + 1.0) * FP_ONE + 0.5);
}

int main(int argc, char **argv) {
    const char *bin=argc>1?argv[1]:"test_e2e_12k.bin";
    FILE *fp=fopen(bin,"rb"); if(!fp){fprintf(stderr,"Cannot open %s\n",bin);return 1;}
    int N; if(fread(&N,sizeof(int),1,fp)!=1){fclose(fp);return 1;}
    fprintf(stderr,"Leggo %d sample da %s\n",N,bin);

    int C=KANML_C;
    long *tp=(long*)calloc(C,sizeof(long)), *fp2=(long*)calloc(C,sizeof(long)), *fn=(long*)calloc(C,sizeof(long));
    long correct=0;
    double raw[PREP_K], qt_out[PREP_K]; int32_t xq[KANML_INDIM], logits[KANML_C];

    for(int n=0;n<N;n++){
        if(fread(raw,sizeof(double),PREP_K,fp)!=(size_t)PREP_K) break;
        int label=0; if(fread(&label,sizeof(int),1,fp)!=1) break;
        prep_one(raw, qt_out);
        for(int j=0;j<PREP_K;j++) xq[j]=to_q16(qt_out[j]);
        int pred=kan_ml_predict(xq,logits);
        if(pred==label) correct++;
        if(pred==label) tp[label]++; else{fp2[pred]++;fn[label]++;}
    }
    fclose(fp);

    double f1s=0;
    for(int c=0;c<C;c++){
        double pr=(tp[c]+fp2[c])>0?(double)tp[c]/(tp[c]+fp2[c]):0;
        double rc=(tp[c]+fn[c])>0?(double)tp[c]/(tp[c]+fn[c]):0;
        f1s+=(pr+rc)>0?2*pr*rc/(pr+rc):0;
    }
    printf("RISULTATI SUI %d SAMPLE:\n",N);
    printf("  Accuracy:  %.4f  (%ld/%d)\n",(double)correct/N,correct,N);
    printf("  Macro-F1:  %.4f\n",f1s/C);
    free(tp);free(fp2);free(fn);
    return 0;
}
