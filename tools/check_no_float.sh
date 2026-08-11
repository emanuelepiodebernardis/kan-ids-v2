#!/usr/bin/env bash
# Verifica che un file assembly non contenga ARITMETICA in virgola mobile.
#
# Distinzione importante: il compilatore usa i registri SSE (pxor, movups,
# movaps) anche per azzerare o copiare array di interi in blocco. Quelle
# sono istruzioni di movimento dati, non operazioni floating point, e non
# implicano una FPU sul target. Cio' che va escluso e' l'aritmetica reale
# e le conversioni intero<->reale.
set -u
f="$1"
FP='\b(adds[sd]|subs[sd]|muls[sd]|divs[sd]|sqrts[sd]|maxs[sd]|mins[sd]|comis[sd]|ucomis[sd]|cvtsi2s[sd]|cvtts[sd]2si|cvts[sd]2s[sd]|fadd|fsub|fmul|fdiv|fld[a-z]*|fst[a-z]*|fsqrt|fprem)\b'
n=$(grep -cE "$FP" "$f" || true)
echo "aritmetica FP in $(basename "$f"): $n"
if [ "$n" -gt 0 ]; then grep -nE "$FP" "$f" | head -10; exit 1; fi
