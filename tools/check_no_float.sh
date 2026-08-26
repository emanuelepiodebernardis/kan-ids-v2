#!/usr/bin/env bash
# Comodita' per la riga di comando: la verifica vera e' in
# tools/check_no_float.py, che e' l'unica copia della regola e funziona
# anche dove bash non c'e' (su Windows `bash` puo' risolvere a una WSL non
# installata, e la suite falliva li' per ragioni estranee alla virgola
# mobile). Stesso codice di uscita: 0 pulito, 1 virgola mobile trovata,
# 2 file assente.
set -u
exec "${PYTHON:-python3}" "$(dirname "$0")/check_no_float.py" "$@"
