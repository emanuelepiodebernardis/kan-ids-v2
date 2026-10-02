"""Ricostruzione delle impronte degli input gia' visti, per la misura dell'esposizione.

Perche' serve
-------------
Per scegliere quale intervallo temporale riservare alla valutazione finale occorre
sapere quanta parte di ciascun intervallo era gia' stata vista attraverso il
campione usato nel lavoro precedente. La misura corrispondente nel codice di
riferimento e' la fase `known_input_overlap`, che confronta l'impronta di ogni
riga con un archivio di impronte note.

Quell'archivio non e' pubblicato. Il README di ids-update-lab dichiara che i CSV
grezzi di TON e gli array NPZ campionati non vengono redistribuiti e che gli
array di input mancanti vanno ricostruiti da sorgenti ottenute lecitamente;
release/provenance/REPRODUCTION_GAPS036.json elenca fra le dipendenze escluse
artifacts/known_input_fingerprints.npz, il suo .sha256 ed external_protocol.json.
La ricostruzione e' quindi la procedura prevista, non un espediente.

Come
----
Non si riscrive nulla: si importano e si chiamano le funzioni pubblicate
`normalize_raw` e `fingerprint_rows` di research028/vendor/audit_ton_full.py, le
stesse che l'audit usa sulle righe di TON. Cosi' le impronte note e quelle
calcolate sui dati nuovi nascono dalla stessa funzione, che e' la condizione
perche' il confronto abbia senso.

L'uscita e' un .npz con un array `fingerprints` di tipo V32 **ordinato**, come
`audit_csv` pretende, piu' i metadati di provenienza.

Limite dichiarato
-----------------
Il .npz originale non e' pubblicato, quindi il suo SHA-256
(6e6559503b11733bcf6b973ad2371b0f0b52022927f9769b7411cef507cdb3e2) non e'
riproducibile e lo script frozen020/tools/evaluate_external.py non e' eseguibile
cosi' com'e'. Questa e' una ricostruzione dichiarata, il cui ancoraggio
verificabile e' lo SHA-256 della sorgente.

Uso
---
    python impronte_note.py --sorgente train_test_network.csv \
        --audit .../research028/vendor/audit_ton_full.py \
        --uscita impronte_note.npz
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SORGENTE_ATTESA = '26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974'
NPZ_ORIGINALE_NON_PUBBLICATO = '6e6559503b11733bcf6b973ad2371b0f0b52022927f9769b7411cef507cdb3e2'
LOTTO = 50000

# Metadati che load_known() di audit_ton_full.py pretende dall'archivio originale.
# Sono l'unico riscontro numerico disponibile sulla ricostruzione: se coincidono,
# la ricostruzione ha la stessa cardinalita' e la stessa sorgente dell'originale.
IMPRONTE_UNICHE_ATTESE = 92330
RIGHE_SORGENTE_ATTESE = 211043


def carica_audit(percorso):
    """Importa il modulo pubblicato senza copiarne il codice."""
    percorso = Path(percorso)
    if not percorso.is_file():
        raise SystemExit('audit_ton_full.py non trovato: ' + str(percorso))
    spec = importlib.util.spec_from_file_location('audit_ton_full', percorso)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules['audit_ton_full'] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def sha256_file(percorso, blocco=8 << 20):
    h = hashlib.sha256()
    with Path(percorso).open('rb') as f:
        for pezzo in iter(lambda: f.read(blocco), b''):
            h.update(pezzo)
    return h.hexdigest()


def costruisci(sorgente, audit):
    """Scorre il CSV a lotti e restituisce le impronte delle righe valide."""
    csv.field_size_limit(16 << 20)
    impronte = []
    conta = {'righe': 0, 'valide': 0, 'invalide': 0, 'zeri_negativi_canonicalizzati': 0}
    motivi = {nome: {} for nome in audit.FEATURES}
    with Path(sorgente).open(encoding='utf-8-sig', newline='') as f:
        lettore = csv.reader(f, strict=True)
        header = next(lettore, None)
        if not header or len(header) != len(set(header)):
            raise SystemExit('header CSV mancante o con nomi duplicati')
        mancanti = [n for n in audit.FEATURES if n not in header]
        if mancanti:
            raise SystemExit('colonne richieste assenti: ' + ', '.join(mancanti))
        indici = [header.index(n) for n in audit.FEATURES]
        lotto = []
        for riga in lettore:
            if len(riga) != len(header):
                raise SystemExit('riga con un numero di campi diverso dall header')
            lotto.append([riga[i] for i in indici])
            if len(lotto) >= LOTTO:
                _consuma(lotto, audit, impronte, conta, motivi)
                lotto = []
        if lotto:
            _consuma(lotto, audit, impronte, conta, motivi)
    return impronte, conta, motivi


def _consuma(lotto, audit, impronte, conta, motivi):
    valori, validi, ragioni, zeri = audit.normalize_raw(lotto)
    conta['righe'] += len(lotto)
    conta['valide'] += int(validi.sum())
    conta['invalide'] += int((~validi).sum())
    conta['zeri_negativi_canonicalizzati'] += int(zeri)
    for j, nome in enumerate(audit.FEATURES):
        for codice, ragione in audit.REASONS.items():
            n = int((ragioni[:, j] == codice).sum())
            if n:
                motivi[nome][ragione] = motivi[nome].get(ragione, 0) + n
    if validi.any():
        impronte.append(audit.fingerprint_rows(valori[validi]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sorgente', required=True, help='train_test_network.csv')
    p.add_argument('--audit', required=True,
                   help='percorso di research028/vendor/audit_ton_full.py')
    p.add_argument('--uscita', required=True, help='file .npz da scrivere')
    p.add_argument('--esigi-sorgente', action='store_true',
                   help="esce diverso da zero se lo SHA-256 della sorgente non e' quello atteso")
    a = p.parse_args(argv)

    audit = carica_audit(a.audit)
    if list(audit.FEATURES) != ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes',
                                'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes']:
        raise SystemExit('FEATURES del modulo pubblicato diverse da quelle attese')

    sorgente = Path(a.sorgente)
    if not sorgente.is_file():
        raise SystemExit('sorgente inesistente: ' + str(sorgente))
    digest = sha256_file(sorgente)
    coincide = digest == SORGENTE_ATTESA
    print('sorgente          ' + sorgente.name)
    print('SHA-256           ' + digest)
    print('atteso dall audit ' + SORGENTE_ATTESA)
    print('coincidenza       ' + ('SI' if coincide else 'NO'))
    if a.esigi_sorgente and not coincide:
        print("la sorgente non e' quella attesa dall'audit pubblicato", file=sys.stderr)
        return 1

    pezzi, conta, motivi = costruisci(sorgente, audit)
    if not pezzi:
        print('nessuna riga valida', file=sys.stderr)
        return 1
    impronte = np.concatenate(pezzi)
    # L'archivio originale contiene le impronte UNICHE e ordinate: load_known()
    # pretende len(unique(known)) == len(known). Si adotta la stessa forma.
    ordinate = np.unique(impronte)
    distinte = int(len(ordinate))

    # Riscontro contro i metadati che load_known() dichiara dell'originale.
    riscontro = {
        'impronte_uniche': distinte,
        'impronte_uniche_attese': IMPRONTE_UNICHE_ATTESE,
        'righe_sorgente': conta['righe'],
        'righe_sorgente_attese': RIGHE_SORGENTE_ATTESE,
    }
    riscontro['coincide'] = (distinte == IMPRONTE_UNICHE_ATTESE
                             and conta['righe'] == RIGHE_SORGENTE_ATTESE
                             and coincide)
    print()
    print('riscontro sui metadati dichiarati dall originale')
    print('  impronte uniche   %d   atteso %d' % (distinte, IMPRONTE_UNICHE_ATTESE))
    print('  righe sorgente    %d   atteso %d' % (conta['righe'], RIGHE_SORGENTE_ATTESE))
    print('  esito             ' + ('COINCIDE' if riscontro['coincide'] else 'NON COINCIDE'))
    print()
    if a.esigi_sorgente and not riscontro['coincide']:
        print('la ricostruzione non riproduce i metadati dichiarati', file=sys.stderr)
        return 1

    uscita = Path(a.uscita)
    uscita.parent.mkdir(parents=True, exist_ok=True)
    np.savez(uscita, fingerprints=ordinate,
             source_sha256=np.array(digest),
             source_rows=np.array(conta['righe']),
             feature_names=np.array(list(audit.FEATURES)))

    rapporto = {
        'sorgente': sorgente.name,
        'sorgente_sha256': digest,
        'sorgente_sha256_atteso': SORGENTE_ATTESA,
        'sorgente_coincide': coincide,
        'feature_names': list(audit.FEATURES),
        'funzioni_riusate': ['normalize_raw', 'fingerprint_rows'],
        'modulo_riusato': str(Path(a.audit).as_posix()),
        'modulo_riusato_sha256': sha256_file(a.audit),
        'conteggi': conta,
        'motivi_di_scarto_per_campo': {k: v for k, v in motivi.items() if v},
        'impronte_uniche': distinte,
        'riscontro_metadati_originale': riscontro,
        'ordinate': bool(np.array_equal(ordinate, np.sort(ordinate))),
        'npz_originale_non_pubblicato_sha256': NPZ_ORIGINALE_NON_PUBBLICATO,
        'nota': ("ricostruzione dichiarata: le impronte nascono dalle funzioni "
                 "pubblicate normalize_raw e fingerprint_rows, non da una "
                 "riscrittura; il .npz originale non e' pubblicato, quindi il suo "
                 "SHA-256 non e' riproducibile"),
    }
    rap = uscita.with_suffix('.json')
    rap.write_text(json.dumps(rapporto, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print('righe lette       %d' % conta['righe'])
    print('righe valide      %d' % conta['valide'])
    print('righe scartate    %d' % conta['invalide'])
    print('impronte distinte %d' % distinte)
    print('scritto           ' + str(uscita))
    print('rapporto          ' + str(rap))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
