"""Download the two versioned public mirrors and verify the TON source hashes."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from prepare_data import sha, HERE

SOURCES={
    'ton':'https://www.kaggle.com/api/v1/datasets/download/mohammedaddoun/ton-iot?datasetVersionNumber=1',
    'unsw':'https://www.kaggle.com/api/v1/datasets/download/mrwellsdavid/unsw-nb15?datasetVersionNumber=1',
}


def fetch(dataset,out):
    out.mkdir(parents=True,exist_ok=True)
    archive=out/(dataset+'_v1.zip');temporary=archive.with_suffix('.download')
    if not archive.exists():
        with urllib.request.urlopen(SOURCES[dataset],timeout=120) as response, temporary.open('wb') as handle:
            shutil.copyfileobj(response,handle,8<<20)
        if not zipfile.is_zipfile(temporary):
            raise ValueError('Expected the public ZIP; a sign-in or service error may have been returned')
        temporary.replace(archive)
    raw=out/(dataset+'_raw');raw.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            dest=(raw/info.filename).resolve()
            if raw.resolve() not in dest.parents and dest!=raw.resolve():
                raise ValueError('Unsafe ZIP member')
        z.extractall(raw)
    files={p.name:sha(p) for p in sorted(raw.glob('*.csv'))}
    if dataset=='ton':
        expected=json.loads((HERE/'vendor/source_manifest.json').read_text())
        for item in expected['files']:
            assert files[item['file']]==item['local_sha256'],item['file']
    elif (HERE/'provenance/unsw/A.json').exists():
        expected=json.loads((HERE/'provenance/unsw/A.json').read_text())['source_sha256']
        for name,digest in expected.items():
            assert files[name]==digest,name
    record={'dataset':dataset,'url':SOURCES[dataset],'archive_sha256':sha(archive),
            'archive_bytes':archive.stat().st_size,'csv_sha256':files,
            'checksum_authority':'local mirror identity; not a publisher-signed checksum'}
    (out/(dataset+'_download_manifest.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(raw)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SOURCES)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();fetch(args.dataset,args.out)
