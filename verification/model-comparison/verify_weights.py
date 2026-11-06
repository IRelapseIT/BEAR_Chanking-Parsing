"""Verify only already-present local model bytes against pinned Hugging Face LFS hashes."""
import hashlib
import json
from pathlib import Path

import httpx

ROOT=Path(__file__).resolve().parent


def main():
    models=[('BAAI/bge-m3','5617a9f61b028005a4858fdac845db406aefb181',Path('D:/CODEX/BEAR/BEAR_BGE_M3/models/bge-m3')),
            ('Qwen/Qwen3-Embedding-0.6B','97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',Path(Path('D:/CODEX/BEAR/review/native/model-path.txt').read_text().strip()))]
    result=[]
    with httpx.Client(timeout=45,follow_redirects=True) as client:
        for repo,rev,folder in models:
            url=f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true'
            response=client.get(url); response.raise_for_status(); info=response.json()
            for item in info['siblings']:
                filename=item['rfilename']
                if '/' in filename or not filename.endswith(('.safetensors','.bin')):
                    continue
                path=folder/filename
                if not path.exists(): continue
                expected=item.get('lfs',{}).get('sha256')
                if not expected: raise ValueError('No authoritative weight checksum')
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for data in iter(lambda:stream.read(4*1024**2),b''):
                        digest.update(data)
                actual=digest.hexdigest()
                if actual!=expected: raise ValueError('Weights checksum mismatch: '+str(path))
                result.append({'model':repo,'revision':rev,'path':str(path),'bytes':path.stat().st_size,
                               'sha256':actual,'expected_sha256':expected,'verified':True,'metadata_url':url})
    (ROOT/'weights-verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
