"""Launch the shipped module, perform real HTTP inference, stop only owned processes."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

ROOT=Path(__file__).resolve().parent
PROJECT=Path('D:/CODEX/BEAR/BEAR_BGE_M3')


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    env=dict(os.environ)
    env['PYTHONPATH']=';'.join([str(PROJECT/'bear-text-beta/src'),'D:/CODEX/BEAR/upgrade/embedding-ml',env.get('PYTHONPATH','')])
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',PYTHONIOENCODING='utf-8')
    with (ROOT/'installed-server.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-m','bear_text_beta.local_inference','--model-dir',
            str(PROJECT/'models/bge-m3'),'--port',str(port)],env=env,cwd=PROJECT,stdout=log,stderr=log)
    try:
        base=f'http://127.0.0.1:{port}/v1'
        with httpx.Client(timeout=30,trust_env=False) as client:
            for _ in range(300):
                if process.poll() is not None: raise RuntimeError((ROOT/'installed-server.log').read_text('utf-8'))
                try:
                    response=client.get(base+'/models',timeout=.5)
                    if response.is_success: break
                except httpx.HTTPError: pass
                time.sleep(.2)
            else: raise RuntimeError('Installed model server did not become ready')
            metadata=response.json()
            assert metadata['data'][0]['id']=='BAAI/bge-m3' and metadata['data'][0]['pooling']=='cls'
            response=client.post(base+'/embeddings',json={'model':'BAAI/bge-m3','input':['Проверка локального сервера.']})
            response.raise_for_status(); vector=response.json()['data'][0]['embedding']
            assert len(vector)==1024 and abs(sum(v*v for v in vector)-1)<1e-5
            assert client.post(base+'/embeddings',json={'model':'Qwen/Qwen3-Embedding-0.6B','input':['test']}).status_code==422
            (ROOT/'installed-server-report.json').write_text(json.dumps({'metadata':metadata,'dimension':len(vector),
                'norm_squared':sum(v*v for v in vector),'wrong_model_rejected':True,'real_inference':True},indent=2),encoding='utf-8')
            print('Installed BGE server: real inference, norm, dimension and model identity checks passed')
    finally:
        try:
            owned=psutil.Process(process.pid)
            family=owned.children(recursive=True)
            for child in reversed(family):
                try: child.kill()
                except psutil.NoSuchProcess: pass
            process.terminate()
        except psutil.NoSuchProcess: pass
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)


if __name__=='__main__':
    main()
