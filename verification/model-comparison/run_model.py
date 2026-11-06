"""Paired dense-only evaluation and real BEAR/Qdrant acceptance; no generated vectors."""
import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import numpy as np
import psutil
from transformers import AutoTokenizer

from bear_text_beta.chunking import SafeTextSplitter
from bear_text_beta.embedding_pipeline import EmbeddingPipeline
from bear_text_beta.embeddings import HFTokenCounter, InfinityEmbeddingProvider, load_profile
from bear_text_beta.errors import StorageError, ValidationError
from bear_text_beta.models import DocumentContext, EmbeddingProfile, ProcessingProfile
from bear_text_beta.runtime import Services
from local_inference import DenseLocalEncoder, make_server

ROOT = Path(__file__).resolve().parent
BGE = Path('D:/CODEX/BEAR/BEAR_BGE_M3/models/bge-m3')
QWEN = Path(Path('D:/CODEX/BEAR/review/native/model-path.txt').read_text().strip())


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        return listener.getsockname()[1]


def metrics(rows):
    return {'queries':len(rows),'hit_at_1':sum(r['rank']==1 for r in rows)/len(rows),
            'hit_at_5':sum(0<r['rank']<=5 for r in rows)/len(rows),
            'recall_at_5':sum(r['recall_at_5'] for r in rows)/len(rows),
            'mrr_at_10':sum(1/r['rank'] if 0<r['rank']<=10 else 0 for r in rows)/len(rows),
            'ndcg_at_10':sum(r['ndcg_at_10'] for r in rows)/len(rows)}


def main():
    args=argparse.ArgumentParser()
    args.add_argument('model',choices=['bge','qwen'])
    args=args.parse_args()
    out=ROOT/args.model
    out.mkdir(exist_ok=True)
    qprofile=EmbeddingProfile(model_id='Qwen/Qwen3-Embedding-0.6B',model_revision=QWEN.name,
        tokenizer_revision=QWEN.name,vector_dimension=1024,max_input_tokens=512,
        query_prefix='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:',
        document_prefix='',similarity_prefix='')
    profile=load_profile(BGE) if args.model=='bge' else qprofile
    model_dir=BGE if args.model=='bge' else QWEN
    report={'model':asdict(profile),'scope':'CPU float32, 4 threads, microbatch 4, exact cosine dense retrieval; no reranker',
            'corpus_sha256':hashlib.sha256((ROOT/'corpus.json').read_bytes()).hexdigest(),
            'project':os.environ.get('PROJECT_UNDER_TEST'),'checks':{}}
    stop=threading.Event(); peak=[0]
    def monitor():
        while not stop.wait(.1):
            try: peak[0]=max(peak[0],psutil.Process().memory_info().rss)
            except psutil.Error: pass
    watcher=threading.Thread(target=monitor,daemon=True); watcher.start()
    started=time.perf_counter()
    encoder=DenseLocalEncoder(model_dir,profile)
    report['load_seconds']=time.perf_counter()-started
    print(args.model,'loaded',round(report['load_seconds'],3),flush=True)
    server=make_server(encoder,port=0)
    serving=threading.Thread(target=server.serve_forever,daemon=True); serving.start()
    url=f'http://127.0.0.1:{server.server_port}/v1'
    token_dir=model_dir
    if args.model=='qwen':
        token_dir=out/'tokenizer'; token_dir.mkdir(exist_ok=True)
        encoder.tokenizer.save_pretrained(token_dir)
        checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in token_dir.iterdir() if p.is_file() and p.name!='bear-model.json'}
        save(token_dir/'bear-model.json',{'embedding_profile':asdict(profile),'tokenizer_checksums':checksums})
    counter=HFTokenCounter(token_dir,profile)
    provider=InfinityEmbeddingProvider(url,profile,counter,max_tokens=480,
        client=httpx.Client(base_url=url+'/',timeout=120,trust_env=False))
    pipeline=EmbeddingPipeline(provider,counter,profile,max_tokens=480)
    qdrant=None; services=None
    try:
        probe=['Срок возврата денег за товар.','Stars form from clouds of gas and dust.']
        grouped=np.asarray(encoder.encode(probe))
        separate=np.asarray([encoder.encode([s])[0] for s in probe])
        assert grouped.shape==(2,1024) and np.isfinite(grouped).all()
        assert np.allclose(grouped,separate,atol=2e-5,rtol=2e-4)
        assert not np.allclose(grouped[0],grouped[1])
        report['checks']['batch_padding_invariance']={'max_abs_difference':float(np.max(abs(grouped-separate))),
            'norms':np.linalg.norm(grouped,axis=1).tolist()}
        with httpx.Client(timeout=5,trust_env=False) as client:
            wrong=client.post(url+'/embeddings',json={'model':'wrong-model-with-same-dimension','input':['test']})
            assert wrong.status_code==422
        report['checks']['wrong_model_rejected']=True
        calls=encoder.metrics['requests']
        try: pipeline.embed_batch(['слово '*2000],'document')
        except ValidationError: pass
        else: raise AssertionError('Oversized input accepted')
        assert encoder.metrics['requests']==calls
        report['checks']['oversize_rejected_before_http']=True
        corpus=json.loads((ROOT/'corpus.json').read_text('utf-8'))
        # Identical character ranges for both models, fitted to BOTH real tokenizers.
        tokenizers=[AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False) for p in [BGE,QWEN]]
        fits=lambda text:all(len(t.encode(text,add_special_tokens=True,truncation=False))<=320 for t in tokenizers)
        chunks=[]
        for doc in corpus['documents']:
            spans=SafeTextSplitter(fits).split(doc['text'])
            assert ''.join(doc['text'][a:b] for a,b in spans)==doc['text']
            chunks.extend({'document_id':doc['id'],'text':doc['text'][a:b],'start':a,'end':b} for a,b in spans if doc['text'][a:b].strip())
        save(out/'fixed-chunks.json',chunks)
        before=dict(encoder.metrics); elapsed=time.perf_counter()
        document_vectors=[]
        for start in range(0,len(chunks),16):
            document_vectors.extend(pipeline.embed_batch([c['text'] for c in chunks[start:start+16]],'document'))
            print(args.model,'embedded',min(start+16,len(chunks)),'/',len(chunks),flush=True)
        document_seconds=time.perf_counter()-elapsed
        elapsed=time.perf_counter()
        query_vectors=pipeline.embed_batch([q['text'] for q in corpus['queries']],'query')
        query_seconds=time.perf_counter()-elapsed
        document_vectors=np.asarray(document_vectors); query_vectors=np.asarray(query_vectors)
        np.savez_compressed(out/'vectors.npz',documents=document_vectors,queries=query_vectors)
        scores=query_vectors@document_vectors.T
        evaluations=[]
        for q,row in zip(corpus['queries'],scores,strict=True):
            # Aggregate by best chunk, then return unique documents.
            best={}
            for c,value in zip(chunks,row,strict=True):
                best[c['document_id']]=max(best.get(c['document_id'],-2),float(value))
            ranking=sorted(best,key=lambda k:(-best[k],k))
            gold=set(q['relevant_ids'])
            ranks=[i+1 for i,d in enumerate(ranking) if d in gold]
            dcg=sum(1/math.log2(i+2) for i,d in enumerate(ranking[:10]) if d in gold)
            idcg=sum(1/math.log2(i+2) for i in range(min(10,len(gold))))
            evaluations.append({**q,'rank':min(ranks,default=0),'recall_at_5':sum(d in gold for d in ranking[:5])/len(gold),
                'ndcg_at_10':dcg/idcg,'top_5':[{'id':d,'score':best[d]} for d in ranking[:5]]})
        report['retrieval']={'documents':len(corpus['documents']),'fixed_chunks':len(chunks),
            'document_seconds':document_seconds,'query_seconds':query_seconds,
            'max_model_tokens':max(counter.full_count(c['text']) for c in chunks),
            'metrics':metrics(evaluations),'groups':{g:metrics([r for r in evaluations if r['group']==g]) for g in sorted({q['group'] for q in evaluations})},
            'inference':{k:encoder.metrics[k]-before[k] for k in before},'queries':evaluations}
        save(out/'report.json',report)
        print(args.model,'retrieval',json.dumps(report['retrieval']['metrics']),flush=True)
        # The actual runtime, model HTTP port and a fresh native Qdrant namespace.
        http_port,grpc_port=free_port(),free_port()
        qcfg=out/'qdrant.yaml'
        qcfg.write_text('telemetry_disabled: true\nlog_level: WARN\nstorage:\n  storage_path: '+json.dumps(str(out/'qdrant-storage'))+
            '\nservice:\n  host: 127.0.0.1\n'+f'  http_port: {http_port}\n  grpc_port: {grpc_port}\n',encoding='utf-8')
        with (out/'qdrant.log').open('w',encoding='utf-8') as log:
            qdrant=subprocess.Popen(['D:/CODEX/BEAR/review/native/qdrant-server/qdrant.exe','--config-path',str(qcfg)],
                cwd=out,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        with httpx.Client(timeout=1,trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get(f'http://127.0.0.1:{http_port}/healthz').is_success: break
                except httpx.HTTPError: pass
                if qdrant.poll() is not None: raise RuntimeError((out/'qdrant.log').read_text())
                time.sleep(.1)
            else: raise RuntimeError('Qdrant readiness timeout')
        for name in ['BEAR_CHUNKER_URL','BEAR_QDRANT_COLLECTION','BEAR_OFFLINE','BEAR_TEST_COUNTER']:
            os.environ.pop(name,None)
        os.environ.update(BEAR_INFINITY_URL=url,BEAR_INFINITY_MODEL=profile.model_id,BEAR_QDRANT_URL=f'http://127.0.0.1:{http_port}')
        services=Services(out/'state',token_dir)
        sources=[
            ('refund.txt','Магазин Норд возвращает деньги за товар в течение 12 рабочих дней.'),
            ('stars.txt','Звёзды образуются из облаков межзвёздного газа и пыли.'),
            ('ssl.txt','Сервер NGINX использует SSL сертификат для защищённого соединения HTTPS.'),
            ('table.md','# Finance\n\n| Metric | 2024 | 2025 |\n|---|---|---|\n| Revenue | 120 | 150 |'),
            ('code.md','# Example\n\n```python\ndef copy_sorted(a):\n    b = sorted(a)\n    return b\n```'),
            ('html.html','<h1>Режим работы</h1><p>Поддержка работает с 09:00 до 18:00 по рабочим дням.</p>'),
            ('unicode.txt','👩‍💻' * 150 + 'x' * 1800),
        ]
        prose=('Магазин Норд возвращает деньги за товар в течение 12 рабочих дней. Срок зависит от способа оплаты. '
               'Для корректной обработки заявки нужно сохранить чек и номер заказа. '
               'На планетах Солнечной системы различаются температура и атмосфера. '
               'Газовые гиганты состоят преимущественно из водорода и гелия. '
               'Программа на Go может передавать сигнал отмены через context.Context. '
               'Для остановки обработки горутина должна проверить канал Done. ')*3
        sources.append(('semantic.txt',prose))
        results=[]
        for filename,text in sources:
            began=time.perf_counter()
            result=services.process(text.encode(),DocumentContext('compare','test',filename,1,filename),ProcessingProfile())
            services.indexer.index(result)
            results.append(result)
            report.setdefault('pipeline',[]).append({'file':filename,'chunks':len(result.chunks),'seconds':time.perf_counter()-began,
                'branches':dict(Counter(c.strategy for c in result.chunks)),'fallbacks':dict(Counter(c.fallback_reason for c in result.chunks if c.fallback_reason)),
                'max_tokens':max(c.token_count for c in result.chunks),'completeness':result.completeness})
            save(out/('processed-'+filename.replace('.','-')+'.json'),asdict(result))
        # Restrict to standalone documents for an unambiguous smoke query.
        hits=services.indexer.search('Как защитить соединение NGINX через HTTPS?','compare','test',5)
        assert hits and hits[0].payload['document_id']=='ssl.txt'
        assert services.indexer.search('SSL','other','test')==[]
        assert services.indexer.search('SSL','compare','other')==[]
        report['checks']['live_qdrant_retrieval_and_isolation']=True
        table=next(r for r in results if r.context.document_id=='table.md')
        assert any('120' in c.text and '150' in c.text for c in table.chunks)
        report['checks']['whole_table_row_and_token_validation']=True
        old=results[0]
        updated=services.process('Норд возвращает деньги за 90 дней.'.encode(),replace(old.context,document_version=2,checksum=''),ProcessingProfile())
        upsert=services.vectors.upsert
        services.vectors.upsert=lambda *_:(_ for _ in ()).throw(StorageError('Controlled write failure'))
        try: services.indexer.index(updated)
        except StorageError: pass
        else: raise AssertionError('Injected write failure was ignored')
        assert old.generation_id in services.manifest.active_generations('compare','test')
        services.vectors.upsert=upsert
        services.indexer.retry(updated.generation_id)
        active=services.manifest.active_generations('compare','test')
        assert updated.generation_id in active and old.generation_id not in active
        report['checks']['failed_reindex_preserves_old_and_retry_activates_new']=True
        report['collection']=services.vectors.collection
        report['server_metrics']=encoder.metrics
    finally:
        if services is not None: services.close()
        pipeline.close()
        if qdrant is not None:
            qdrant.terminate()
            try: qdrant.wait(timeout=10)
            except subprocess.TimeoutExpired: qdrant.kill(); qdrant.wait(timeout=10)
        server.shutdown(); server.server_close(); serving.join(timeout=5)
        stop.set(); watcher.join(timeout=2)
        report['peak_rss_mb']=peak[0]/1024**2
        report['total_seconds']=time.perf_counter()-started
        save(out/'report.json',report)
    print(args.model,'finished',json.dumps({'checks':report['checks'],'peak_rss_mb':report['peak_rss_mb']}),flush=True)


if __name__=='__main__':
    main()
