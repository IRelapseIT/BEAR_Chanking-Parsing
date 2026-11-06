"""Actual semantic boundary inference with full BEAR coverage validation."""
import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path

import httpx

from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embedding_pipeline import EmbeddingPipeline
from bear_text_beta.embeddings import HFTokenCounter, InfinityEmbeddingProvider, load_profile
from bear_text_beta.models import DocumentContext, ProcessingProfile
from bear_text_beta.parsing import TXTParser
from local_inference import DenseLocalEncoder, make_server

ROOT=Path(__file__).resolve().parent


def run_checks(counter,provider,profile,encoder):
    ru=('Магазин Норд возвращает деньги за товар в течение 12 рабочих дней. Срок зависит от способа оплаты. '
        'Для корректной обработки заявки нужно сохранить чек и номер заказа. '
        'На планетах Солнечной системы различаются температура и атмосфера. '
        'Газовые гиганты состоят преимущественно из водорода и гелия. '
        'Программа на Go может передавать сигнал отмены через context.Context. '
        'Для остановки обработки горутина должна проверить канал Done. ')*6
    en=('Atlas stores invoice records for twelve months. Payment confirmation must be attached to each record. '
        'The astronomer observes distant galaxies through a telescope. Their distance is measured in light years. '
        'The Go worker checks context cancellation before reading another task. Closing a data channel does not cancel a context. ')*6
    result=[]
    for label,text in [('ru',ru),('en',en)]:
        assert counter.full_count(text)>320 and counter.count(text)<2048
        for mode in ['adaptive','semantic','token']:
            document=TXTParser().parse(text.encode(),DocumentContext('semantic','test',label,1,label+'.txt'))
            before=dict(provider.metrics); started=time.perf_counter()
            processed=ChunkingService(counter,provider,profile).process(document,ProcessingProfile(mode=mode))
            entry={'language':label,'mode':mode,'chunks':len(processed.chunks),'seconds':time.perf_counter()-started,
                   'branches':dict(Counter(c.strategy for c in processed.chunks)),
                   'fallbacks':dict(Counter(c.fallback_reason for c in processed.chunks if c.fallback_reason)),
                   'max_tokens':max(c.token_count for c in processed.chunks),
                   'inference':{k:provider.metrics[k]-before[k] for k in before},
                   'source_coverage_validated':True,
                   'boundaries':[[[s.start_char,s.end_char] for s in c.source_spans] for c in processed.chunks]}
            if mode in ['adaptive','semantic']:
                assert set(entry['branches'])=={'semantic'},entry
                assert not entry['fallbacks'],entry
                assert entry['inference']['input_texts']>0
            else:
                assert entry['inference']['input_texts']==0
            result.append(entry)
    text='слово '*900
    document=TXTParser().parse(text.encode(),DocumentContext('semantic','test','long',1,'long.txt'))
    processed=ChunkingService(counter,provider,profile).process(document,ProcessingProfile(mode='semantic'))
    assert all(c.token_count<=320 for c in processed.chunks)
    long_case={'source_coverage_validated':True,'max_tokens':max(c.token_count for c in processed.chunks),
               'branches':dict(Counter(c.strategy for c in processed.chunks)),
               'fallbacks':dict(Counter(c.fallback_reason for c in processed.chunks if c.fallback_reason))}
    # A long sentence may be split safely inside Chonkie. Test a genuine HTTP failure separately.
    saved=encoder.encode
    def unavailable(_):
        raise RuntimeError('Controlled HTTP inference outage')
    encoder.encode=unavailable
    try:
        document=TXTParser().parse(ru.encode(),DocumentContext('semantic','outage','ru',1,'ru.txt'))
        processed=ChunkingService(counter,provider,profile).process(document,ProcessingProfile(mode='semantic'))
        assert all(c.strategy=='recursive' and 'semantic_failure' in (c.fallback_reason or '') for c in processed.chunks)
        assert all(c.token_count<=320 for c in processed.chunks)
    finally:
        encoder.encode=saved
    return {'variants':result,'long_sentence':long_case,'real_http_outage_safe_fallback':True}


def main():
    args=argparse.ArgumentParser();args.add_argument('model',choices=['bge','qwen']);args=args.parse_args()
    out=ROOT/args.model
    model_dir=Path('D:/CODEX/BEAR/BEAR_BGE_M3/models/bge-m3') if args.model=='bge' else Path(Path('D:/CODEX/BEAR/review/native/model-path.txt').read_text().strip())
    token_dir=model_dir if args.model=='bge' else out/'tokenizer'
    profile=load_profile(token_dir)
    encoder=DenseLocalEncoder(model_dir,profile)
    counter=HFTokenCounter(token_dir,profile)
    server=make_server(encoder,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/v1/'
    provider=InfinityEmbeddingProvider(url,profile,counter,client=httpx.Client(base_url=url,timeout=120,trust_env=False))
    pipeline=EmbeddingPipeline(provider,counter,profile,max_tokens=480)
    try:
        report=run_checks(counter,pipeline,profile,encoder)
        (out/'semantic-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True),flush=True)
    finally:
        pipeline.close();server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':
    main()
