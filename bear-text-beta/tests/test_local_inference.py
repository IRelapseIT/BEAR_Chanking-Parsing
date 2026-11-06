import threading
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from bear_text_beta.local_inference import DenseLocalEncoder, make_server, pool_hidden
from bear_text_beta.model_profiles import model_profile


def test_pooling_distinguishes_cls_and_last_significant_token():
    torch=pytest.importorskip('torch')
    hidden=torch.arange(2*4*3,dtype=torch.float32).reshape(2,4,3)
    right=torch.tensor([[1,1,0,0],[1,1,1,1]])
    left=torch.tensor([[0,0,1,1],[1,1,1,1]])
    assert torch.equal(pool_hidden(hidden,right,'cls'),hidden[:,0])
    assert torch.equal(pool_hidden(hidden,right,'last'),torch.stack([hidden[0,1],hidden[1,3]]))
    assert torch.equal(pool_hidden(hidden,left,'last'),hidden[:,-1])
    with pytest.raises(ValueError,match='Empty'):
        pool_hidden(hidden,torch.zeros_like(left),'last')


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError,match='pooling'):
        pool_hidden(None,None,'mean-by-accident')


def test_http_rejects_wrong_same_dimension_model_and_preserves_inputs():
    profile=replace(model_profile('bge-m3'),model_revision='pinned',tokenizer_revision='pinned')
    seen=[]
    def encode(texts):
        seen.append(texts)
        return [[1.0]+[0.0]*1023 for _ in texts]
    encoder=SimpleNamespace(profile=profile,encode=encode,pooling='cls',metrics={},max_tokens=480)
    server=make_server(encoder,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}',trust_env=False) as client:
            models=client.get('/v1/models').json()
            assert models['data'][0]['pooling']=='cls' and models['data'][0]['dimension']==1024
            assert client.post('/v1/embeddings',json={'model':'Qwen/Qwen3-Embedding-0.6B','input':['same size']}).status_code==422
            assert seen==[]
            response=client.post('/v1/embeddings',json={'model':profile.model_id,'input':['  raw\ntext  ']})
            assert response.is_success and seen==[['  raw\ntext  ']]
            assert response.json()['data'][0]['index']==0
            assert client.post('/unknown',json={}).status_code==404
            assert client.post('/v1/embeddings',content='[]').status_code==422
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_over_budget_input_rejected_before_model_call():
    encoder=object.__new__(DenseLocalEncoder)
    encoder.max_tokens=5
    encoder.tokenizer=SimpleNamespace(encode=lambda text,**kwargs:list(range(len(text))))
    # No model or lock exists: an accidental inference call fails this test.
    with pytest.raises(ValueError,match='truncation'):
        encoder.encode(['123456'])
    for values in [[],[' '],[1]]:
        with pytest.raises(ValueError):
            encoder.encode(values)
