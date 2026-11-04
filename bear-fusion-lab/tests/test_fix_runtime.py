import importlib.util
import io
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from bear_fusion.markdown_adapter import StructuredMarkdownParser
from bear_fusion.pdf_adapter import ocr_region, parse_pdf
from bear_fusion.tables import TableProcessor, validate_table_chunks
from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embeddings import (
    CharacterTokenCounter,
    UnavailableEmbeddingProvider,
)
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import DocumentContext, EmbeddingProfile, ProcessingProfile


@pytest.fixture
def demo(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'demo.py'
    spec = importlib.util.spec_from_file_location('demo_fixes', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Pool:
        fail = False
        def __init__(self, **kwargs): pass
        def submit(self, *args):
            if self.fail: raise RuntimeError('executor stopped')
        def shutdown(self, **kwargs): pass
    class Server:
        def __init__(self, address, handler): module.Handler = handler
        def serve_forever(self): pass
        def server_close(self): pass
    module.JOBS = tmp_path / 'jobs'
    monkeypatch.setattr(module, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(module, 'ThreadingHTTPServer', Server)
    monkeypatch.setattr('sys.argv', ['demo.py'])
    module.main()
    module.Pool = Pool
    return module


def post(demo, body, path='/jobs'):
    handler = object.__new__(demo.Handler)
    data = json.dumps(body).encode()
    handler.path = path
    handler.headers = {'X-Bear-Demo': 'local', 'Content-Length': str(len(data))}
    handler.rfile = io.BytesIO(data)
    output = []
    handler.response = lambda value, code=200: output.append((code, value))
    handler.do_POST()
    return output[-1]


def test_upload_write_failures_do_not_poison_capacity(demo, monkeypatch):
    original = Path.write_bytes
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'write_bytes', lambda self, data: (_ for _ in ()).throw(OSError('disk full')))
        for _ in range(8):
            assert post(demo, {'filename': 'a.txt', 'data': 'aGVsbG8='})[0] == 400
    assert Path.write_bytes is original
    code, _ = post(demo, {'filename': 'a.txt', 'data': 'aGVsbG8='})
    assert code == 202
    assert sum(s['status'] in {'queued', 'running'} for s in demo.STATE.values()) == 1


def test_submit_failure_releases_capacity(demo):
    demo.Pool.fail = True
    for _ in range(8):
        assert post(demo, {'filename': 'a.txt', 'data': 'aGVsbG8='})[0] == 503
    demo.Pool.fail = False
    assert post(demo, {'filename': 'a.txt', 'data': 'aGVsbG8='})[0] == 202


@pytest.mark.parametrize('body', [[], None, 'text', 3, {'filename': [], 'data': 'aA=='}, {'filename': 'x.txt', 'data': {}}, {'filename': 'a'*300+'.txt', 'data': 'aA=='}])
def test_malformed_upload_is_typed_4xx_without_job(demo, body):
    assert post(demo, body)[0] == 400
    assert demo.STATE == {}


def test_restart_preserves_results_and_terminates_interrupted_jobs(demo):
    _, value = post(demo, {'filename': 'a.txt', 'data': 'aGVsbG8='})
    job_id = value['job_id']
    demo.STATE.clear()
    demo.restore_jobs()
    assert demo.STATE[job_id] == {'status': 'failed', 'error': 'interrupted_by_restart'}
    demo.write_json(demo.JOBS / job_id / 'result.json', {'result': {}})
    demo.update_state(job_id, {'status': 'done'})
    demo.STATE.clear()
    demo.restore_jobs()
    assert demo.STATE[job_id]['status'] == 'done'


def test_long_table_header_preserves_leaf_qualifier_and_separator():
    profile = EmbeddingProfile()
    counter = CharacterTokenCounter(profile)
    header = 'Long accounting description '*100 + 'Net_profit_RUB_2025'
    document = StructuredMarkdownParser().parse(
        ('| Metric | '+header+' |\n|---|---|\n| Revenue | 150 |').encode(),
        DocumentContext('t', 's', 'd', 1, 'x.md'))
    processor = TableProcessor(counter)
    result = ChunkingService(counter, UnavailableEmbeddingProvider(), profile,
                             {'TABLE': processor}).process(document, ProcessingProfile())
    values = [c for c in result.chunks if '150' in c.text]
    assert values and any('Net_profit_RUB_2025: 150' in c.text for c in values)
    assert document.blocks[0].metadata['header_context_truncated']
    validate_table_chunks(document, result.chunks, counter, 320)
    # Revalidation at the index boundary may use a different hard maximum.
    validate_table_chunks(document, result.chunks, counter, 480)
    altered = deepcopy(result.chunks)
    for chunk in altered:
        if '150' in chunk.text:
            meta = chunk.metadata.get('cell_segments', [chunk.metadata])[-1]
            meta['header_context_full'] = 'forged header'
    with pytest.raises(ValueError, match='context proof'):
        validate_table_chunks(document, altered, counter, 320)


def test_table_context_retains_year_and_units_before_long_description():
    profile = EmbeddingProfile()
    counter = CharacterTokenCounter(profile)
    header = '2025 RUB '+('long description '*100)+'Revenue'
    document = StructuredMarkdownParser().parse(
        ('| Metric | '+header+' |\n|---|---|\n| Account | 150 |').encode(),
        DocumentContext('t', 's', 'd', 1, 'x.md'))
    result = ChunkingService(counter, UnavailableEmbeddingProvider(), profile,
                             {'TABLE': TableProcessor(counter)}).process(document)
    value = next(c for c in result.chunks if c.text.endswith(': 150'))
    assert all(term in value.text for term in ['2025', 'RUB', 'Revenue: 150'])
    validate_table_chunks(document, result.chunks, counter, 480)


def test_two_column_pdf_reads_columns_with_full_width_heading(tmp_path):
    from reportlab.pdfgen import canvas
    path = tmp_path / 'columns.pdf'
    c = canvas.Canvas(str(path), pagesize=(600, 800))
    c.drawString(180, 760, 'FULL WIDTH DOCUMENT HEADING')
    for index in range(3):
        c.drawString(40, 700-index*24, f'LEFT_{index+1}')
        c.drawString(340, 700-index*24, f'RIGHT_{index+1}')
    c.save()
    parsed = parse_pdf(path, DocumentContext('t', 's', 'd', 1, path.name), FileLimits())
    text = '\n'.join(b.text for b in parsed.blocks)
    assert text.index('HEADING') < text.index('LEFT_1') < text.index('LEFT_3') < text.index('RIGHT_1') < text.index('RIGHT_3')
    assert any('pdf_reading_order_inferred' in warning for warning in parsed.warnings)
    assert all(b.source.bbox for b in parsed.blocks if b.text)


def test_pdf_column_order_restarts_after_full_width_section(tmp_path):
    from reportlab.pdfgen import canvas
    path = tmp_path / 'bands.pdf'
    c = canvas.Canvas(str(path), pagesize=(600, 800))
    for index in range(2):
        c.drawString(40, 740-index*24, f'A_LEFT_{index}')
        c.drawString(340, 740-index*24, f'A_RIGHT_{index}')
    c.drawString(160, 650, 'FULL WIDTH SECOND SECTION HEADING')
    for index in range(2):
        c.drawString(40, 600-index*24, f'B_LEFT_{index}')
        c.drawString(340, 600-index*24, f'B_RIGHT_{index}')
    c.save()
    result = parse_pdf(path, DocumentContext('t', 's', 'd', 1, path.name), FileLimits())
    text = '\n'.join(b.text for b in result.blocks)
    expected = ['A_LEFT_0', 'A_LEFT_1', 'A_RIGHT_0', 'A_RIGHT_1', 'HEADING',
                'B_LEFT_0', 'B_LEFT_1', 'B_RIGHT_0', 'B_RIGHT_1']
    assert [text.index(value) for value in expected] == sorted(text.index(value) for value in expected)


def test_ocr_crop_geometry_maps_pixels_to_pdf_points(monkeypatch):
    import bear_fusion.pdf_adapter as adapter
    from PIL import Image
    picture = Image.new('RGB', (600, 600), 'grey')
    class Page:
        def get_size(self): return 200, 200
        def render(self, scale): return SimpleNamespace(to_pil=lambda: picture.copy(), close=lambda: None)
        def close(self): pass
    tsv = 'block_num\tpar_num\tline_num\ttext\tconf\tleft\ttop\twidth\theight\n1\t1\t1\t123\t99\t6\t9\t12\t15\n'
    monkeypatch.setenv('BEAR_OCR_DPI', '216')
    monkeypatch.setattr(adapter.shutil, 'which', lambda name: 'test-tesseract')
    monkeypatch.setattr(adapter.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=tsv.encode()))
    text, metadata = ocr_region([Page()], 0, [10, 20, 110, 120])
    assert text == '123'
    assert metadata['ocr_words'][0]['bbox_pdf_points'] == [12, 23, 16, 28]
    assert metadata['pixel_to_pdf_transform'] == [1/3, 0, 0, 1/3, 10, 20]
    assert metadata['numeric_review_required']
