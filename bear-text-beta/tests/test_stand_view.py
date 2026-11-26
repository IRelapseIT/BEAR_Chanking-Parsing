from dataclasses import replace

from test_ingestion import submit, system  # noqa: F401

from bear_text_beta.models import ProcessingProfile
from bear_text_beta.process_worker import run_once


def test_table_processor_error_preserves_actual_text(parsed, chunker):
    class Broken:
        def process(self, block, context):
            raise ValueError("broken table structure")

    document = parsed("Before.\n\n| Item | Value |\n| --- | --- |\n| A | 21 |\n\nAfter.", "document.md")
    original = next(block for block in document.blocks if block.type == "TABLE")
    chunker.external_processors["TABLE"] = Broken()
    result = chunker.process(document, ProcessingProfile())
    fallback = [chunk for chunk in result.chunks if chunk.strategy == "external_fallback"]
    assert "".join(chunk.text for chunk in fallback) == original.metadata["raw_source"]
    assert fallback[0].source_spans[0].block_id == original.id
    assert fallback[0].metadata["cell_relationships"] == "unverified"


def test_view_original_of_empty_checkpoint(system, context):
    services, client = system
    original = b"\n \n"
    job = submit(client, context, original).json()["job_id"]
    run_once(services)
    scope = {"tenant_id": context.tenant_id, "space_id": context.space_id}
    assert client.get(f"/v1/jobs/{job}/artifacts/original", params=scope).content == original
    assert client.get(f"/v1/jobs/{job}/artifacts/original", params={**scope, "space_id": "other"}).status_code == 404


def test_rechunk_uses_saved_parse_without_calling_parser(system, context, monkeypatch):
    services, client = system
    job = submit(client, replace(context, filename="long.txt"), b"Long story. " * 100).json()["job_id"]
    run_once(services)
    original = services.manifest.load(job)
    monkeypatch.setattr(services.parser, "parse", lambda *args: (_ for _ in ()).throw(AssertionError("reparsed")))
    scope = {"tenant_id": context.tenant_id, "space_id": context.space_id}
    submitted = client.post(f"/v1/jobs/{job}/rechunk", params=scope,
                            json={"mode": "fixed", "target_tokens": 120})
    assert submitted.status_code == 202
    new_job = submitted.json()["job_id"]
    run_once(services)
    result = services.manifest.load(new_job)
    assert result.parsed_document == original.parsed_document
    assert len(result.chunks) > len(original.chunks)
    assert client.get(f"/v1/jobs/{new_job}").json()["search_available"]
