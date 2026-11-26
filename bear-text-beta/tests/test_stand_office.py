import io
from dataclasses import replace
from zipfile import ZIP_STORED, ZipFile

import pytest

from bear_text_beta.errors import ParseError, ServiceUnavailableError
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.parsing import ParsingPipeline


def make_odt(phrase):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=ZIP_STORED)
        archive.writestr("content.xml", '<?xml version="1.0" encoding="UTF-8"?>'
            '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
            ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" office:version="1.2">'
            f'<office:body><office:text><text:p>{phrase}</text:p></office:text></office:body>'
            '</office:document-content>')
        archive.writestr("META-INF/manifest.xml", '<?xml version="1.0" encoding="UTF-8"?>'
            '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"'
            ' manifest:version="1.2"><manifest:file-entry manifest:full-path="/"'
            ' manifest:media-type="application/vnd.oasis.opendocument.text"/>'
            '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
            '</manifest:manifest>')
    return buffer.getvalue()


def test_office_unavailable_checked_before_conversion(tmp_path, context, monkeypatch):
    monkeypatch.setenv("BEAR_SOFFICE", str(tmp_path / "missing-executable"))
    pipeline = ParsingPipeline(tmp_path)
    assert {".odt", ".doc", ".rtf"} <= pipeline.parsers.keys()
    assert pipeline.parsers[".rtf"].health()["ready"] is False
    with pytest.raises(ServiceUnavailableError, match="LibreOffice"):
        pipeline.parse(br"{\rtf1 example}", replace(context, filename="example.rtf"))


@pytest.mark.parametrize(("suffix", "payload"), [("rtf", b"not RTF"), ("odt", b"not ODT"), ("doc", b"not DOC")])
def test_office_format_validation_precedes_launch(tmp_path, context, suffix, payload, monkeypatch):
    monkeypatch.setenv("BEAR_SOFFICE", str(tmp_path / "missing-executable"))
    with pytest.raises(ParseError, match=suffix.upper()):
        ParsingPipeline(tmp_path).parse(payload, replace(context, filename="example." + suffix))


def test_odt_expansion_is_bounded_before_conversion(tmp_path, context):
    pipeline = ParsingPipeline(tmp_path, limits=FileLimits(max_docx_uncompressed_bytes=100))
    with pytest.raises(ParseError, match="лимит"):
        pipeline.parse(make_odt("Meeting notes"), replace(context, filename="example.odt"))


@pytest.mark.parametrize(("suffix", "payload"), [
    ("rtf", br"{\rtf1\ansi Contract orchard text.}"), ("odt", make_odt("Contract orchard text.")),
])
def test_real_office_adapter_marks_derivative_provenance(tmp_path, context, suffix, payload):
    pipeline = ParsingPipeline(tmp_path)
    if not pipeline.parsers.get("." + suffix) or not pipeline.parsers["." + suffix].health()["ready"]:
        pytest.skip("LibreOffice is absent")
    parsed = pipeline.parse(payload, replace(context, filename="notes." + suffix))
    block = parsed.blocks[0]
    assert block.text == "Contract orchard text."
    assert block.source.format == suffix
    assert block.metadata["conversion"]["derived_format"] == "docx"
    assert block.metadata["conversion"]["version"]
    assert block.metadata["derived_locator"]["format"] == "docx"
    assert "office_derivative_coordinates" in parsed.warnings
