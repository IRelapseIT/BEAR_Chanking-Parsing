"""Bounded LibreOffice conversion into the canonical DOCX parser."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

from .errors import ParseError, ServiceUnavailableError
from .file_validation import MIME_TYPES, FileValidator

OFFICE_SECURITY_PROFILE = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry">
  <item oor:path="/org.openoffice.Office.Common/Security/Scripting">
    <prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop>
    <prop oor:name="DisableMacrosExecution" oor:op="fuse"><value>true</value></prop>
  </item>
  <item oor:path="/org.openoffice.Office.Writer/Content/Update">
    <prop oor:name="Link" oor:op="fuse"><value>2</value></prop>
    <prop oor:name="Field" oor:op="fuse"><value>false</value></prop>
    <prop oor:name="Chart" oor:op="fuse"><value>false</value></prop>
  </item>
</oor:items>
"""


def libreoffice_executable():
    configured = os.getenv("BEAR_SOFFICE") or os.getenv("BEAR_LIBREOFFICE")
    if configured:
        path = Path(configured)
        # On Windows the console wrapper provides usable stdout for --version.
        if path.name.lower() == "soffice.exe" and path.with_suffix(".com").is_file():
            path = path.with_suffix(".com")
        return str(path.resolve()) if path.is_file() else None
    candidate = shutil.which("soffice.com") or shutil.which("soffice") or shutil.which("libreoffice")
    if candidate:
        return candidate
    program_files = os.getenv("ProgramFiles")
    if program_files:
        path = Path(program_files) / "LibreOffice/program/soffice.com"
        if path.is_file():
            return str(path.resolve())
    return None


@lru_cache(maxsize=4)
def libreoffice_version(executable):
    try:
        output = subprocess.run([executable, "--version"], timeout=10, check=True, capture_output=True, text=True)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ServiceUnavailableError("LibreOffice не прошёл проверку запуска") from exc
    version = output.stdout.strip()
    if not version.startswith("LibreOffice "):
        raise ServiceUnavailableError("LibreOffice не вернул версию")
    return version


def validate_doc(data):
    from xlrd.compdoc import CompDoc, CompDocError

    try:
        stream = CompDoc(data).get_named_stream("WordDocument")
        if not stream or not stream.startswith(b"\xec\xa5"):
            raise ParseError("В контейнере DOC нет двоичного документа Word")
    except (CompDocError, ValueError, IndexError, OSError) as exc:
        raise ParseError("Повреждённый контейнер DOC") from exc


class OfficeConversionParser:
    def __init__(self, limits, docx_parser):
        self.limits = limits
        self.docx_parser = docx_parser

    def health(self):
        return {"ready": libreoffice_executable() is not None, "tool": "LibreOffice"}

    def parse(self, data, context):
        from bear_fusion.ocr_verification import run_worker

        suffix = Path(context.filename).suffix.lower()
        source_format = suffix[1:]
        if not data.startswith(FileValidator.SIGNATURES[suffix]):
            raise ParseError(f"Содержимое не соответствует сигнатуре {source_format.upper()}")
        validator = FileValidator(self.limits)
        if suffix == ".odt":
            validator.validate_odt(data)
        elif suffix == ".doc":
            validate_doc(data)
        executable = libreoffice_executable()
        if not executable:
            raise ServiceUnavailableError("Для конвертации DOC/ODT/RTF установите LibreOffice")
        version = libreoffice_version(executable)
        timeout = 90
        maximum_disk_bytes = self.limits.max_file_bytes * 3
        with tempfile.TemporaryDirectory(prefix="bear-office-") as folder:
            directory = Path(folder)
            profile = directory / "profile"
            scratch = directory / "scratch"
            scratch.mkdir()
            (profile / "user").mkdir(parents=True)
            (profile / "user/registrymodifications.xcu").write_text(OFFICE_SECURITY_PROFILE, encoding="utf-8")
            source = directory / ("source" + suffix)
            converted = directory / "source.docx"
            source.write_bytes(data)
            args = [executable, "-env:UserInstallation=" + profile.as_uri(), "--headless", "--nologo",
                    "--norestore", "--convert-to", "docx:Office Open XML Text", "--outdir", str(directory),
                    str(source)]
            env = {**os.environ, "TEMP": str(scratch), "TMP": str(scratch), "TMPDIR": str(scratch)}
            try:
                run_worker(args, converted, timeout, env,
                           max_output_bytes=self.limits.max_file_bytes, work_dir=directory,
                           max_work_bytes=maximum_disk_bytes)
                content = converted.read_bytes()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise ParseError(f"LibreOffice не получил DOCX из {source_format.upper()}: {type(exc).__name__}") from exc
        validator.validate_docx(content)
        derivative_context = replace(context, filename="source.docx", mime_type=MIME_TYPES[".docx"], checksum="")
        result = self.docx_parser.parse(content, derivative_context)
        conversion = {"tool": "LibreOffice", "version": version, "original_format": source_format,
                      "derived_format": "docx", "timeout_seconds": timeout, "disk_limit_bytes": maximum_disk_bytes,
                      "original_layout_preserved": False}
        result.context = context
        result.parser_profile = f"{source_format}-libreoffice-v1:{result.parser_profile}"
        result.warnings.extend(["office_derivative_coordinates", "office_conversion_layout_may_differ"])
        for block in result.blocks:
            block.metadata.update(conversion=conversion, derived_locator=asdict(block.source))
            block.source = replace(block.source, format=source_format, page=None, bbox=None)
        return result
