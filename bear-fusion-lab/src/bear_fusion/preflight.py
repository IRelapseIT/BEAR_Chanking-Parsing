"""Reject inputs known to be truncated by the pinned upstream renderer."""
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile


def reject_known_table_loss(source: Path) -> None:
    if source.suffix.lower() in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(source.read_bytes(), "lxml")
        tables = [[cell.get_text(" ", strip=True) for cell in table.find_all(["td", "th"])]
                  for table in soup.find_all("table")]
        row_counts = [len(table.find_all("tr")) for table in soup.find_all("table")]
    elif source.suffix.lower() == ".docx":
        # The caller MUST perform bounded ZIP validation before this function.
        with ZipFile(source) as archive:
            tree = ElementTree.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        nodes = tree.findall(".//w:tbl", ns)
        tables = [["".join(t.text or "" for t in cell.findall(".//w:t", ns))
                   for cell in table.findall(".//w:tc", ns)] for table in nodes]
        row_counts = [len(table.findall(".//w:tr", ns)) for table in nodes]
    else:
        return
    if any(count > 5000 for count in row_counts) or any(
        len(cell.replace("|", "\\|")) > 512 for cells in tables for cell in cells
    ):
        raise ValueError("UPSTREAM_TABLE_TRUNCATION_RISK: refusing known lossy parse; original retained")
