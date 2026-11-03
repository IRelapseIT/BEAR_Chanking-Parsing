from bear_text_beta.models import DocumentContext

from bear_fusion.html_adapter import parse_structured_html


def parse(text):
    return parse_structured_html(text.encode(), DocumentContext("t", "s", "d", 1, "a.html"))


def test_code_preserves_whitespace_and_unicode():
    code = "if True:\n    print('Привет')\n"
    document = parse("<pre>" + code + "</pre><p>tail</p>")
    assert document.blocks[0].role == "code"
    assert document.blocks[0].text == code
    assert document.blocks[-1].text == "tail"


def test_merged_table_and_cell_values_survive():
    document = parse('<table><tr><th colspan="2">Header</th></tr><tr><td rowspan="2">A</td><td>B</td></tr></table>')
    table = document.blocks[0]
    assert table.type == "TABLE"
    assert 'colspan="2"' in table.metadata["raw_source"]
    assert 'rowspan="2"' in table.metadata["raw_source"]


def test_comments_and_scripts_do_not_become_chunks():
    document = parse("<div><!-- hidden-comment --><script>hidden-script</script><pre>visible</pre></div>")
    assert [b.text for b in document.blocks] == ["visible"]
    assert "excluded_html_comments:1" in document.warnings


def test_plain_html_stays_on_upstream():
    assert parse("<h1>Title</h1><p>Text</p>") is None
