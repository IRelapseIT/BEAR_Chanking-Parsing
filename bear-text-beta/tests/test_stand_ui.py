from fastapi.testclient import TestClient

from bear_text_beta.api import create_app


def test_stand_serves_packaged_routes_without_loading_models():
    with TestClient(create_app()) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert 'lang="ru"' in page.text and 'id="viewer"' in page.text
        assert '/stand/app.js' in page.text
        assets = [client.get("/stand/" + name) for name in ("app.js", "style.css")]
        assert all(item.status_code == 200 for item in assets)
        assert "runtime/manual" not in assets[0].text
