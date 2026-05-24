from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_validates_empty_question() -> None:
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


@patch("app.main.build_rag_chain")
def test_chat_returns_answer_with_sources(mock_build) -> None:
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Ridge uses L2 penalty [1]."

    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "https://scikit-learn.org/stable/linear.html"}
    mock_doc.page_content = "Ridge regression addresses..."
    mock_retriever.invoke.return_value = [mock_doc]

    mock_build.return_value = (mock_chain, mock_retriever)

    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "What is Ridge?"})

    assert response.status_code == 200
    body = response.json()
    assert "Ridge uses L2" in body["answer"]
    assert len(body["sources"]) == 1
    assert "scikit-learn.org" in body["sources"][0]["url"]


def test_disable_nginx_buffering_middleware_on_queue_path() -> None:
    from fastapi import FastAPI

    from app.main import DisableNginxBufferingMiddleware

    test_app = FastAPI()
    test_app.add_middleware(DisableNginxBufferingMiddleware)

    @test_app.get("/queue/data")
    def _queue():
        return {"ok": True}

    @test_app.get("/queue/join")
    def _join():
        return {"ok": True}

    @test_app.get("/other")
    def _other():
        return {"ok": True}

    with TestClient(test_app) as client:
        queue_data = client.get("/queue/data")
        queue_join = client.get("/queue/join")
        other = client.get("/other")

    assert queue_data.headers.get("x-accel-buffering") == "no"
    assert queue_data.headers.get("cache-control") == "no-cache"

    assert queue_join.headers.get("x-accel-buffering") == "no"
    assert queue_join.headers.get("cache-control") == "no-cache"

    assert other.headers.get("x-accel-buffering") is None
    assert other.headers.get("cache-control") != "no-cache"
