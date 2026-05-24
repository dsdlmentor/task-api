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


def test_health_response_has_no_xaccel_buffering_header() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.headers.get("X-Accel-Buffering") is None


def test_unknown_queue_path_gets_xaccel_buffering_no() -> None:
    with TestClient(app) as client:
        response = client.get("/gradio_api/queue/data?session_hash=test")
    assert response.headers.get("X-Accel-Buffering") == "no"
    assert response.headers.get("Cache-Control") == "no-cache"
