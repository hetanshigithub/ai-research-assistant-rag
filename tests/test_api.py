import pytest
from fastapi.testclient import TestClient

from api.main import Settings, create_app


@pytest.fixture()
def client(tmp_path, data_dir):
    settings = Settings(
        data_dir=data_dir,
        index_dir=tmp_path / "index",
        upload_dir=tmp_path / "uploads",
        embedding_model="hashing",
        chunk_size=180,
        chunk_overlap=35,
        index_type="flat",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_reports_ready_after_startup(client) -> None:
    body = client.get("/health").json()
    assert body == {"status": "ok", "index_ready": True}


def test_index_info_describes_the_loaded_index(client) -> None:
    body = client.get("/index").json()
    assert body["embedding_model"] == "hashing-2048"
    assert body["documents"] >= 11
    assert body["chunks"] >= body["documents"]
    assert body["index_type"] == "flat"
    assert body["rebuilt_on_startup"] is True


def test_query_returns_answer_chunks_and_latency(client) -> None:
    response = client.post(
        "/query", json={"question": "Why is an inner-product index used for cosine similarity?", "top_k": 3}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["summary"]
    assert 0.0 <= body["answer"]["confidence_score"] <= 1.0
    assert len(body["retrieved"]) == 3
    assert [c["rank"] for c in body["retrieved"]] == [1, 2, 3]
    scores = [c["score"] for c in body["retrieved"]]
    assert scores == sorted(scores, reverse=True)
    assert body["retrieved"][0]["source_file"] == "vector_search_and_faiss.txt"
    assert body["retrieved"][0]["text"]
    assert body["latency_ms"] >= 0


def test_include_chunks_false_omits_the_text(client) -> None:
    body = client.post(
        "/query", json={"question": "What is recall at k?", "top_k": 2, "include_chunks": False}
    ).json()
    assert all(chunk["text"] is None for chunk in body["retrieved"])


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "  "},
        {"question": "ab"},
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "top_k": 21},
        {},
    ],
)
def test_invalid_requests_are_rejected(client, payload) -> None:
    assert client.post("/query", json=payload).status_code == 422


def test_second_startup_reuses_the_saved_index(tmp_path, data_dir) -> None:
    settings = Settings(
        data_dir=data_dir, index_dir=tmp_path / "index", embedding_model="hashing"
    )
    with TestClient(create_app(settings)) as first:
        assert first.get("/index").json()["rebuilt_on_startup"] is True
    with TestClient(create_app(settings)) as second:
        assert second.get("/index").json()["rebuilt_on_startup"] is False


def test_requests_before_startup_get_503(tmp_path, data_dir) -> None:
    settings = Settings(data_dir=data_dir, index_dir=tmp_path / "index", embedding_model="hashing")
    # Not used as a context manager, so the startup hook never runs.
    unstarted = TestClient(create_app(settings))

    assert unstarted.get("/health").json() == {"status": "starting", "index_ready": False}
    assert unstarted.post("/query", json={"question": "anything at all"}).status_code == 503


def test_app_refuses_to_start_without_documents(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "missing", index_dir=tmp_path / "index", embedding_model="hashing"
    )
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(settings)):
            pass
