import pytest
from fastapi.testclient import TestClient

from api.citations import locate_citations
from api.main import Settings, create_app
from src.chunking import TextChunk
from src.retriever import RetrievalResult as Retrieved

NOTE = (
    "Quarterly reports should be archived for seven years. "
    "The archive must be stored in a separate region from production data. "
    "Access to the archive requires approval from two managers."
)


def make_client(tmp_path, data_dir, **overrides):
    settings = Settings(
        data_dir=data_dir,
        upload_dir=tmp_path / "uploads",
        index_dir=tmp_path / "index",
        embedding_model="hashing",
        **overrides,
    )
    return TestClient(create_app(settings))


@pytest.fixture()
def client(tmp_path, data_dir):
    with make_client(tmp_path, data_dir) as test_client:
        yield test_client


def upload(client, name="retention.txt", content=NOTE.encode(), content_type="text/plain"):
    return client.post("/documents", files={"file": (name, content, content_type)})


def test_documents_lists_the_sample_files(client) -> None:
    body = client.get("/documents").json()
    assert len(body["documents"]) >= 11
    assert {d["origin"] for d in body["documents"]} == {"sample"}
    assert body["total_chunks"] == sum(d["chunks"] for d in body["documents"])


def test_uploaded_document_becomes_searchable(client) -> None:
    before = client.get("/index").json()["chunks"]

    response = upload(client)
    assert response.status_code == 201
    document = response.json()["document"]
    assert document["name"] == "retention.txt"
    assert document["origin"] == "uploaded"
    assert document["chunks"] >= 1
    assert client.get("/index").json()["chunks"] == before + document["chunks"]

    answer = client.post(
        "/query", json={"question": "How long should quarterly reports be archived?", "top_k": 3}
    ).json()
    assert answer["retrieved"][0]["source_file"] == "retention.txt"


def test_deleting_an_upload_removes_it_from_search(client) -> None:
    upload(client)
    assert client.delete("/documents/retention.txt").status_code == 204

    names = [d["name"] for d in client.get("/documents").json()["documents"]]
    assert "retention.txt" not in names
    answer = client.post("/query", json={"question": "How long should quarterly reports be archived?"}).json()
    assert all(c["source_file"] != "retention.txt" for c in answer["retrieved"])


def test_sample_documents_cannot_be_deleted(client) -> None:
    response = client.delete("/documents/vector_search_and_faiss.txt")
    assert response.status_code == 403


def test_deleting_an_unknown_document_is_404(client) -> None:
    assert client.delete("/documents/missing.txt").status_code == 404


def test_duplicate_names_are_rejected(client) -> None:
    assert upload(client).status_code == 201
    assert upload(client).status_code == 409
    # Sample names count too, whatever the letter case.
    assert upload(client, name="Vector_Search_And_FAISS.txt").status_code == 409


@pytest.mark.parametrize("name", ["notes.docx", "no_extension", ".hidden.txt", "bad<name>.txt"])
def test_unsupported_or_unsafe_names_are_rejected(client, name) -> None:
    assert upload(client, name=name).status_code == 415


def test_path_components_in_the_file_name_are_dropped(client, tmp_path) -> None:
    response = upload(client, name="../../escape.txt")
    assert response.status_code == 201
    assert response.json()["document"]["name"] == "escape.txt"
    assert (tmp_path / "uploads" / "escape.txt").is_file()
    assert not (tmp_path.parent / "escape.txt").exists()


def test_empty_file_is_rejected(client) -> None:
    assert upload(client, content=b"   \n").status_code == 422


def test_pdf_without_text_is_rejected_and_not_kept(client, tmp_path) -> None:
    response = upload(client, name="scan.pdf", content=b"%PDF-1.4 not really a pdf", content_type="application/pdf")
    assert response.status_code == 422
    assert not (tmp_path / "uploads" / "scan.pdf").exists()
    assert "scan.pdf" not in [d["name"] for d in client.get("/documents").json()["documents"]]


def test_oversized_upload_is_rejected(tmp_path, data_dir) -> None:
    with make_client(tmp_path, data_dir, max_upload_bytes=100) as small:
        assert upload(small, content=b"word " * 100).status_code == 413


def test_uploads_survive_a_restart(tmp_path, data_dir) -> None:
    with make_client(tmp_path, data_dir) as first:
        upload(first)
    with make_client(tmp_path, data_dir) as second:
        names = [d["name"] for d in second.get("/documents").json()["documents"]]
        assert "retention.txt" in names
        assert second.get("/index").json()["rebuilt_on_startup"] is False


def test_cors_allows_the_configured_origin(tmp_path, data_dir) -> None:
    with make_client(tmp_path, data_dir, cors_origins=("http://localhost:5173",)) as c:
        ok = c.options(
            "/query",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
        other = c.options(
            "/query",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in other.headers


def test_query_marks_low_confidence_as_not_answerable(tmp_path, data_dir) -> None:
    with make_client(tmp_path, data_dir, confidence_threshold=0.99) as c:
        body = c.post("/query", json={"question": "What is recall at k?"}).json()
        assert body["answerable"] is False
        assert body["confidence_threshold"] == 0.99
    with make_client(tmp_path, data_dir, confidence_threshold=0.0) as c:
        assert c.post("/query", json={"question": "What is recall at k?"}).json()["answerable"] is True


def test_every_citation_points_at_the_exact_statement(client) -> None:
    body = client.post(
        "/query", json={"question": "Why is an inner-product index used for cosine similarity?", "top_k": 4}
    ).json()
    chunks = {c["rank"]: c["text"] for c in body["retrieved"]}
    citations = body["answer"]["citations"]

    assert len(citations) == len(body["answer"]["key_insights"])
    for citation in citations:
        assert citation["found"], citation["statement"]
        text = chunks[citation["rank"]]
        located = text[citation["start"] : citation["end"]]
        assert located == citation["statement"].rstrip(".") or located == citation["statement"]


def chunk(rank_text: str, chunk_id: int = 1, source: str = "data/a.txt") -> Retrieved:
    return Retrieved(chunk=TextChunk(chunk_id=chunk_id, source=source, text=rank_text), score=0.5)


def test_locate_prefers_the_highest_ranked_chunk() -> None:
    sentence = "The archive must be stored in a separate region."
    first = chunk(f"Intro words here. {sentence} Tail words.", chunk_id=7)
    second = chunk(f"{sentence} More.", chunk_id=8)
    result = locate_citations([sentence], [first, second])[0]
    assert (result.rank, result.chunk_id) == (1, 7)
    assert first.chunk.text[result.start : result.end] == sentence


def test_locate_handles_a_sentence_cut_off_by_a_chunk_boundary() -> None:
    # The generator appends a period to sentences with no ending punctuation.
    result = locate_citations(["Access requires approval from two."], [chunk("Start. Access requires approval from two")])[0]
    assert result.found
    assert result.end == len("Start. Access requires approval from two")


def test_locate_reports_missing_statements_instead_of_guessing() -> None:
    result = locate_citations(["This sentence is nowhere in the text."], [chunk("Something else entirely.")])[0]
    assert result.found is False
    assert result.start is None and result.rank is None
