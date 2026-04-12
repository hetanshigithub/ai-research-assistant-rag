from src.chunking import chunk_documents
from src.data_loader import Document


def test_chunk_documents_creates_overlapping_chunks() -> None:
    document = Document(source="memory.txt", text=" ".join(str(i) for i in range(30)))

    chunks = chunk_documents([document], chunk_size=10, overlap=2)

    assert len(chunks) == 4
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
