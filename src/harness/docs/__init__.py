"""docs: document ingestion (PDF + text formats, vision image captioning),
hybrid retrieval/RAG, and the physical parts graph.

``docs`` does NOT depend on ``inspect`` (see spec dependency order). The curated
register catalog lives in ``inspect``; ``docs`` only ingests/retrieves the
uploaded documents.
"""