import app


def test_app_exposes_factories_without_side_effects():
    # Importing app must NOT connect to Pinecone/OpenAI.
    assert callable(app.create_app)
    assert callable(app.build_rag_chain)


def test_no_module_level_rag_chain():
    # A connected chain must never be built at import time.
    assert not hasattr(app, "rag_chain")
