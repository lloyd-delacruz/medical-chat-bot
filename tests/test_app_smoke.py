import app


def test_app_exposes_factories_without_side_effects():
    # Importing app must NOT connect to Pinecone/OpenAI.
    assert callable(app.create_app)
    assert callable(app.build_rag_chain)
