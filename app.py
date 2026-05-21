import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.config import Config
from src.helper import download_embeddings
from src.prompt import system_prompt


def build_rag_chain(config: Config):
    embeddings = download_embeddings(config)
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=config.index_name, embedding=embeddings
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": config.retriever_k}
    )
    chat_model = ChatOpenAI(model=config.llm_model)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    qa_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, qa_chain)


def create_app(config: Config | None = None) -> Flask:
    load_dotenv()
    config = config or Config.from_env()
    config.require_keys()
    os.environ["PINECONE_API_KEY"] = config.pinecone_api_key
    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    app = Flask(__name__)
    rag_chain = build_rag_chain(config)

    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        print("User:", msg)
        response = rag_chain.invoke({"input": msg})
        print("Bot:", response["answer"])
        return str(response["answer"])

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8080, debug=True)
