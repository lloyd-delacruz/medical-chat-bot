# Medical Chatbot (LLMs · LangChain · Pinecone · Flask)

A Retrieval-Augmented Generation (RAG) medical Q&A chatbot. It ingests medical
reference material from three sources, embeds it with a local sentence-transformer
model, stores the vectors in Pinecone, and answers questions with OpenAI through a
safety-aware prompt.

> ⚠️ **Medical disclaimer:** This project is for educational use only. Its answers are
> general health information, **not a substitute for professional medical care**. For
> emergencies, contact your local emergency services.

## Architecture

```
curated_data.py   data/ folder        WHO/CDC/NIH/MedlinePlus
   (built-in)     (drop-in files)        (live web fetch)
        \              |                      /
         \             |                     /
          v            v                    v
        src/data_sources.gather_documents()   (rich metadata)
                       |
        filter_to_minimal_docs → text_split → BGE embeddings (384-dim)
                       |
              Pinecone index "medical-chatbot"
                       |
   app.py: retriever (top-k) → gpt-4.1 + safety prompt → answer
```

## Data sources (kept up to date)

Ingestion is unified in `src/data_sources.py` and each source can be toggled in `.env`:

- **Curated** (`ENABLE_CURATED`) — built-in, cited reference docs in `src/curated_data.py`
  (stroke BE-FAST, vital signs, when to seek emergency care, hypertension, type 2
  diabetes, medication safety, prevention), grounded in current WHO/CDC/MedlinePlus pages.
- **Drop-in files** (`ENABLE_DROPIN`) — drop any `.pdf`, `.txt`, or `.md` files into `data/`
  and re-run ingestion.
- **Live web** (`ENABLE_WEB`) — fetches current WHO/MedlinePlus fact sheets. Fail-soft: an
  unreachable page is logged and skipped, never crashing the build.

# How to run?

### STEP 01 — Create and activate an environment
```bash
conda create -n medibot python=3.12 -y
conda activate medibot
```

### STEP 02 — Install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03 — Configure `.env`
Copy `.env.example` to `.env` and fill in your keys (every other variable has a sensible
default — see `.env.example` for the full list):

```ini
PINECONE_API_KEY="your-pinecone-key"
OPENAI_API_KEY="your-openai-key"
```

Key defaults: `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` (384-dim, runs locally),
`LLM_MODEL=gpt-4.1` (set `gpt-4o-mini` for cheaper runs), `INDEX_NAME=medical-chatbot`.

### STEP 04 — Build the Pinecone index
```bash
python store_index.py
```

> ⚠️ This **recreates** the `medical-chatbot` index: if it already exists it is **deleted**
> and rebuilt from scratch, so it only contains current BGE vectors. If the upsert step
> fails partway through, just re-run the command. The first run also downloads the
> embedding model (~130 MB).

### STEP 05 — Run the app
```bash
python app.py
```

Then open http://localhost:8080 . (Set `FLASK_DEBUG=true` to enable Flask debug mode.)

## Running the tests
```bash
pip install -r requirements-dev.txt
pytest
```

The test suite runs fully offline — no Pinecone/OpenAI calls or model downloads required.

### Techstack Used:

- Python 3.12
- LangChain (community, huggingface, text-splitters, pinecone, openai)
- Sentence-Transformers (BAAI/bge-small-en-v1.5)
- Pinecone (serverless vector store)
- OpenAI (gpt-4.1)
- Flask
- BeautifulSoup4 + requests (web ingestion)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - OPENAI_API_KEY
