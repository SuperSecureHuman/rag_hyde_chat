# Chat with Lex (or at least his guests)

RAG using HyDE with completely open-source, self-hostable models.

TLDR; check folder 7 for the final app.

## Components

1. **Vector DB** - [Milvus](https://milvus.io/) - Dockerized
2. **Attu GUI for Milvus** - [Attu](https://zilliz.com/attu) - Dockerized
3. **Embedding Model** - [MixedBread - Embed Large model](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) - Chosen because it is one of the top models on the embedding leaderboard.
4. **Serving LLM** - [Vllm](https://github.com/vllm-project/vllm)
5. **Onnxruntime GPU** - [Onnxruntime GPU](https://onnxruntime.ai/) (Optional: speeds up embeddings)
6. **Llamaindex** - [Llamaindex](https://www.llamaindex.ai/) - Provides the RAG framework
7. **Streamlit** - [Streamlit](https://streamlit.io/) - For the UI
8. **OpenAI API** - Optional: useful to isolate bugs from self-hosting models


## Plan of the System

The major components of this system will be the VectorDB, LLM inference, and the embedding model. The Streamlit application is quite light in comparison.

By choosing scalable bases for the major components, the entire setup will be inherently scalable.

- **Milvus**: Supports using a Kubernetes (k8s) cluster, allowing us to set scaling rules so the DB will scale according to the load.
  
- **VLLM**: This is by far the fastest LLM serving engine I've used. We can scale it to use any number of GPUs within a single node, and we can set up multi-node inference using a Ray cluster. It's also possible to set up k8s load-based scaling, provided that we define the resources properly.

- **Embedding**: In this case, I've used local serving for the embedding. Ideally, I would opt for serving the embedding separately and scaling them independently.

Put together, the system design would look something like this:

![System Design](https://i.imgur.com/tp7EmZb.png)


## Let's Build It!

### Step 1 - Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Set up Milvus (this will be our vector DB):

Milvus runs as a Docker container. There are scripts in the Milvus docs:

```bash
# Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh

# Start the Docker container
bash standalone_embed.sh start
```

Make sure to download the dataset from Kaggle!

![Portainer Instace Screenshot](https://i.imgur.com/GIyY6UL.png)

### Step 2 - Test Milvus

Check the folder [1. Milvus Setup Testing](./1_milvus_setup_testing/). It contains the default template "Hello World" for Milvus. This installation assumes you used the install script, which runs Milvus at port 19530.

### Step 3 - Embedding Model

We are using Fastembed backend for the embedding model. Let's test the model. During the first-time test, the model is also downloaded, which saves us time in later stages.

The folder [2. Embedding Setup Testing](./2_embedding_setup_testing/) contains a script - [test.py](./2_embedding_setup_testing/test.py) that downloads the model, loads it, and encodes a few sample sentences.

### Step 4 - Test LlamaIndex Setup

We use a template code from LlamaIndex docs to check if our LlamaIndex installation works as intended. For testing purposes, I used an OpenAI key at this stage to minimize bugs. This will change in later stages.

### Step 5 - Ingest Data to the VectorDB

The dataset is a single CSV file. I've converted each row (which represents each episode) into its own JSON file (see [data_splitter.py](./4_data_ingestion_vector_db/data_splitter.py)). This makes organizing much easier. Another advantage of this approach is that the file name is automatically included in the metadata of each entry in the VectorDB, making it easier to organize the DB without parsing other columns in the CSV to generate the metadata.

Now, I can ingest the data into the VectorDB. Please note that this step is a one-time process, and using a GPU at this stage is recommended. CPU-only ingestion takes a very long time. On my laptop (with RTX 3060), it took ~5 minutes. On CPU, the estimated time was ~4 hours (i7, 8C 16T CPU). The script to ingest the VectorDB from the generated JSON files can be found here - [ingest.py](./4_data_ingestion_vector_db/ingest.py).

If you installed Attu, then you should be able to see this data over there

![Attu Dashboard](https://i.imgur.com/1B7OQS1.jpeg)

### Step 6 - Test Inference in LlamaIndex

At this stage, we are going to put together our VectorDB and embedding model with LlamaIndex and test if we can do a retrieval with it. This script loads up the embedding model and VectorDB. If you get an output at this stage, congrats! The setup is done! Kinda...

[5. Sample Inference LlamaIndex VectorDB](./5_sample_inference_llamaindex_vectordb/)

![](https://i.imgur.com/jlof6HM.png)

### Step 7 - Demo Chat App

This uses the Streamlit template to have a multi-turn chat with our VectorDB as the chat engine.

[6. Streamlit Basic Chat](./6_streamlit_app_basic_chat/)

![](https://i.imgur.com/OH9VmPI.png)

### Step 8 - Setup VLLM

I've hosted Llama 3 8B on an A100 instance using VLLM. This model runs on VLLM, using FP8 cache, which enables the model to run with high throughput using just ~20GB VRAM (allowing it to run on cheaper GPUs like L4). I've hosted this model and used a reverse proxy to one of my subdomains - `https://llm.parasu.in/v1/`.

![](https://i.imgur.com/LvnDEyF.png)


### Final Setup - Integrate Custom LLM with Streamlit

Here, we use the OpenAILike class from LlamaIndex, which can load models that have OpenAI-compatible endpoints. The base URL, API key, and the model name are kept as inputs in the Streamlit app itself. At this point, the Milvus URL is hardcoded, but it can be replaced with an environment variable when needed. The code is well-commented and should be self-explanatory.

Given a query, it outputs the answer,  the Hypothetical Document it generated, along with the sources of the retervial.

[app.py](./7_hyde_streamlit_customllm/app.py)

![](https://i.imgur.com/HWSzlXX.png)