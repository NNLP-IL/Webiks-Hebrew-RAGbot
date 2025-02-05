# **Webiks-Hebrew-RAGbot**

## **Overview**

This project is a search engine that uses machine learning models and Elasticsearch to provide advanced document retrieval.
You can use [Webiks-Hebrew-RAGbot-Demo](https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot-Demo) to demonstrate the engine's document retrieval abilities

## **Features**

Document representation and validation
Document embedding and indexing in Elasticsearch
Advanced search using machine learning model
Integration with LLM (Large Language Model) client for query answering

## **Installation**

1.  Clone the repository:
   
`git clone https://github.com/NNLP-IL/Webiks-Hebrew-RAGbot.git`

`cd Webiks-Hebrew-RAGbot`

2.  Create a virtual environment and activate it:  

`python -m venv venv`

`source venv/bin/activate`

On Windows use `\venv\\Scripts\\activate\`

3.  Install the required dependencies:  

`pip install -r requirements.txt`

## **Configuration**

Set the following environment variables:  

ES\_EMBEDDING\_INDEX: The name of the Elasticsearch index for embeddings.

TOKENIZER\_LOCATION: The location of the tokenizer model.
