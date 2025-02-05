import logging
import time
import torch
from datetime import datetime
from .llm_client import LLMClient
from . import config
from .elastic_model import es_model_factory
from .document import document_definition_factory
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
definitions = document_definition_factory()


class Engine:
    """
       Engine class for handling document search and retrieval using Elasticsearch and LLMs.

       Attributes:
           llms_client (LLMClient): The LLM client instance.
           elastic_model (Model): The Elasticsearch model instance.
            model_name (Name, optional): The name of the model. Default is None
            field_to_embed (Field, optional): The field it will embed. Default is None
            retrieval_model(Path, optional): The path to the model. Default is none
           identifier_field (str): The identifier field for documents.
       Methods:
           update_docs(list_of_docs, embed_only_fields=None, delete_existing=False):
               Updates or creates documents in the Elasticsearch index.
           search_documents(query, top_k):
               Searches for documents based on the query and returns the top_k results.
           answer_query(query, top_k, model):
               Answers a query using the top_k documents and the specified model.
       """

    def __init__(self, llms_client: LLMClient, elastic_model=None, model_name=None, field_to_embed=None, retrieval_model=None, es_client=None):
        """
          Initializes the Engine instance.
          Args:
              llms_client (LLMClient): The LLM client instance.
              elastic_model (Model, optional): The Elasticsearch model instance. Default is None.
              model_name (Name, optional): The name of the model. Default is None
              field_to_embed (Field, optional): The field it will embed. Default is None
              retrieval_model(Path, optional): The retrieval model, made with SentenceTransformer and our model. Default is none
              es_client (optional): The Elasticsearch client instance. Default is None.
          """
        if elastic_model is None:
            self.elastic_model = es_model_factory(es_client)
        else:
            self.elastic_model = elastic_model

        self.llms_client = llms_client

        self.identifier_field = definitions.identifier
        if model_name is None:
            self.model_name = definitions.model_name
        else:
            self.model_name = model_name
        if field_to_embed is None:
            self.field_to_embed = definitions.field_to_embed
        if retrieval_model is None:
            self.retrieval_model = SentenceTransformer(config.MODEL_LOCATION + "/" + self.model_name).to(device)
        else:
            self.retrieval_model = retrieval_model
        self.retrieval_model.eval()


    def update_docs(self, list_of_docs: list[dict], delete_existing=False):
        """
          Updates or creates documents in the Elasticsearch index.
          Args:
              list_of_docs (list[dict]): A list of dictionaries representing the documents to be indexed.
              delete_existing (bool, optional): Whether to delete existing documents. Default is False.
          """
        for doc in list_of_docs:
            if definitions.field_to_embed in doc.keys():
                content_vectors = self.retrieval_model.encode(doc[definitions.field_to_embed])
                doc[f'{definitions.field_to_embed}_{definitions.model_name}_vectors'] = content_vectors

            doc['last_update'] = datetime.now()
        self.elastic_model.create_or_update_documents(list_of_docs, delete_existing)


    def create_paragraphs(self,list_of_paragraphs:list[dict]):
        """
        This function gets paragraphs, embeds them and adding to elastic
        Parameters
        ----------
        list_of_paragraphs

        Returns
        -------

        """
        for doc in list_of_paragraphs:
            if definitions.field_to_embed in doc.keys():
                content_vectors = self.retrieval_model.encode(doc[definitions.field_to_embed])
                doc[f'{definitions.field_to_embed}_{definitions.model_name}_vectors'] = content_vectors
                doc['last_update'] = datetime.now()
                self.elastic_model.create_paragraph(doc)



    def search_documents(self, query: str, top_k: int):
        """
           Searches for documents based on the query and returns the top_k results.
           Args:
               query (str): The query string.
               top_k (int): The number of top documents to return.
           Returns:
               list: A list of top k documents.
           """
        query_embeddings = self.retrieval_model.encode(query)
        all_docs = self.elastic_model.search(query_embeddings)
        top_k_documents = []
        top_doc_ids = []

        for doc in all_docs:
            if doc["_source"]["doc_id"] not in top_doc_ids:
                top_k_documents.append(doc["_source"])
                top_doc_ids.append(doc["_source"]["doc_id"])
            if len(top_doc_ids) >= top_k:
                break

        return top_k_documents


    def answer_query(self, query, top_k: int, model):
        """
               Answers a query using the top_k documents and the specified model.
               Args:
                   query (str): The query string.
                   top_k (int): The number of top documents to use for answering the query.
                   model: The model to use for answering the query.
               Returns:
                   tuple: A tuple containing the top k documents, the answer, and the stats.
               """
        before_retrieval = time.perf_counter()
        top_k_documents = self.search_documents(query, top_k)

        retrieval_time = round(time.perf_counter() - before_retrieval, 4)
        logging.info(f"retrieval time: {retrieval_time}")

        llm_answer, llm_elapsed, tokens = self.llms_client.answer(query, top_k_documents)
        stats = {
            "retrieval_time": retrieval_time,
            "llm_model": model,
            "llm_time": llm_elapsed,
            "tokens": tokens
        }
        return top_k_documents, llm_answer, stats


engine = None


def engine_factory(llms_client: LLMClient, es_client=None):
    global engine
    if engine is None:
        engine = Engine(llms_client=llms_client, es_client=es_client)
    return engine
