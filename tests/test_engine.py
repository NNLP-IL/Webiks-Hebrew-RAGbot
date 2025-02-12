import pytest
from unittest.mock import MagicMock, patch
import builtins
import sys
import os
import importlib
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

project_root = Path(__file__).parent.parent
fake_config_path = project_root / "example-conf.json"

class ElasticSetup:
    @staticmethod
    def setup():
        with patch.dict(os.environ, {"DOCUMENT_DEFINITION_CONFIG": str(fake_config_path)}), \
                patch.object(builtins, "open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "{\"identifier_field\": \"doc_id\", \"saved_fields\": {\"title\": \"text\", \"doc_id\": \"integer\", \"link\": \"text\", \"content\": \"text\"}, \"field_for_llm\": \"content\", \"model_name\": \"Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0\", \"field_to_embed\": \"content\"}"
        return (
            importlib.import_module("ragbot.engine").Engine,
            importlib.import_module("ragbot.llm_client").LLMClient,
        )


Engine ,LLMClient = ElasticSetup.setup()


@pytest.fixture
def mock_dependencies(mocker):
    """Mock LLM client, Elasticsearch model, and sentence transformer."""
    mock_llm_client = mocker.patch('ragbot.llm_client.LLMClient')
    mock_es_model = mocker.patch('ragbot.elastic_model.es_model_factory')
    mock_sentence_model = mocker.patch('sentence_transformers.SentenceTransformer')

    mock_llm_client_instance = mock_llm_client.return_value
    mock_es_model_instance = mock_es_model.return_value
    mock_sentence_model_instance = mock_sentence_model.return_value

    engine = Engine(
        llms_client=mock_llm_client_instance,
        elastic_model=mock_es_model_instance,
        retrieval_model=mock_sentence_model_instance
    )

    return engine, mock_llm_client_instance, mock_es_model_instance, mock_sentence_model_instance


def test_update_docs(mock_dependencies):
    engine, mock_llm_client, mock_es_model, mock_sentence_model = mock_dependencies
    list_of_docs = [{"title": "Test", "content": "This is a test."}]
    mock_sentence_model.encode.return_value = [1.0, 2.0, 3.0]
    mock_es_model.create_or_update_documents = MagicMock()

    engine.update_docs(list_of_docs)

    assert list_of_docs[0][f'{engine.field_to_embed}_{engine.model_name}_vectors'] == [1.0, 2.0, 3.0]
    mock_es_model.create_or_update_documents.assert_called_once_with(list_of_docs, False)


def test_create_paragraphs(mock_dependencies):
    engine, mock_llm_client, mock_es_model, mock_sentence_model = mock_dependencies

    list_of_paragraphs = [{"content": "This is a paragraph."}]
    mock_sentence_model.encode.return_value = [1.0, 2.0, 3.0]
    mock_es_model.create_paragraph = MagicMock()

    engine.create_paragraphs(list_of_paragraphs)

    assert list_of_paragraphs[0][f'{engine.field_to_embed}_{engine.model_name}_vectors'] == [1.0, 2.0, 3.0]
    mock_es_model.create_paragraph.assert_called_once_with(list_of_paragraphs[0])


def test_search_documents(mock_dependencies):
    engine, mock_llm_client, mock_es_model, mock_sentence_model = mock_dependencies

    query = "Test query"
    top_k = 5
    mock_sentence_model.encode.return_value = [1.0, 2.0, 3.0]
    mock_es_model.search = MagicMock(return_value=[{"_source": {"doc_id": 1, "title": "Test"}}])

    result = engine.search_documents(query, top_k)

    assert len(result) == 1
    assert result[0]["title"] == "Test"


def test_answer_query(mock_dependencies):
    engine, mock_llm_client, mock_es_model, mock_sentence_model = mock_dependencies

    query = "Test query"
    top_k = 5
    model = "Test model"

    mock_sentence_model.encode.return_value = [1.0, 2.0, 3.0]
    mock_es_model.search = MagicMock(return_value=[{"_source": {"doc_id": 1, "title": "Test"}}])
    mock_llm_client.answer = MagicMock(return_value=("Answer", 2.5, 10))

    top_k_docs, answer, stats = engine.answer_query(query, top_k, model)

    assert answer == "Answer"
    assert stats["retrieval_time"] >= 0
    assert stats["llm_model"] == model
    assert "llm_time" in stats
    assert "tokens" in stats
