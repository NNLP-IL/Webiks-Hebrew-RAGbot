import os

EMBEDDING_INDEX = os.getenv("ES_EMBEDDING_INDEX", "embedded_index")
ES_EMBEDDING_INDEX_LENGTH = int(os.getenv("ES_EMBEDDING_INDEX_LENGTH", "1000"))
MODEL_LOCATION = os.getenv("MODEL_LOCATION", "model")

