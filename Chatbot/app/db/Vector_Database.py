from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from app.Config import Config

class VectorDatabase:
 config = Config()
 client = QdrantClient(url=config.QDRANT_URL)

 def create_collection(self,vector_size: int):
    if self.config.COLLECTION_NAME not in [collection.name for collection in self.client.get_collections().collections]:
        self.client.create_collection(
            collection_name=self.config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )