from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
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

 def filename_exists(self, filename: str) -> bool:
        if self.config.COLLECTION_NAME not in [c.name for c in self.client.get_collections().collections]:
           return None
        results, _ = self.client.scroll(
            collection_name=self.config.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename)
                    )
                ]
            ),
            limit=1,          
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0
 
 def get_file_hash(self, filename: str) -> str | None:
        if self.config.COLLECTION_NAME not in [c.name for c in self.client.get_collections().collections]:
           return None
        results, _ = self.client.scroll(
            collection_name=self.config.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=filename))]
            ),
            limit=1,
            with_payload=True,  
            with_vectors=False,
        )
        if results:
            return results[0].payload.get("file_hash")
        return None

 def delete_by_filename(self, filename: str):
        self.client.delete(
            collection_name=self.config.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename)
                    )
                ]
            )
        )        

        