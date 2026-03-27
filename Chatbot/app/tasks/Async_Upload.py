import os
import shutil
import gc
import time
import logging
from celery import Celery
from app.Config import Config

config = Config()
celery = Celery("tasks", broker=config.REDIS_URL, backend=config.REDIS_URL)
celery.conf.update(
    result_expires=3600,          
    task_track_started=True,  
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

@celery.task(bind=True)
def async_upload(self, path: str):
    try:
        from app.services.Ingest_Service import Upload
        upload = Upload()
        self.update_state(state="PROGRESS", meta={"status": "Extracting documents..."})
        result = upload.upload_documents(path)

        self.update_state(state="PROGRESS", meta={"status": "Refreshing search index..."})
        from app.services.Query_Service import Retrieve
        retriever = Retrieve()
        retriever.refresh_bm25()
        return {"status": "SUCCESS", "result": result}
    except Exception as e:
        self.update_state(state="FAILURE", meta={"status": str(e)})
        raise e
    finally:
        gc.collect()

        for attempt in range(5):
            try:
                if not os.path.exists(path):
                    break
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                logging.info(f"Deleted: {path}")
                break
            except PermissionError:
                logging.warning(f"Cleanup attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
        else:
            logging.error(f"Could not delete {path} after 5 attempts")