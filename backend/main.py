from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import asyncio
from crawl import crawl_parallel
from rag import answer_question

app = FastAPI()

class CrawlRequest(BaseModel):
    urls: List[str]


class ChatRequest(BaseModel):
    question:str
# Track crawl status
crawl_status = {}

@app.post("/crawl")
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    
    task_id = str(hash(frozenset(request.urls)))
    crawl_status[task_id] = "in_progress"

    def run_crawl():
        try:
            asyncio.run(crawl_parallel(request.urls))
            crawl_status[task_id] = "completed"
        except Exception as e:
            crawl_status[task_id] = f"failed: {str(e)}"
    
    background_tasks.add_task(run_crawl)
    return {"task_id": task_id, "status": "Crawl started"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in crawl_status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return {"task_id": task_id, "status": crawl_status[task_id]}


@app.post("/chat")
async def chat(request:ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail = "No question provided")
    answer = await answer_question(request.question)
    return {"question":request.question,  "answer":answer}

    
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
