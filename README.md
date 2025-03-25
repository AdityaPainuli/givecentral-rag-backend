# RAG System Backend API for Give-central

This is a simple backend API collection containing the essential endpoints for a RAG (Retrieval-Augmented Generation) system designed for Give-central.

** Postman collection ** - https://app.getpostman.com/join-team?invite_code=3595eb29a6f9c6ec2ed132e7133e5eaba062e312952fb5450d7c5a20ca56ccfb&target_code=5587f65b74476f676d2cf5673927fe7c


## Features
- **Website Crawling**: Submit URLs to crawl and store the content in Pinecone.
- **Chat with Data**: Ask questions based on the crawled content.
- **Status Tracking**: Track the progress of website crawls.
- **Health Check**: Simple endpoint to check if the API is running.

## Prerequisites
Ensure you have the following installed:
- Python 3.9+
- FastAPI
- Uvicorn
- OpenAI SDK
- Pinecone SDK
- dotenv
- langchain
- crawl4ai

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in your project directory with the following variables:
Alternatively, you can rename `.env.example` to `.env` and update the values.

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
LLM_MODEL=gpt-4o-mini
```

## Endpoints

### 1. Start Crawl
**URL**: `/crawl`  
**Method**: `POST`  
**Body**:  
```json
{
  "urls": ["https://example.com", "https://another-site.com"]
}
```
**Response**:  
```json
{
  "task_id": "123456789",
  "status": "Crawl started"
}
```

---

### 2. Check Status
**URL**: `/status/{task_id}`  
**Method**: `GET`  
**Response**:  
```json
{
  "task_id": "123456789",
  "status": "in_progress" | "completed" | "failed: error_message"
}
```

---

### 3. Chat with Data
**URL**: `/chat`  
**Method**: `POST`  
**Body**:  
```json
{
  "question": "How do smart stickers work?"
}
```
**Response**:  
```json
{
  "question": "How do smart stickers work?",
  "answer": "Smart stickers work by..."
}
```

---

### 4. Health Check
**URL**: `/health`  
**Method**: `GET`  
**Response**:  
```json
{
  "status": "ok"
}
```

## Running the Application 🚀
```bash
python3 main.py
```

## Additional Notes
- Ensure `crawl.py` and `rag.py` are in the same directory as `main.py`.
- Logs will be printed to the console for monitoring.
- Use the `/status` endpoint to monitor crawl progress.

Happy Crawling and Chatting! 🤩

