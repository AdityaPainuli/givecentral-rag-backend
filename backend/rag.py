import os
import asyncio
from openai import AsyncOpenAI
from pinecone import Pinecone
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI and Pinecone clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Pinecone Index
PINECONE_INDEX_NAME = "website-crawl-index"
pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  

async def fetch_relevant_chunks(query: str, top_k: int = 10) -> List[Dict]:
    """Retrieve top-k relevant document chunks from Pinecone."""
    query_embedding = await get_embedding(query)
    results = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results['matches']

async def answer_question(question: str) -> str:
    """Retrieve relevant context and generate an answer."""
    relevant_chunks = await fetch_relevant_chunks(question)
    context = "\n\n".join([chunk['metadata']['content'] for chunk in relevant_chunks])
    
    system_prompt = """You are a helpful AI assistant. Use the provided context to answer the question accurately."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I couldn't generate an answer."

async def main():
    question = "How smart stickers works? "
    answer = await answer_question(question)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
