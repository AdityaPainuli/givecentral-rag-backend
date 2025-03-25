import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode 
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from typing import List, Optional


load_dotenv()

# Initialize OpenAI and Pinecone clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Constants for Pinecone
PINECONE_INDEX_NAME = "website-crawl-index"
PINECONE_DIMENSION = 1536  # Dimension for text-embedding-3-small

# Initialize or get Pinecone index
def get_or_create_index():
    """Get or create Pinecone index for website crawl data."""
    try:
        # Check if index exists
        if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
            # Create a new index
            pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Get index
        return pinecone_client.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Error with Pinecone index: {e}")
        raise

# Initialize the index
pinecone_index = get_or_create_index()

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(
    markdown_content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    headers_to_split_on: Optional[List[str]] = None
) -> List[str]:
    """
    Chunk markdown content into smaller pieces for RAG vector storage.
    
    Args:
        markdown_content: The markdown content to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        headers_to_split_on: List of markdown headers to use as chunk boundaries 
                            (e.g. ["#", "##", "###"])
    
    Returns:
        List of text chunks
    """
    # Default headers to split on if none provided
    if headers_to_split_on is None:
        headers_to_split_on = ["#", "##", "###"]
    
    # First try markdown-aware splitting to preserve semantic structure
    try:
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = markdown_splitter.split_text(markdown_content)
    except Exception as e:
        # Fallback to recursive character splitting if markdown splitting fails
        print(f"Markdown splitting failed, falling back to character splitting. Error: {e}")
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = recursive_splitter.split_text(markdown_content)
    
    # Filter out empty chunks and strip whitespace
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from website document chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "givencentral",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Pinecone."""
    try:
        # Create a unique ID for this chunk
        chunk_id = f"{urlparse(chunk.url).netloc}_{urlparse(chunk.url).path}_{chunk.chunk_number}"
        chunk_id = chunk_id.replace("/", "_").replace(".", "_")
        
        # Prepare metadata
        metadata = {
            "url": chunk.url,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "chunk_number": chunk.chunk_number,
            **chunk.metadata  # Include all other metadata
        }
        
        # Upsert the vector into Pinecone
        pinecone_index.upsert(
            vectors=[
                {
                    "id": chunk_id,
                    "values": chunk.embedding,
                    "metadata": metadata
                }
            ]
        )
        
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url} into Pinecone")
    except Exception as e:
        print(f"Error inserting chunk into Pinecone: {e}")

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        

        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

async def main():
    urls = ["https://www.givecentral.org/smart-stickers"]
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())