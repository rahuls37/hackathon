from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import PyPDF2
# Removed chromadb to reduce bundle size
import google.generativeai as genai
import os
from typing import List
import uuid
from dotenv import load_dotenv
import logging
from io import BytesIO
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="HackRx 6.0 - RAG Q&A API", version="1.0.0")

# Configure Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
if GEMINI_API_KEY != "your-gemini-api-key-here":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✅ Gemini API configured successfully")
    except Exception as e:
        logger.warning(f"⚠️ Gemini API configuration failed: {e}")
        model = None
else:
    logger.warning("⚠️ No Gemini API key provided")
    model = None

# In-memory storage for documents and chunks (lightweight alternative to ChromaDB)
processed_documents_cache = {}
document_chunks_cache = {}  # Store chunks for text search

# Pydantic models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Helper functions
async def download_pdf(url: str) -> bytes:
    """Download PDF from URL"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def store_document_chunks(doc_id: str, chunks: List[str]):
    """Store document chunks in memory for lightweight text search"""
    try:
        # Store chunks in memory with metadata
        document_chunks_cache[doc_id] = [
            {"text": chunk, "chunk_index": i, "document_id": doc_id}
            for i, chunk in enumerate(chunks)
        ]
        logger.info(f"Stored {len(chunks)} chunks for document {doc_id}")
    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        raise

def extract_query_entities(question: str) -> dict:
    """Extract key entities from the query using pattern matching and keywords"""
    question_lower = question.lower()
    entities = {
        'age': None,
        'gender': None,
        'procedure': None,
        'location': None,
        'policy_duration': None,
        'medical_condition': None,
        'amount': None
    }
    
    # Age extraction
    import re
    age_patterns = [r'(\d+)\s*(?:year|yr|y)(?:s)?(?:\s*old)?', r'(\d+)m(?:\s|$)', r'(\d+)f(?:\s|$)', r'age\s*(\d+)']
    for pattern in age_patterns:
        match = re.search(pattern, question_lower)
        if match:
            entities['age'] = int(match.group(1))
            break
    
    # Gender extraction
    if any(word in question_lower for word in ['male', 'm', 'man', 'mr']):
        entities['gender'] = 'male'
    elif any(word in question_lower for word in ['female', 'f', 'woman', 'mrs', 'ms']):
        entities['gender'] = 'female'
    
    # Medical procedures and conditions
    medical_terms = {
        'surgery': ['surgery', 'operation', 'surgical', 'procedure'],
        'knee': ['knee', 'joint', 'arthroscopy'],
        'heart': ['heart', 'cardiac', 'bypass', 'angioplasty'],
        'eye': ['eye', 'cataract', 'lasik', 'retina'],
        'maternity': ['maternity', 'pregnancy', 'delivery', 'childbirth'],
        'dental': ['dental', 'tooth', 'oral']
    }
    
    for condition, keywords in medical_terms.items():
        if any(keyword in question_lower for keyword in keywords):
            if entities['procedure'] is None:
                entities['procedure'] = condition
            if entities['medical_condition'] is None:
                entities['medical_condition'] = condition
    
    # Location extraction (Indian cities)
    cities = ['pune', 'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'ahmedabad']
    for city in cities:
        if city in question_lower:
            entities['location'] = city.title()
            break
    
    # Policy duration
    duration_patterns = [r'(\d+)\s*(?:month|mon|m)(?:s)?(?:\s*old)?(?:\s*policy)?', r'(\d+)\s*(?:year|yr|y)(?:s)?(?:\s*policy)?']
    for pattern in duration_patterns:
        match = re.search(pattern, question_lower)
        if match and 'policy' in question_lower:
            duration = int(match.group(1))
            if 'month' in pattern or 'mon' in pattern or ('m' in pattern and 'policy' in question_lower):
                entities['policy_duration'] = f"{duration} months"
            else:
                entities['policy_duration'] = f"{duration} years"
            break
    
    return entities

def generate_semantic_queries(question: str, entities: dict) -> List[str]:
    """Generate multiple semantic queries to improve retrieval"""
    queries = [question]  # Original query
    
    # Generate entity-based queries
    if entities.get('procedure') or entities.get('medical_condition'):
        condition = entities.get('procedure') or entities.get('medical_condition')
        queries.extend([
            f"{condition} coverage",
            f"{condition} waiting period",
            f"{condition} exclusions",
            f"medical treatment {condition}"
        ])
    
    if entities.get('age'):
        queries.append(f"age limit {entities['age']} years")
    
    if entities.get('policy_duration'):
        queries.extend([
            f"waiting period {entities['policy_duration']}",
            f"policy duration {entities['policy_duration']}",
            "pre-existing disease waiting period"
        ])
    
    # Add general insurance terms
    queries.extend([
        "coverage benefits",
        "exclusions",
        "waiting period",
        "sum insured",
        "claim process"
    ])
    
    return list(set(queries))  # Remove duplicates

def retrieve_relevant_chunks(question: str, doc_id: str, top_k: int = 10) -> List[str]:
    """Lightweight text search for relevant chunks"""
    try:
        if doc_id not in document_chunks_cache:
            return []
        
        chunks = document_chunks_cache[doc_id]
        question_lower = question.lower()
        
        # Extract entities for better matching
        entities = extract_query_entities(question)
        
        # Score chunks based on keyword matching
        scored_chunks = []
        for chunk_data in chunks:
            chunk_text = chunk_data["text"].lower()
            score = 0
            
            # Score based on question words
            question_words = question_lower.split()
            for word in question_words:
                if len(word) > 2:  # Skip very short words
                    score += chunk_text.count(word) * 2
            
            # Score based on extracted entities
            for entity_type, entity_value in entities.items():
                if entity_value:
                    entity_str = str(entity_value).lower()
                    score += chunk_text.count(entity_str) * 3
            
            # Score based on common insurance terms
            insurance_terms = ['coverage', 'waiting period', 'exclusion', 'benefit', 'claim', 'policy', 'premium']
            for term in insurance_terms:
                if term in question_lower and term in chunk_text:
                    score += 2
            
            if score > 0:
                scored_chunks.append((score, chunk_data["text"]))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        relevant_chunks = [chunk[1] for chunk in scored_chunks[:top_k]]
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks using text search")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error in text search retrieval: {e}")
        return []

async def generate_answer(question: str, context: str) -> str:
    """Generate intelligent answer using enhanced prompting"""
    try:
        # Extract entities for more targeted answering
        entities = extract_query_entities(question)
        
        # Create an enhanced prompt with reasoning
        prompt = f"""
You are an expert insurance policy analyst. Your task is to answer questions about insurance policies using the provided document context.

IMPORTANT INSTRUCTIONS:
1. ALWAYS try to find relevant information in the context, even if the question seems vague
2. Use semantic understanding - look for related concepts, not just exact keyword matches
3. If the question mentions specific details (age, procedure, duration), find related policy rules
4. Provide clear, actionable answers
5. If you find partial information, provide what you can determine
6. Only say information is not available if you truly cannot find ANY related content

EXTRACTED QUERY DETAILS:
{entities}

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

PLEASE ANALYZE THE CONTEXT AND PROVIDE A COMPREHENSIVE ANSWER:

Think step by step:
1. What specific information is the user asking for?
2. What related concepts or rules exist in the document?
3. How do the policy terms apply to this situation?
4. What is the most helpful answer I can provide?

ANSWER:"""
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Post-process the answer to ensure quality
        if "information is not available" in answer.lower() and len(context) > 100:
            # Try a simpler, more direct approach
            simplified_prompt = f"""
Based on this insurance policy document:

{context}

Question: {question}

Provide a helpful answer using any relevant information from the document. Look for related terms, coverage details, waiting periods, exclusions, or similar concepts.

Answer:"""
            
            try:
                simplified_response = model.generate_content(simplified_prompt)
                simplified_answer = simplified_response.text.strip()
                if len(simplified_answer) > len(answer) and "not available" not in simplified_answer.lower():
                    answer = simplified_answer
            except:
                pass  # Keep original answer
        
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I apologize, but I encountered an error while generating the answer."

# API Routes
@app.get("/")
async def root():
    return {"message": "HackRx 6.0 - RAG Q&A API is running!", "version": "1.0.0", "endpoints": {"/health": "Health check", "/api/v1/hackrx/run": "Main processing endpoint"}}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "HackRx RAG API"}

# Legacy endpoint for backward compatibility
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_questions_legacy(request: QueryRequest, http_request: Request):
    """Legacy endpoint - redirects to new API version"""
    return await process_questions(request, http_request)

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_questions(request: QueryRequest, http_request: Request):
    """Main endpoint to process documents and answer questions"""
    try:
        # Extract bearer token
        auth_header = http_request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        api_key = auth_header.split(" ")[1]
        logger.info(f"Processing request with API key: {api_key[:10]}...")
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Check if document was already processed (using in-memory cache)
        if request.documents in processed_documents_cache:
            logger.info("Using cached document")
            document_text = processed_documents_cache[request.documents]
        else:
            # Download and process document
            logger.info(f"Downloading document from: {request.documents}")
            pdf_content = await download_pdf(request.documents)
            
            # Extract text
            logger.info("Extracting text from PDF")
            document_text = extract_text_from_pdf(pdf_content)
            
            if not document_text.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from the document")
            
            # Store in cache
            processed_documents_cache[request.documents] = document_text
        
        # Chunk and store document
        logger.info("Chunking document")
        chunks = chunk_text(document_text)
        store_document_chunks(doc_id, chunks)
        
        # Process each question
        answers = []
        for question in request.questions:
            logger.info(f"Processing question: {question[:50]}...")
            
            # Retrieve relevant chunks
            relevant_chunks = retrieve_relevant_chunks(question, doc_id)
            context = "\n\n".join(relevant_chunks)
            
            # Generate answer
            answer = await generate_answer(question, context)
            answers.append(answer)
        
        logger.info(f"Successfully processed {len(request.questions)} questions")
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
