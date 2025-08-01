import os
import asyncio
import httpx
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2
from io import BytesIO

# Load environment variables
load_dotenv()

async def test_pdf_download():
    """Test PDF download"""
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            print(f"✅ PDF download successful. Size: {len(response.content)} bytes")
            
            # Test PDF text extraction
            pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                if i < 2:  # Show first 2 pages
                    print(f"Page {i+1} extracted text (first 200 chars): {page_text[:200]}...")
            
            print(f"✅ Total extracted text length: {len(text)} characters")
            return text
            
    except Exception as e:
        print(f"❌ Error downloading/processing PDF: {e}")
        return None

def test_gemini_api():
    """Test Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"API Key loaded: {api_key[:20]}..." if api_key else "❌ No API key found")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        test_prompt = "Hello, can you respond with 'Gemini API is working correctly'?"
        response = model.generate_content(test_prompt)
        print(f"✅ Gemini API test successful: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

async def main():
    print("🔍 Starting component tests...\n")
    
    # Test Gemini API
    print("1. Testing Gemini API...")
    gemini_works = test_gemini_api()
    print()
    
    # Test PDF download and extraction
    print("2. Testing PDF download and extraction...")
    pdf_text = await test_pdf_download()
    print()
    
    if gemini_works and pdf_text:
        print("3. Testing full RAG pipeline...")
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Take a sample of the text
            sample_text = pdf_text[:2000]  # First 2000 characters
            question = "What type of document is this?"
            
            prompt = f"""
You are an expert assistant that answers questions based on provided context. 
Answer the question using only the information from the context provided. 

Context:
{sample_text}

Question: {question}

Answer:"""
            
            response = model.generate_content(prompt)
            print(f"✅ Full RAG test successful: {response.text[:200]}...")
            
        except Exception as e:
            print(f"❌ Full RAG test failed: {e}")
    
    print("\n🎯 Component testing complete!")

if __name__ == "__main__":
    asyncio.run(main())
