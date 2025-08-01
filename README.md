# HackRx 6.0 - RAG Q&A API

A FastAPI-based application for document analysis and question answering using RAG (Retrieval Augmented Generation) with Google Gemini AI.

## Features

- PDF document processing and text extraction
- Intelligent chunking and vector storage using ChromaDB
- Advanced entity extraction and semantic query generation
- Integration with Google Gemini AI for answer generation
- RESTful API with FastAPI

## Deployment on Vercel

This application is configured for serverless deployment on Vercel.

### Prerequisites

1. A GitHub account
2. A Vercel account (free tier available)
3. Google Gemini API key

### Deployment Steps

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy to Vercel"
   git push origin main
   ```

2. **Deploy to Vercel**:
   - Visit [vercel.com](https://vercel.com) and sign in
   - Click "New Project"
   - Import your GitHub repository
   - Configure environment variables:
     - `GEMINI_API_KEY`: Your Google Gemini API key
     - `PORT`: 8000 (optional, already set in vercel.json)

3. **Access your deployed API**:
   - Your API will be available at `https://your-project-name.vercel.app`
   - Health check: `GET https://your-project-name.vercel.app/health`
   - Main endpoint: `POST https://your-project-name.vercel.app/api/v1/hackrx/run`

## API Usage

### Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `POST /api/v1/hackrx/run` - Main processing endpoint
- `POST /hackrx/run` - Legacy endpoint (same functionality)

### Request Format

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the coverage for heart surgery?",
    "What are the waiting periods?"
  ]
}
```

### Response Format

```json
{
  "answers": [
    "Heart surgery is covered under...",
    "The waiting periods are..."
  ]
}
```

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Environment Variables

- `GEMINI_API_KEY`: Required for AI-powered answer generation
- `PORT`: Optional, defaults to 8000

## Architecture

The application uses:
- **FastAPI** for the web framework
- **ChromaDB** for vector storage (in-memory for serverless)
- **Google Gemini AI** for answer generation
- **PyPDF2** for PDF text extraction
- **httpx** for async HTTP requests
