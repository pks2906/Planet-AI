from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import uvicorn
import os
from datetime import datetime

from app.database import get_db
from app.core.pdf_processor import PDFProcessor
from app.models import models
from app.database import engine

from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI(title="PDF Q&A Service")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PDF processor
pdf_processor = PDFProcessor(upload_dir="uploads")

@app.get("/")
async def root():
    return {"message": "Welcome to PDF Q&A Service"}

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process PDF
        file_path, text_chunks = await pdf_processor.process_pdf(content, file.filename)
        
        # Create document record
        db_document = models.Document(
            filename=file.filename,
            file_path=file_path,
            processed_status="completed"
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Store extracted text chunks
        for index, chunk in enumerate(text_chunks):
            text_record = models.ExtractedText(
                document_id=db_document.id,
                content=chunk,
                chunk_index=index
            )
            db.add(text_record)
        
        db.commit()
        
        return {
            "message": "PDF uploaded and processed successfully",
            "document_id": db_document.id,
            "filename": file.filename,
            "num_chunks": len(text_chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # TODO: Process the question and generate answer
            await manager.send_personal_message(f"You asked: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
