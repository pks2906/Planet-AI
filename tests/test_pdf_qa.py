# tests/test_pdf_qa.py

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import asyncio
import websockets
import json
import os
from datetime import datetime

from app.main import app
from app.database import Base, get_db
from app.models.models import Document, ExtractedText, UserSession

# Create test database
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Test data
TEST_PDF_PATH = "tests/test_files/sample.pdf"

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_client():
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

def test_upload_document(test_client, test_db):
    # Test file upload
    with open(TEST_PDF_PATH, "rb") as f:
        response = test_client.post(
            "/upload/",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    assert "document_id" in response.json()

@pytest.mark.asyncio
async def test_websocket_connection():
    # Test WebSocket connection
    uri = "ws://localhost:8000/ws/1"  # Assuming document_id = 1
    async with websockets.connect(uri) as websocket:
        # Test sending a question
        await websocket.send(json.dumps({
            "type": "question",
            "content": "What is this document about?"
        }))
        
        response = await websocket.recv()
        response_data = json.dumps(response)
        assert "type" in response_data
        assert "content" in response_data

def test_document_processing(test_client, test_db):
    # Test document processing status
    db = next(override_get_db())
    doc = Document(
        filename="test.pdf",
        file_path=TEST_PDF_PATH,
        upload_date=datetime.utcnow()
    )
    db.add(doc)
    db.commit()
    
    response = test_client.get(f"/document/{doc.id}/status")
    assert response.status_code == 200
    assert "status" in response.json()

