from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import asyncio
from app.core.nlp_processor import NLPProcessor
from app.database import get_db
from sqlalchemy.orm import Session
from app.models.models import Document

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.chat_history: Dict[str, List] = {}
        self.nlp_processor = NLPProcessor()
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.MAX_REQUESTS_PER_MINUTE = 30

    async def connect(self, websocket: WebSocket, client_id: str, document_id: str):
        await websocket.accept()
        if document_id not in self.active_connections:
            self.active_connections[document_id] = {}
        self.active_connections[document_id][client_id] = websocket
        self.chat_history[client_id] = []
        self.rate_limits[client_id] = []

    def disconnect(self, client_id: str, document_id: str):
        if document_id in self.active_connections:
            self.active_connections[document_id].pop(client_id, None)
        self.chat_history.pop(client_id, None)
        self.rate_limits.pop(client_id, None)

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.rate_limits[client_id] = [
            time for time in self.rate_limits[client_id]
            if time > minute_ago
        ]
        
        # Check limit
        if len(self.rate_limits[client_id]) >= self.MAX_REQUESTS_PER_MINUTE:
            return False
        
        self.rate_limits[client_id].append(now)
        return True

    async def process_message(self, message: str, client_id: str, document_id: str, db: Session):
        """Process incoming WebSocket message"""
        try:
            # Check rate limit
            if not await self.check_rate_limit(client_id):
                await self.send_error(client_id, document_id, "Rate limit exceeded")
                return

            # Get document from database
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                await self.send_error(client_id, document_id, "Document not found")
                return

            # Get answer from NLP processor
            answer = await self.nlp_processor.get_answer(
                document_id,
                message,
                self.chat_history[client_id]
            )

            # Update chat history
            self.chat_history[client_id].append((message, answer))

            # Send response
            await self.send_personal_message(answer, client_id, document_id)

        except Exception as e:
            await self.send_error(client_id, document_id, str(e))

    async def send_personal_message(self, message: str, client_id: str, document_id: str):
        """Send message to specific client"""
        if document_id in self.active_connections and client_id in self.active_connections[document_id]:
            await self.active_connections[document_id][client_id].send_text(
                json.dumps({"message": message, "type": "answer"})
            )

    async def send_error(self, client_id: str, document_id: str, error: str):
        """Send error message to client"""
        if document_id in self.active_connections and client_id in self.active_connections[document_id]:
            await self.active_connections[document_id][client_id].send_text(
                json.dumps({"error": error, "type": "error"})
            )

manager = ConnectionManager()

async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    document_id: str,
    db: Session = Depends(get_db)
):
    await manager.connect(websocket, client_id, document_id)
    try:
        while True:
            message = await websocket.receive_text()
            await manager.process_message(message, client_id, document_id, db)
    except WebSocketDisconnect:
        manager.disconnect(client_id, document_id)