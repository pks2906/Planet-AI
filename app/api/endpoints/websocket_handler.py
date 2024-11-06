from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List
from datetime import datetime, timedelta
import json
import asyncio
from app.core.nlp_processor import NLPProcessor
from app.database import get_db
from sqlalchemy.orm import Session
from app.models.models import Document, UserSession
import uuid

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.chat_history: Dict[str, List] = {}
        self.nlp_processor = NLPProcessor()
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.MAX_REQUESTS_PER_MINUTE = 30

    async def connect(self, websocket: WebSocket, session_id: str, document_id: int, db: Session):
        await websocket.accept()
        
        # Create or update user session
        session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
        if not session:
            session = UserSession(
                session_id=session_id,
                document_id=document_id,
                last_activity=datetime.utcnow()
            )
            db.add(session)
        else:
            session.last_activity = datetime.utcnow()
        db.commit()

        if document_id not in self.active_connections:
            self.active_connections[document_id] = {}
        self.active_connections[document_id][session_id] = websocket
        self.chat_history[session_id] = []
        self.rate_limits[session_id] = []

    def disconnect(self, session_id: str, document_id: int, db: Session):
        if document_id in self.active_connections:
            self.active_connections[document_id].pop(session_id, None)
        self.chat_history.pop(session_id, None)
        self.rate_limits.pop(session_id, None)

        # Update session last activity
        session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
        if session:
            session.last_activity = datetime.utcnow()
            db.commit()

    async def check_rate_limit(self, session_id: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        self.rate_limits[session_id] = [
            time for time in self.rate_limits[session_id]
            if time > minute_ago
        ]
        
        if len(self.rate_limits[session_id]) >= self.MAX_REQUESTS_PER_MINUTE:
            return False
        
        self.rate_limits[session_id].append(now)
        return True

    async def process_message(self, message: str, session_id: str, document_id: int, db: Session):
        try:
            if not await self.check_rate_limit(session_id):
                await self.send_error(session_id, document_id, "Rate limit exceeded")
                return

            # Update session last activity
            session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
            if session:
                session.last_activity = datetime.utcnow()
                db.commit()

            # Check document status
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                await self.send_error(session_id, document_id, "Document not found")
                return
            
            if document.processed_status != "completed":
                await self.send_error(session_id, document_id, f"Document processing {document.processed_status}")
                return

            answer = await self.nlp_processor.get_answer(
                document_id,
                message,
                self.chat_history[session_id]
            )

            self.chat_history[session_id].append((message, answer))
            await self.send_personal_message(answer, session_id, document_id)

        except Exception as e:
            await self.send_error(session_id, document_id, str(e))

    async def send_personal_message(self, message: str, session_id: str, document_id: int):
        if document_id in self.active_connections and session_id in self.active_connections[document_id]:
            await self.active_connections[document_id][session_id].send_text(
                json.dumps({"message": message, "type": "answer"})
            )

    async def send_error(self, session_id: str, document_id: int, error: str):
        if document_id in self.active_connections and session_id in self.active_connections[document_id]:
            await self.active_connections[document_id][session_id].send_text(
                json.dumps({"error": error, "type": "error"})
            )

manager = ConnectionManager()

async def websocket_endpoint(
    websocket: WebSocket,
    document_id: int,
    db: Session = Depends(get_db)
):
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, session_id, document_id, db)
    try:
        while True:
            message = await websocket.receive_text()
            await manager.process_message(message, session_id, document_id, db)
    except WebSocketDisconnect:
        manager.disconnect(session_id, document_id, db)