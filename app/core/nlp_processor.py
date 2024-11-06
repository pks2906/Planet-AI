from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from typing import List, Dict
import json
from sqlalchemy.orm import Session
from app.models.models import Document, ExtractedText
import numpy as np

class NLPProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo"
        )
        self.document_stores: Dict[int, FAISS] = {}

    async def process_document(self, document_id: int, text_content: str, db: Session):
        """Process a document and store chunks with embeddings"""
        try:
            # Update document status
            document = db.query(Document).filter(Document.id == document_id).first()
            document.processed_status = "processing"
            db.commit()

            # Split the text into chunks
            chunks = self.text_splitter.split_text(text_content)
            
            # Process chunks and store in database
            for idx, chunk in enumerate(chunks):
                # Generate embedding
                embedding_vector = self.embeddings.embed_query(chunk)
                
                # Store in database
                extracted_text = ExtractedText(
                    document_id=document_id,
                    content=chunk,
                    chunk_index=idx,
                    embedding=json.dumps(embedding_vector)  # Serialize embedding vector
                )
                db.add(extracted_text)
            
            # Create vector store
            texts_with_embeddings = db.query(ExtractedText).filter(
                ExtractedText.document_id == document_id
            ).all()
            
            texts = [t.content for t in texts_with_embeddings]
            embeddings = [np.array(json.loads(t.embedding)) for t in texts_with_embeddings]
            
            vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self.embeddings
            )
            
            self.document_stores[document_id] = vectorstore
            
            # Update document status
            document.processed_status = "completed"
            db.commit()
            
            return True
        except Exception as e:
            # Update document status on failure
            document = db.query(Document).filter(Document.id == document_id).first()
            document.processed_status = "failed"
            db.commit()
            raise e

    async def get_answer(self, document_id: int, question: str, chat_history: List = None) -> str:
        """Get answer for a question using the processed document"""
        if document_id not in self.document_stores:
            raise ValueError("Document not processed or not found")

        if chat_history is None:
            chat_history = []

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.document_stores[document_id].as_retriever(),
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        )

        response = qa_chain({"question": question, "chat_history": chat_history})
        return response["answer"]

    def cleanup_document(self, document_id: int):
        """Remove document from memory when no longer needed"""
        if document_id in self.document_stores:
            del self.document_stores[document_id]