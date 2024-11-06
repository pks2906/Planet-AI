from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from typing import List, Dict
import os

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
        self.document_stores: Dict[str, FAISS] = {}

    async def process_document(self, document_id: str, text_content: str):
        """Process a document and prepare it for Q&A"""
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text_content)
        
        # Create vector store
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )
        
        self.document_stores[document_id] = vectorstore
        return True

    async def get_answer(self, document_id: str, question: str, chat_history: List = None) -> str:
        """Get answer for a question using the processed document"""
        if document_id not in self.document_stores:
            raise ValueError("Document not found")

        if chat_history is None:
            chat_history = []

        # Create a conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.document_stores[document_id].as_retriever(),
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        )

        # Get response
        response = qa_chain({"question": question, "chat_history": chat_history})
        return response["answer"]

    def cleanup_document(self, document_id: str):
        """Remove document from memory when no longer needed"""
        if document_id in self.document_stores:
            del self.document_stores[document_id]