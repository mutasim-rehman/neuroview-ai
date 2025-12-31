"""
Document Store for Medical Knowledge Base.

Handles document loading, chunking, and vector storage.
Supports multiple vector store backends (ChromaDB, FAISS).
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }


class DocumentStore:
    """
    Document store for RAG system.
    
    Manages document storage, chunking, and retrieval
    using vector similarity search.
    """
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "neuroview_medical",
        embedding_model: Optional[Any] = None,
        vector_store_type: str = "chroma"
    ):
        """
        Initialize the document store.
        
        Args:
            persist_directory: Directory to persist vector store
            collection_name: Name of the vector collection
            embedding_model: EmbeddingModel instance
            vector_store_type: Type of vector store ('chroma', 'faiss')
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the vector store backend."""
        if self.vector_store_type == "chroma":
            self._init_chroma()
        elif self.vector_store_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def _init_chroma(self):
        """Initialize ChromaDB backend."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with {self._collection.count()} documents")
            
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise
    
    def _init_faiss(self):
        """Initialize FAISS backend."""
        try:
            import faiss
            
            # FAISS index will be initialized when first document is added
            self._index = None
            self._documents = []
            self._faiss_index_path = self.persist_directory / "faiss_index.bin"
            self._documents_path = self.persist_directory / "documents.json"
            
            # Load existing index if available
            if self._faiss_index_path.exists():
                self._index = faiss.read_index(str(self._faiss_index_path))
                with open(self._documents_path, 'r') as f:
                    self._documents = json.load(f)
                logger.info(f"FAISS loaded with {len(self._documents)} documents")
            
        except ImportError:
            logger.error("faiss not installed. Run: pip install faiss-cpu")
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the store.
        
        Args:
            documents: List of Document objects
            batch_size: Batch size for processing
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Generate embeddings if not present
        texts_to_embed = [doc.content for doc in documents if doc.embedding is None]
        if texts_to_embed and self.embedding_model:
            embeddings = self.embedding_model.encode_documents(texts_to_embed)
            embed_idx = 0
            for doc in documents:
                if doc.embedding is None:
                    doc.embedding = embeddings[embed_idx].tolist()
                    embed_idx += 1
        
        if self.vector_store_type == "chroma":
            return self._add_to_chroma(documents, batch_size)
        elif self.vector_store_type == "faiss":
            return self._add_to_faiss(documents, batch_size)
        
        return 0
    
    def _add_to_chroma(self, documents: List[Document], batch_size: int) -> int:
        """Add documents to ChromaDB."""
        added = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            self._collection.add(
                ids=[doc.id for doc in batch],
                documents=[doc.content for doc in batch],
                embeddings=[doc.embedding for doc in batch if doc.embedding],
                metadatas=[doc.metadata for doc in batch]
            )
            added += len(batch)
        
        logger.info(f"Added {added} documents to ChromaDB")
        return added
    
    def _add_to_faiss(self, documents: List[Document], batch_size: int) -> int:
        """Add documents to FAISS."""
        import faiss
        import numpy as np
        
        embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)
        
        if self._index is None:
            # Initialize index
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized)
        
        self._index.add(embeddings)
        self._documents.extend([doc.to_dict() for doc in documents])
        
        # Persist
        faiss.write_index(self._index, str(self._faiss_index_path))
        with open(self._documents_path, 'w') as f:
            json.dump(self._documents, f)
        
        logger.info(f"Added {len(documents)} documents to FAISS")
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if self.vector_store_type == "chroma":
            return self._search_chroma(query, top_k, filter_metadata)
        elif self.vector_store_type == "faiss":
            return self._search_faiss(query, top_k, filter_metadata)
        
        return []
    
    def _search_chroma(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[Document, float]]:
        """Search ChromaDB."""
        # Get query embedding
        query_embedding = None
        if self.embedding_model:
            query_embedding = self.embedding_model.encode_query(query).tolist()
        
        # Build where clause
        where = filter_metadata if filter_metadata else None
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding] if query_embedding else None,
            query_texts=[query] if not query_embedding else None,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to Document objects
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                )
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # For cosine distance
                documents.append((doc, similarity))
        
        return documents
    
    def _search_faiss(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[Document, float]]:
        """Search FAISS index."""
        import numpy as np
        
        if self._index is None or not self._documents:
            return []
        
        # Get query embedding
        query_embedding = self.embedding_model.encode_query(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search
        scores, indices = self._index.search(query_embedding, top_k)
        
        # Convert to Document objects
        documents = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._documents):
                doc_dict = self._documents[idx]
                
                # Apply metadata filter
                if filter_metadata:
                    match = all(
                        doc_dict.get("metadata", {}).get(k) == v
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue
                
                doc = Document(
                    id=doc_dict["id"],
                    content=doc_dict["content"],
                    metadata=doc_dict.get("metadata", {})
                )
                documents.append((doc, float(score)))
        
        return documents
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        if self.vector_store_type == "chroma":
            return self._collection.count()
        elif self.vector_store_type == "faiss":
            return len(self._documents)
        return 0
    
    def clear(self):
        """Clear all documents from the store."""
        if self.vector_store_type == "chroma":
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        elif self.vector_store_type == "faiss":
            self._index = None
            self._documents = []
            if self._faiss_index_path.exists():
                self._faiss_index_path.unlink()
            if self._documents_path.exists():
                self._documents_path.unlink()
        
        logger.info("Document store cleared")


class DocumentProcessor:
    """
    Processes raw documents for ingestion into the document store.
    
    Handles:
    - Text extraction from various formats
    - Chunking with overlap
    - Metadata extraction
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_text(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process raw text into document chunks.
        
        Args:
            text: Raw text content
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            List of Document chunks
        """
        chunks = self._chunk_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc_id = self._generate_id(source, i)
            doc_metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            
            documents.append(Document(
                id=doc_id,
                content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for sep in ['. ', '.\n', '\n\n', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _generate_id(self, source: str, chunk_index: int) -> str:
        """Generate a unique document ID."""
        content = f"{source}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def process_file(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a file into document chunks.
        
        Args:
            file_path: Path to the file
            metadata: Additional metadata
            
        Returns:
            List of Document chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        # Read file based on extension
        text = ""
        if file_path.suffix in ['.txt', '.md']:
            text = file_path.read_text(encoding='utf-8')
        elif file_path.suffix == '.json':
            data = json.loads(file_path.read_text(encoding='utf-8'))
            text = self._json_to_text(data)
        elif file_path.suffix == '.pdf':
            text = self._extract_pdf(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
        file_metadata = {
            "filename": file_path.name,
            "file_type": file_path.suffix,
            **(metadata or {})
        }
        
        return self.process_text(text, str(file_path), file_metadata)
    
    def _json_to_text(self, data: Any) -> str:
        """Convert JSON data to text."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            parts = []
            for key, value in data.items():
                parts.append(f"{key}: {self._json_to_text(value)}")
            return "\n".join(parts)
        elif isinstance(data, list):
            return "\n".join(self._json_to_text(item) for item in data)
        else:
            return str(data)
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2
            
            text_parts = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text())
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.warning("PyPDF2 not installed. Run: pip install PyPDF2")
            return ""
        except Exception as e:
            logger.error(f"Failed to extract PDF: {e}")
            return ""

