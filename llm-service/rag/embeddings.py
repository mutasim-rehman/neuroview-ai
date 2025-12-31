"""
Embedding model for document and query vectorization.

Uses sentence-transformers for efficient embedding generation.
Optimized for medical/scientific text retrieval.
"""

import logging
from typing import List, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding model wrapper for RAG system.
    
    Supports multiple embedding backends:
    - sentence-transformers (default)
    - HuggingFace transformers
    - OpenAI embeddings (optional)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on ('cpu', 'cuda', 'auto')
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device if self.device != "auto" else None
            )
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar for batch encoding
            
        Returns:
            NumPy array of embeddings (N x embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return self.encode(query, normalize=True)[0]
    
    def encode_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of documents.
        
        Args:
            documents: List of document texts
            show_progress: Show progress bar
            
        Returns:
            Document embeddings matrix
        """
        return self.encode(
            documents,
            normalize=True,
            show_progress=show_progress
        )
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Document embeddings matrix
            
        Returns:
            Similarity scores
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        return np.dot(document_embeddings, query_embedding)
    
    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


class MedicalEmbeddingModel(EmbeddingModel):
    """
    Specialized embedding model for medical/scientific text.
    
    Uses PubMedBERT or similar medical-domain embeddings for
    better retrieval of medical literature.
    """
    
    def __init__(
        self,
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        device: str = "auto",
        batch_size: int = 16  # Smaller batch for larger model
    ):
        super().__init__(model_name, device, batch_size)
    
    def encode_medical_query(self, query: str, disease_context: Optional[str] = None) -> np.ndarray:
        """
        Encode a medical query with optional disease context.
        
        Args:
            query: Query text
            disease_context: Optional disease name for context
            
        Returns:
            Query embedding vector
        """
        if disease_context:
            query = f"[{disease_context}] {query}"
        return self.encode_query(query)

