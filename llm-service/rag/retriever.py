"""
Medical Knowledge Retriever for NeuroView LLM.

Provides specialized retrieval for neurological disease information
with context-aware query processing.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .document_store import DocumentStore, Document
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    
    query: str
    documents: List[Document]
    scores: List[float]
    context: str  # Combined context for LLM
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "num_documents": len(self.documents),
            "top_score": self.scores[0] if self.scores else 0,
            "context_length": len(self.context),
            "metadata": self.metadata
        }


class MedicalRetriever:
    """
    Specialized retriever for medical/neurological knowledge.
    
    Features:
    - Disease-aware query expansion
    - Multi-source retrieval
    - Context assembly for LLM prompts
    - Relevance filtering
    """
    
    # Disease-specific keywords for query expansion
    DISEASE_KEYWORDS = {
        "glioma": [
            "glioma", "brain tumor", "astrocytoma", "oligodendroglioma",
            "glioblastoma", "GBM", "WHO grade", "IDH mutation",
            "infiltrative", "white matter", "enhancement"
        ],
        "meningioma": [
            "meningioma", "meningeal tumor", "dura mater", "extra-axial",
            "calcification", "hyperostosis", "dural tail", "WHO grade"
        ],
        "pituitary_tumor": [
            "pituitary", "adenoma", "sella turcica", "hypophysis",
            "prolactinoma", "acromegaly", "Cushing", "visual field",
            "optic chiasm", "hormone secreting"
        ],
        "brain_metastases": [
            "metastasis", "metastatic", "secondary tumor", "primary cancer",
            "multiple lesions", "edema", "ring enhancement", "gray-white junction"
        ],
        "alzheimer": [
            "Alzheimer", "dementia", "neurodegeneration", "cognitive decline",
            "hippocampal atrophy", "temporal lobe", "amyloid", "tau",
            "memory loss", "cortical atrophy"
        ],
        "healthy_brain": [
            "normal brain", "healthy", "no abnormality", "unremarkable",
            "normal anatomy", "reference", "baseline"
        ]
    }
    
    def __init__(
        self,
        document_store: DocumentStore,
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        max_context_length: int = 3000
    ):
        """
        Initialize the medical retriever.
        
        Args:
            document_store: Document store instance
            embedding_model: Embedding model instance
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity for inclusion
            max_context_length: Maximum context length for LLM
        """
        self.document_store = document_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
    
    def retrieve(
        self,
        query: str,
        disease: Optional[str] = None,
        expand_query: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            disease: Optional disease context for focused retrieval
            expand_query: Whether to expand query with disease keywords
            filter_metadata: Optional metadata filters
            
        Returns:
            RetrievalResult with documents and assembled context
        """
        # Expand query if disease context is provided
        search_query = query
        if expand_query and disease:
            search_query = self._expand_query(query, disease)
        
        # Add disease filter if provided
        if disease and filter_metadata is None:
            filter_metadata = {"disease": disease}
        elif disease and filter_metadata:
            filter_metadata["disease"] = disease
        
        # Search document store
        results = self.document_store.search(
            query=search_query,
            top_k=self.top_k * 2,  # Retrieve more, filter later
            filter_metadata=filter_metadata
        )
        
        # Filter by similarity threshold
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= self.similarity_threshold
        ][:self.top_k]
        
        # Separate documents and scores
        documents = [doc for doc, _ in filtered_results]
        scores = [score for _, score in filtered_results]
        
        # Assemble context for LLM
        context = self._assemble_context(documents, disease)
        
        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores,
            context=context,
            metadata={
                "disease": disease,
                "expanded_query": search_query if expand_query else None,
                "num_results_before_filter": len(results),
                "num_results_after_filter": len(filtered_results)
            }
        )
    
    def _expand_query(self, query: str, disease: str) -> str:
        """
        Expand query with disease-specific keywords.
        
        Args:
            query: Original query
            disease: Disease name
            
        Returns:
            Expanded query
        """
        disease_key = disease.lower().replace(" ", "_")
        keywords = self.DISEASE_KEYWORDS.get(disease_key, [])
        
        if keywords:
            # Add top keywords to query
            top_keywords = keywords[:3]
            expanded = f"{query} {' '.join(top_keywords)}"
            return expanded
        
        return query
    
    def _assemble_context(
        self,
        documents: List[Document],
        disease: Optional[str] = None
    ) -> str:
        """
        Assemble retrieved documents into context for LLM.
        
        Args:
            documents: Retrieved documents
            disease: Disease context
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        # Add disease header if provided
        if disease:
            header = f"Medical Information for {disease.replace('_', ' ').title()}:\n"
            context_parts.append(header)
            current_length += len(header)
        
        # Add document contents
        for i, doc in enumerate(documents):
            # Format document
            source = doc.metadata.get("source", "Unknown")
            content = doc.content.strip()
            
            doc_text = f"\n[Source {i+1}: {source}]\n{content}\n"
            
            # Check length limit
            if current_length + len(doc_text) > self.max_context_length:
                # Truncate if needed
                remaining = self.max_context_length - current_length - 50
                if remaining > 100:
                    doc_text = f"\n[Source {i+1}: {source}]\n{content[:remaining]}...\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts)
    
    def retrieve_for_prediction(
        self,
        disease: str,
        location: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> RetrievalResult:
        """
        Retrieve context for a disease prediction from the vision model.
        
        Args:
            disease: Predicted disease name
            location: Predicted anatomical location
            confidence: Prediction confidence score
            
        Returns:
            RetrievalResult with comprehensive disease context
        """
        # Build structured query for prediction context
        query_parts = [f"Detailed information about {disease}"]
        
        if location:
            query_parts.append(f"located in {location}")
        
        query = " ".join(query_parts)
        
        # Retrieve with disease context
        result = self.retrieve(
            query=query,
            disease=disease,
            expand_query=True
        )
        
        # Add prediction metadata
        result.metadata["prediction_confidence"] = confidence
        result.metadata["anatomical_location"] = location
        
        return result
    
    def get_disease_overview(self, disease: str) -> RetrievalResult:
        """
        Get a comprehensive overview of a specific disease.
        
        Args:
            disease: Disease name
            
        Returns:
            RetrievalResult with disease overview
        """
        overview_queries = [
            f"What is {disease}? Definition and classification",
            f"{disease} MRI imaging features and appearance",
            f"{disease} causes risk factors etiology",
            f"{disease} treatment options and management"
        ]
        
        all_documents = []
        all_scores = []
        
        for query in overview_queries:
            result = self.retrieve(
                query=query,
                disease=disease,
                expand_query=False
            )
            
            for doc, score in zip(result.documents, result.scores):
                if doc.id not in [d.id for d in all_documents]:
                    all_documents.append(doc)
                    all_scores.append(score)
        
        # Sort by score and limit
        sorted_pairs = sorted(
            zip(all_documents, all_scores),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]
        
        documents = [doc for doc, _ in sorted_pairs]
        scores = [score for _, score in sorted_pairs]
        
        context = self._assemble_context(documents, disease)
        
        return RetrievalResult(
            query=f"Overview of {disease}",
            documents=documents,
            scores=scores,
            context=context,
            metadata={"disease": disease, "query_type": "overview"}
        )


class HybridRetriever(MedicalRetriever):
    """
    Hybrid retriever combining dense and sparse retrieval.
    
    Uses both:
    - Dense: Semantic similarity (embeddings)
    - Sparse: Keyword matching (BM25)
    """
    
    def __init__(
        self,
        document_store: DocumentStore,
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        max_context_length: int = 3000,
        dense_weight: float = 0.7
    ):
        super().__init__(
            document_store,
            embedding_model,
            top_k,
            similarity_threshold,
            max_context_length
        )
        self.dense_weight = dense_weight
        self.sparse_weight = 1 - dense_weight
        self._bm25_index = None
    
    def _init_sparse_index(self, documents: List[Document]):
        """Initialize BM25 index for sparse retrieval."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents
            tokenized = [doc.content.lower().split() for doc in documents]
            self._bm25_index = BM25Okapi(tokenized)
            self._bm25_documents = documents
            
        except ImportError:
            logger.warning("rank_bm25 not installed. Using dense retrieval only.")
    
    def hybrid_search(
        self,
        query: str,
        disease: Optional[str] = None
    ) -> RetrievalResult:
        """
        Perform hybrid dense + sparse search.
        
        Args:
            query: Search query
            disease: Optional disease context
            
        Returns:
            RetrievalResult with hybrid-ranked documents
        """
        # Get dense results
        dense_result = self.retrieve(
            query=query,
            disease=disease,
            expand_query=True
        )
        
        # If BM25 not available, return dense results
        if self._bm25_index is None:
            return dense_result
        
        # Get sparse results
        tokenized_query = query.lower().split()
        sparse_scores = self._bm25_index.get_scores(tokenized_query)
        
        # Normalize sparse scores
        max_sparse = max(sparse_scores) if max(sparse_scores) > 0 else 1
        sparse_scores = [s / max_sparse for s in sparse_scores]
        
        # Combine scores
        combined_results = []
        for i, doc in enumerate(self._bm25_documents):
            # Find dense score for this document
            dense_score = 0
            for d, s in zip(dense_result.documents, dense_result.scores):
                if d.id == doc.id:
                    dense_score = s
                    break
            
            sparse_score = sparse_scores[i]
            combined_score = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score
            )
            
            if combined_score >= self.similarity_threshold:
                combined_results.append((doc, combined_score))
        
        # Sort and limit
        combined_results.sort(key=lambda x: x[1], reverse=True)
        combined_results = combined_results[:self.top_k]
        
        documents = [doc for doc, _ in combined_results]
        scores = [score for _, score in combined_results]
        
        context = self._assemble_context(documents, disease)
        
        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores,
            context=context,
            metadata={
                "disease": disease,
                "search_type": "hybrid",
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight
            }
        )

