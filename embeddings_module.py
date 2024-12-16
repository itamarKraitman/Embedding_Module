import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import fitz  
from docx import Document

class EmbeddingsModule:
    """
    A class implementing a simple Retrieval-Augmented Generation (RAG) system for document segmentation,
    embedding generation, and text retrieval.

    Attributes:
        splitting_strategy (int): The strategy for splitting text into chunks. Options are:
            0: Fixed overlap splitting.
            1: Sentence splitting.
            2: Paragraph splitting.
        chunk_size (int): The number of words in each chunk (only for fixed overlap splitting).
        overlap (int): The number of overlapping words between consecutive chunks (only for fixed overlap splitting).
        device (torch.device): The device (CPU or GPU) for running the model.
        tokenizer (transformers.AutoTokenizer): Tokenizer for processing input text.
        model (transformers.AutoModel): Pre-trained transformer model for embedding generation.
        chunks (List[str]): The chunks of text extracted from the document.
        chunk_embeddings (List[torch.Tensor]): The embeddings for the chunks, stored as a database.
    """

    def __init__(self, splitting_strategy: int = 0, chunk_size: int = 100, overlap: int = 50):
        """
        Initializes the RAGSystem with specified splitting strategy and parameters.
        Args:
            splitting_strategy (int): The strategy for splitting text (default: 0 = fixed overlap).
            chunk_size (int): The number of words per chunk for fixed overlap splitting (default: 100).
            overlap (int): The number of overlapping words between chunks for fixed overlap splitting (default: 50).
        """
        self.splitting_strategy = splitting_strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-small").to(self.device)
        self.chunks = []
        self.chunk_embeddings = []  # Database for chunk embeddings

    def _read_pdf(self, file_path: str) -> str:
        """
        Reads and extracts text from a PDF file.
        Args:
            file_path (str): Path to the PDF file.
        Returns:
            str: Extracted text from the PDF.
        """
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        return text

    def _read_docx(self, file_path: str) -> str:
        """
        Reads and extracts text from a DOCX file.
        Args:
            file_path (str): Path to the DOCX file.
        Returns:
            str: Extracted text from the DOCX file.
        """
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    def _split_fixed_overlap(self, text: str) -> List[str]:
        """
        Splits text into chunks using a fixed overlap strategy.
        Args:
            text (str): The input text to split.
        Returns:
            List[str]: List of text chunks.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunks.append(" ".join(words[i:i + self.chunk_size]))
        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Splits text into chunks by sentences.
        Args:
            text (str): The input text to split.
        Returns:
            List[str]: List of sentences.
        """
        sentences = text.split('.')
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Splits text into chunks by paragraphs.
        Args:
            text (str): The input text to split.
        Returns:
            List[str]: List of paragraphs.
        """
        paragraphs = text.split('\n')
        return [para.strip() for para in paragraphs if para.strip()]

    def _split_text(self, text: str) -> List[str]:
        """
        Splits text based on the selected splitting strategy.
        Args:
            text (str): The input text to split.
        Returns:
            List[str]: List of text chunks.
        Raises:
            ValueError: If an invalid splitting strategy is specified.
        """
        if self.splitting_strategy == 0:  # Fixed overlap
            return self._split_fixed_overlap(text)
        elif self.splitting_strategy == 1:  # Sentence splitter
            return self._split_by_sentences(text)
        elif self.splitting_strategy == 2:  # Paragraph splitter
            return self._split_by_paragraphs(text)
        else:
            raise ValueError(f"Unknown splitting strategy: {self.splitting_strategy}")

    def _embed_text(self, text: str) -> torch.Tensor:
        """
        Generates an embedding for a given text using the BERT model.
        Args:
            text (str): The input text to embed.
        Returns:
            torch.Tensor: The embedding of the input text as a PyTorch tensor.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def process_document(self, file_path: str, file_format: str):
        """
        Processes a document by reading, splitting, and generating embeddings for its content.
        Args:
            file_path (str): Path to the document file.
            file_format (str): Format of the document ('pdf' or 'docx').
        Raises:
            ValueError: If an unsupported file format is provided.
        """
        if file_format == "pdf":
            text = self._read_pdf(file_path)
        elif file_format == "docx":
            text = self._read_docx(file_path)
        else:
            raise ValueError("Unsupported file format. Use 'pdf' or 'docx'.")

        self.chunks = self._split_text(text)
        self.chunk_embeddings = [self._embed_text(chunk) for chunk in self.chunks]

    def retrieve(self, prompt: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Retrieves the most relevant chunks from the processed document based on a prompt.
        Args:
            prompt (str): The query or prompt for retrieval.
            top_k (int): The number of top matches to return (default: 1).
        Returns:
            List[Tuple[str, float]]: A list of tuples containing the most relevant chunks and their similarity scores.
        """
        query_embedding = self._embed_text(prompt)

        # Vectorized cosine similarity computation on GPU
        chunk_embeddings_tensor = torch.stack(self.chunk_embeddings).to(self.device)
        query_embedding = query_embedding.to(self.device)

        similarities = torch.nn.functional.cosine_similarity(chunk_embeddings_tensor, query_embedding.unsqueeze(0))

        top_k_indices = torch.topk(similarities, k=top_k).indices.tolist()
        return [(self.chunks[i], similarities[i].item()) for i in top_k_indices]


if __name__ == "__main__":
    embed = EmbeddingsModule(splitting_strategy=0, chunk_size=50, overlap=25)  # 0 = fixed_overlap default
    embed.process_document("sample.pdf", file_format="pdf")
    results = embed.retrieve("What is the content about?", top_k=1)
    print("Top Matches:")
    for chunk, similarity in results:
        print(f"Similarity: {similarity:.4f}, Chunk: {chunk}")
