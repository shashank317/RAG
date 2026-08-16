import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# Project root setup
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# ---------------------------------------------------------
# RAG Search Class
# ---------------------------------------------------------
class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemini-3.7-flash"
    ):

        # Store model name for debugging
        self.llm_model = llm_model

        # -------------------------------------------------
        # Initialize vector store
        # -------------------------------------------------
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        # -------------------------------------------------
        # Load existing FAISS index or build a new one
        # -------------------------------------------------
        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):
            print("[INFO] FAISS index not found. Building vector store...")

            from src.data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        else:
            self.vectorstore.load()

        # -------------------------------------------------
        # Get Gemini API key
        # -------------------------------------------------
        google_api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        print(
            f"[DEBUG] .env path: "
            f"{PROJECT_ROOT / '.env'}"
        )

        print(
            f"[DEBUG] GEMINI_API_KEY set: "
            f"{bool(os.getenv('GEMINI_API_KEY'))}"
        )

        print(
            f"[DEBUG] GOOGLE_API_KEY set: "
            f"{bool(os.getenv('GOOGLE_API_KEY'))}"
        )

        print(
            f"[DEBUG] Selected model: "
            f"{self.llm_model}"
        )

        # -------------------------------------------------
        # Validate API key
        # -------------------------------------------------
        if not google_api_key:
            raise ValueError(
                "Missing Gemini API key. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "in the .env file."
            )

        # -------------------------------------------------
        # Initialize Gemini
        # -------------------------------------------------
        try:

            self.llm = ChatGoogleGenerativeAI(
                google_api_key=google_api_key,
                model=self.llm_model
            )

            print(
                f"[INFO] Google Generative AI LLM "
                f"initialized: {self.llm_model}"
            )

        except Exception as e:

            print(
                f"[ERROR] Failed to initialize Gemini "
                f"model '{self.llm_model}'."
            )

            print(
                "[ERROR] This usually means the model "
                "name is invalid or the API key is not "
                "usable for that model."
            )

            raise


    # -----------------------------------------------------
    # Search documents and summarize using Gemini
    # -----------------------------------------------------
    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # -------------------------------------------------
        # Retrieve relevant documents from FAISS
        # -------------------------------------------------
        print(
            f"[INFO] Querying vector store for: "
            f"'{query}'"
        )

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        # -------------------------------------------------
        # Extract text from retrieved documents
        # -------------------------------------------------
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]

        # Combine retrieved chunks into one context
        context = "\n\n".join(texts)

        # -------------------------------------------------
        # Handle no results
        # -------------------------------------------------
        if not context:
            return "No relevant documents found."

        # -------------------------------------------------
        # Create prompt for Gemini
        # -------------------------------------------------
        prompt = f"""
Summarize the following context for the query:

Query:
{query}

Context:
{context}

Summary:
"""

        print(
            f"[DEBUG] Sending prompt to LLM with model: "
            f"{self.llm_model}"
        )

        # -------------------------------------------------
        # Send prompt to Gemini
        # -------------------------------------------------
        try:

            response = self.llm.invoke(prompt)

            # -------------------------------------------------
            # Extract clean text from Gemini response
            # -------------------------------------------------
            if isinstance(response.content, list):

                text = "\n".join(
                    item.get("text", "")
                    for item in response.content
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                    )
                )

            else:

                text = response.content

            return text.strip()

        except Exception as e:

            print(
                "[ERROR] LLM request failed "
                "during invoke()."
            )

            print(
                f"[ERROR] Details: "
                f"{type(e).__name__}: {e}"
            )

            print(
                "[ERROR] Check: .env key validity, "
                "API quota, and whether the selected "
                "model is still available."
            )

            raise


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":

    rag_search = RAGSearch()

    query = "nosql"

    summary = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    # -----------------------------------------------------
    # Clean terminal output
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print(summary)

    print("=" * 70)
