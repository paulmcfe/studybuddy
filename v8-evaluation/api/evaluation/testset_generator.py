"""Synthetic test data generation using RAGAS.

This module generates synthetic test cases from reference documents
using the RAGAS knowledge graph approach.
"""

from pathlib import Path

# Load environment variables before any LangChain/RAGAS imports
# This ensures LANGCHAIN_TRACING_V2=false is respected
from dotenv import load_dotenv
load_dotenv()

# RAGAS uses asyncio.run() internally, which conflicts with FastAPI's event loop.
# nest_asyncio allows nested event loops to work around this.
import nest_asyncio

nest_asyncio.apply()

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd


def load_reference_documents(docs_path: str = "documents") -> list[Document]:
    """Load and chunk reference documents for test generation.

    Args:
        docs_path: Path to documents directory (relative to api/ parent)

    Returns:
        List of LangChain Document objects, chunked for test generation
    """
    # Resolve path relative to the api module's parent directory
    documents_dir = Path(__file__).parent.parent.parent / docs_path

    if not documents_dir.exists():
        print(f"Documents directory not found: {documents_dir}")
        return []

    guide_files = sorted(documents_dir.glob("ref-*.md"))
    if not guide_files:
        print("No ref-*.md files found in documents directory")
        return []

    # Use larger chunks for test generation (more context per test case)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )

    all_chunks = []
    for filepath in guide_files:
        doc_name = filepath.stem.replace("ref-", "").replace("-", " ").title()

        loader = TextLoader(str(filepath))
        documents = loader.load()
        chunks = splitter.split_documents(documents)

        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source"] = doc_name

        all_chunks.extend(chunks)

    print(f"Loaded {len(all_chunks)} chunks from {len(guide_files)} reference guides")
    return all_chunks


def create_testset_generator() -> TestsetGenerator:
    """Initialize RAGAS testset generator with configured LLMs.

    Uses GPT-4o-mini to avoid rate limits (30k TPM limit on gpt-4o is too low
    for the many calls RAGAS makes during knowledge graph extraction).

    Returns:
        Configured TestsetGenerator instance
    """
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    return TestsetGenerator(llm=generator_llm, embedding_model=embeddings)


def generate_tutoring_testset(
    documents: list[Document],
    test_size: int = 100,
) -> pd.DataFrame:
    """Generate synthetic test cases for tutoring evaluation.

    Uses RAGAS to generate diverse questions from the knowledge graph
    extracted from the documents.

    Args:
        documents: List of chunked documents to generate tests from
        test_size: Number of test cases to generate

    Returns:
        DataFrame with user_input, reference, synthesizer_name columns
    """
    if not documents:
        print("No documents provided for test generation")
        return pd.DataFrame()

    generator = create_testset_generator()

    print(f"Generating {test_size} test cases...")

    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=test_size,
    )

    df = testset.to_pandas()
    print(f"Generated {len(df)} test cases")

    # Rename columns for consistency with our evaluation system
    if "user_input" in df.columns:
        df = df.rename(columns={"user_input": "question"})
    if "reference" in df.columns:
        df = df.rename(columns={"reference": "ground_truth"})

    return df


def validate_testset(testset_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and filter generated test cases for quality.

    Removes empty, too-short, or duplicate test cases.

    Args:
        testset_df: DataFrame from testset generation

    Returns:
        Filtered DataFrame with only valid test cases
    """
    if testset_df.empty:
        return testset_df

    original_count = len(testset_df)

    # Determine column names (RAGAS v0.4 uses different names)
    question_col = "question" if "question" in testset_df.columns else "user_input"
    answer_col = "ground_truth" if "ground_truth" in testset_df.columns else "reference"

    # Remove empty/null entries
    if question_col in testset_df.columns and answer_col in testset_df.columns:
        testset_df = testset_df.dropna(subset=[question_col, answer_col])

        # Remove very short questions (likely garbage)
        testset_df = testset_df[testset_df[question_col].str.len() > 20]

        # Remove very short answers
        testset_df = testset_df[testset_df[answer_col].str.len() > 30]

        # Remove duplicates
        testset_df = testset_df.drop_duplicates(subset=[question_col])

    filtered_count = len(testset_df)
    if filtered_count < original_count:
        print(f"Filtered {original_count - filtered_count} low-quality test cases")

    return testset_df
