from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env from parent directory (v2-rag-from-scratch/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Check if running on Vercel
IS_VERCEL = os.environ.get("VERCEL") == "1"


# ============== RAG Implementation ==============

# Sample study material embedded directly in the code
STUDY_MATERIAL = """
Mitosis: Cell Division Explained

Mitosis is the process by which a single cell divides to produce two identical daughter cells. It is essential for growth, repair, and maintenance of living organisms. Mitosis ensures that each new cell receives an exact copy of the parent cell's genetic material.

The Phases of Mitosis

Mitosis consists of four main phases: prophase, metaphase, anaphase, and telophase. Each phase has distinct characteristics and events.

Prophase

During prophase, the chromatin condenses into visible chromosomes. Each chromosome consists of two sister chromatids joined at the centromere. The nuclear envelope begins to break down, and the mitotic spindle starts to form from the centrioles. The spindle fibers extend from the centrioles toward the center of the cell.

Metaphase

In metaphase, the chromosomes align along the cell's equator, forming what is called the metaphase plate. The spindle fibers attach to the centromeres of each chromosome. This alignment ensures that each daughter cell will receive one copy of each chromosome.

Anaphase

Anaphase begins when the sister chromatids separate at the centromere. The spindle fibers shorten, pulling the separated chromatids toward opposite poles of the cell. By the end of anaphase, each pole has a complete set of chromosomes.

Telophase

During telophase, the chromosomes arrive at the poles and begin to decondense back into chromatin. The nuclear envelope reforms around each set of chromosomes, creating two separate nuclei. The spindle fibers disassemble.

Cytokinesis

Following mitosis, cytokinesis divides the cytoplasm, producing two separate daughter cells. In animal cells, a cleavage furrow forms and pinches the cell in two. In plant cells, a cell plate forms between the two nuclei and develops into a new cell wall.

Photosynthesis: Converting Light to Energy

Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This process is fundamental to life on Earth, as it produces oxygen and forms the base of most food chains.

The Overall Equation

The overall equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. This means six molecules of carbon dioxide and six molecules of water, using light energy, produce one molecule of glucose and six molecules of oxygen.

Light-Dependent Reactions

The light-dependent reactions occur in the thylakoid membranes of chloroplasts. Chlorophyll and other pigments absorb light energy, which is used to split water molecules (photolysis), releasing oxygen as a byproduct. This process generates ATP and NADPH, which are energy carriers used in the next stage.

The Calvin Cycle

The Calvin Cycle, also called the light-independent reactions, occurs in the stroma of chloroplasts. It uses the ATP and NADPH from the light-dependent reactions to fix carbon dioxide into glucose. The cycle involves three main stages: carbon fixation, reduction, and regeneration of the starting molecule RuBP.

The Water Cycle

The water cycle, also known as the hydrological cycle, describes the continuous movement of water on, above, and below Earth's surface. It is driven primarily by solar energy and gravity.

Evaporation and Transpiration

Water evaporates from oceans, lakes, and rivers when heated by the sun. Plants also release water vapor through transpiration from their leaves. This water vapor rises into the atmosphere.

Condensation

As water vapor rises and cools, it condenses around tiny particles in the atmosphere to form clouds. This process releases heat energy into the atmosphere.

Precipitation

When water droplets in clouds become too heavy, they fall as precipitation—rain, snow, sleet, or hail. Precipitation replenishes surface water and groundwater supplies.

Collection and Runoff

Precipitation collects in oceans, lakes, rivers, and underground aquifers. Some water flows over land as runoff, eventually returning to larger bodies of water. The cycle then repeats continuously.
"""


class SimpleVectorDB:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add(self, vector: List[float], text: str, meta: dict):
        self.vectors.append(np.array(vector))
        self.texts.append(text)
        self.metadata.append(meta)

    def search(self, query_vector: List[float], k: int = 5):
        if not self.vectors:
            return []

        query = np.array(query_vector)
        similarities = []

        for vec in self.vectors:
            similarity = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
            similarities.append(similarity)

        top_k = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k:
            results.append({'text': self.texts[idx], 'metadata': self.metadata[idx],
                           'score': float(similarities[idx])})

        return results


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))

    return chunks


# Initialize vector database (client initialized lazily)
vector_db = SimpleVectorDB()
_client = None


def get_client():
    """Lazy initialization of OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed_text(text: str) -> List[float]:
    response = get_client().embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    return response.data[0].embedding


def index_document(text: str, doc_name: str):
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        vector_db.add(embedding, chunk, {'doc_name': doc_name, 'chunk_id': i})


def index_study_material():
    """Index the built-in study material."""
    if vector_db.vectors:
        return len(vector_db.vectors)

    index_document(STUDY_MATERIAL, "study-guide")
    return len(vector_db.vectors)


def retrieve_context(query: str, k: int = 3) -> str:
    index_study_material()

    query_embedding = embed_text(query)
    results = vector_db.search(query_embedding, k=k)

    context_parts = []
    for result in results:
        context_parts.append(f"[From {result['metadata']['doc_name']}]:\n{result['text']}")

    return '\n\n'.join(context_parts)


def answer_question(question: str) -> str:
    context = retrieve_context(question)

    prompt = f'''You are StudyBuddy, a helpful tutoring assistant.

Given the following context from study materials, answer the student's
question. Be clear and thorough, and cite which document you're
referencing when relevant.

Context:
{context}

Question: {question}'''

    response = get_client().chat.completions.create(
        model='gpt-5-nano',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )

    return response.choices[0].message.content


# ============== FastAPI App ==============

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    message: str


@app.get("/api/status")
def get_status():
    """Check indexing status."""
    return {
        "indexing_complete": len(vector_db.vectors) > 0,
        "chunks_in_db": len(vector_db.vectors)
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    try:
        reply = answer_question(request.message)
        return {"reply": reply}

    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)


# Serve frontend static files (local development only)
if not IS_VERCEL:
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
