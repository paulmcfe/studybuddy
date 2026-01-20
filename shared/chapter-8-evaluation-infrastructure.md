# Chapter 8: Evaluation Infrastructure

We've spent seven chapters building increasingly sophisticated AI systems. Basic RAG, agentic RAG, multi-agent architectures, memory systems, deep agents with planning and delegation. These are serious capabilities. But here's a question that should be nagging at you: *How do we know any of this actually works?*

Sure, you've run your agents, seen them produce responses, maybe even shown a demo to someone who said "wow, that's cool." But "cool" isn't a metric. "Seems to work" isn't a quality bar. And "I think it's getting better" isn't a measurement strategy.

This chapter is about getting serious. We're building the evaluation infrastructure that transforms gut feelings into data, hunches into hypotheses, and vibes into verifiable improvements. Without this foundation, you're flying blind. With it, you can systematically identify weaknesses, measure progress, and build confidence that your system actually does what you think it does.

Here's what we're covering: deep research patterns as an agentic capability, why evaluation matters so much for AI systems specifically, the cold-start problem of test data creation, synthetic data generation with the RAGAS framework, knowledge graph approaches for comprehensive coverage, and metrics-driven development practices. Then we'll put it all together by building evaluation infrastructure for our learning assistant.

Let's get into it.

## Deep Research Patterns

Before we dive into evaluation, let's quickly cover deep research as an agentic capability. This connects to evaluation in an important way: when you're trying to define what "good" looks like for your system, you often need to research the domain, understand best practices, and synthesize information from multiple sources. Research is itself an agent pattern worth understanding.

The core pattern for deep research follows three steps: scope, research, and write. In the scoping phase, the agent clarifies what questions need to be answered, identifies the boundaries of the research, and creates a plan for gathering information. The research phase involves actually gathering information from multiple sources, often using subagents to parallelize the work. Finally, the writing phase synthesizes findings into a coherent output, whether that's a report, a summary, or structured data.

Here's what this looks like in practice:

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class ResearchPlan(BaseModel):
    main_question: str
    sub_questions: list[str]
    sources_to_check: list[str]
    expected_output_format: str

class ResearchFinding(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: list[str]

def scope_research(query: str, llm: ChatOpenAI) -> ResearchPlan:
    """Phase 1: Define what we're researching."""
    prompt = f"""Given this research question: {query}
    
Break it down into:
1. A clear main question
2. 3-5 specific sub-questions that would help answer the main question
3. Types of sources that would be valuable
4. What format the final output should take

Respond with JSON matching ResearchPlan schema."""
    
    response = llm.invoke(prompt)
    return ResearchPlan.model_validate_json(response.content)

async def research_questions(
    plan: ResearchPlan, 
    search_tool,
    llm: ChatOpenAI
) -> list[ResearchFinding]:
    """Phase 2: Gather information for each sub-question."""
    findings = []
    
    for question in plan.sub_questions:
        # Search for relevant information
        search_results = await search_tool.invoke(question)
        
        # Synthesize findings for this question
        synthesis_prompt = f"""Question: {question}

Search results:
{search_results}

Synthesize an answer to this question based on the search results.
Rate your confidence 0-1. List the sources used."""
        
        response = llm.invoke(synthesis_prompt)
        findings.append(parse_finding(response.content))
    
    return findings

def write_synthesis(
    plan: ResearchPlan, 
    findings: list[ResearchFinding],
    llm: ChatOpenAI
) -> str:
    """Phase 3: Synthesize findings into final output."""
    prompt = f"""Original question: {plan.main_question}

Research findings:
{format_findings(findings)}

Write a comprehensive answer to the original question, synthesizing 
all the findings. Note any contradictions or gaps. Format as: 
{plan.expected_output_format}"""
    
    return llm.invoke(prompt).content
```

This pattern matters for evaluation because defining "what good looks like" often requires research. What makes an explanation clear? What does pedagogically effective flashcard design look like? What retrieval quality do users actually need? These aren't questions you can answer from first principles. You need to investigate, synthesize, and codify your findings into evaluation criteria.

The connection to evaluation runs deeper than that, though. Deep research agents themselves need to be evaluated. How do you know if your research agent is finding the right information? How do you measure synthesis quality? The evaluation infrastructure we're building in this chapter applies to research agents just as much as any other agent type.

One pattern that works well for internal research capabilities is giving your tutor agent the ability to research unfamiliar topics before attempting to explain them. If a user asks about something outside the indexed materials, the tutor can scope what needs to be learned, research it through web search or other tools, and synthesize an explanation. This keeps the system helpful even when the knowledge base has gaps. The three-step pattern becomes: recognize the knowledge gap, research to fill it, then explain with confidence.

```python
def research_enhanced_response(
    query: str,
    knowledge_base,
    research_tool,
    llm: ChatOpenAI
) -> str:
    """Enhance responses with research when knowledge is insufficient."""
    
    # First, check what we know
    existing_context = knowledge_base.search(query, k=3)
    
    # Assess whether we can answer confidently
    assessment_prompt = f"""Given this question: {query}

And this available context:
{format_context(existing_context)}

Can you answer this question well with the available context?
Respond with: "SUFFICIENT" or "NEEDS_RESEARCH: [what to research]" """
    
    assessment = llm.invoke(assessment_prompt).content
    
    if assessment.startswith("SUFFICIENT"):
        return generate_response(query, existing_context, llm)
    else:
        # Extract research needs and gather more info
        research_query = assessment.replace("NEEDS_RESEARCH:", "").strip()
        additional_context = research_tool.invoke(research_query)
        
        combined_context = existing_context + additional_context
        return generate_response(query, combined_context, llm)
```

## Why Evaluation Matters

Evaluating AI systems is fundamentally different from evaluating traditional software. With traditional software, you write tests that check specific behaviors: "When I call `add(2, 2)`, I expect to get `4` back." The expected behavior is deterministic and well-defined. Either the function returns 4 or it doesn't.

AI systems don't work that way. Ask your RAG system "What is machine learning?" and there are thousands of valid responses. Some are better than others, but there's no single correct answer. The response depends on the model's training, the retrieved context, the specific prompt, and even random sampling in generation. You can run the same query twice and get meaningfully different responses.

This creates four interconnected challenges that make evaluation hard:

**Outputs are open-ended.** When you ask "explain quantum computing to a beginner," there's no reference answer you can check against. A response might be accurate but confusing. Another might be clear but oversimplified. A third might be technically precise but completely inappropriate for a beginner. You need evaluation criteria that capture what "good" means across multiple dimensions.

**Quality is subjective.** What's "good enough" depends entirely on context. A response that's perfect for a PhD student might be incomprehensible to a high schooler. A comprehensive 500-word explanation might be exactly right for one use case and way too long for another. Your evaluation needs to account for who's using the system and what they're trying to accomplish.

**Edge cases can be catastrophic.** A system that works great 95% of the time might generate dangerous misinformation the other 5%. Unlike traditional software where bugs often manifest as crashes or error codes, AI failures can be subtle, confidently wrong, and hard to detect. Evaluation needs to actively seek out these failure modes.

**Manual evaluation doesn't scale.** You can have humans review a hundred responses. Maybe a thousand if you've got budget and patience. But production systems generate millions of responses. You need automated evaluation that can scale without requiring human review of every output.

These challenges explain why vibe checks, while useful for early development, aren't sufficient as systems mature. A vibe check is exactly what it sounds like: you run some queries, read the outputs, and assess whether they seem good. This is fine when you're exploring, but it's subjective, not reproducible, and doesn't let you track changes over time.

Systematic evaluation replaces intuition with data. Instead of "I think it's getting better," you can say "faithfulness improved from 0.78 to 0.85 after the prompt change." Instead of "users seem happy," you can say "task completion rate is 87% with an average of 2.3 tool calls per successful completion." Numbers let you make decisions, track progress, and catch regressions before they hit production.

The goal isn't to eliminate human judgment. Humans are still the gold standard for evaluating quality, and you'll always need human review for calibration and edge cases. The goal is to automate the routine evaluation so humans can focus on the cases that actually need their attention.

## The Challenge of Creating Test Data

Here's the problem: systematic evaluation requires test data. Lots of it. You need examples that cover the breadth of what your system should handle, including easy cases, hard cases, edge cases, and adversarial cases. Where does this data come from?

The obvious answer is "collect it from users." And that's a great answer once you have users. Real user queries are invaluable because they reflect actual usage patterns, surface real edge cases, and help you understand what your system needs to handle. But there's a catch: you don't have users yet. Or you don't have enough users. Or your users haven't hit the edge cases you're worried about. This is the cold-start problem.

Manual test creation is the traditional fallback. You sit down and write test cases by hand: "If someone asks X, the system should respond with something like Y." This works for small test sets covering obvious scenarios. But it has serious limitations. Humans are bad at imagining edge cases. We tend to write tests for the happy path because that's what we naturally think of. We also underestimate the diversity of real queries. A test set of 50 hand-written questions might cover 10% of the actual query space.

There's also a coverage problem. Manual creation produces test sets that reflect the creator's mental model of the system. If you wrote the prompts, you'll test the cases you thought about when writing those prompts. The cases you didn't think about—the ones most likely to cause problems—won't be in your test set.

Production data helps once you have it, but comes with its own challenges. Privacy is the obvious one: real user queries often contain sensitive information that you can't use in test sets without careful anonymization. There's also the distribution problem: production data tells you what users actually ask, but it may not cover what they should be able to ask. If users have learned to work around your system's limitations, their queries won't surface those limitations.

What you need is a way to generate test data at scale, covering diverse scenarios, including edge cases and failure modes, without requiring either manual creation or production traffic. That's where synthetic data comes in.

## Introduction to Synthetic Data

Synthetic data is generated data that mimics the characteristics of real data without being collected from real users. For evaluation purposes, synthetic test data consists of questions, expected answers, and other test artifacts generated by models rather than humans.

The core idea is simple: use an LLM to generate the test cases that will be used to evaluate your system. This might sound circular—using AI to evaluate AI—but it works for several reasons. The generation model and evaluation target can be different, so biases don't necessarily transfer. Generated data can be reviewed and filtered by humans, giving you scale with human oversight. And for many evaluation tasks, you're not asking the generator to be creative, you're asking it to produce variations on patterns you've defined.

Synthetic data offers several compelling advantages for evaluation. Scale is the obvious one: you can generate thousands of test cases in minutes rather than months. Coverage improves because you can explicitly generate cases for different scenarios, difficulty levels, and edge cases. There are no privacy concerns because the data doesn't come from real users. And you have control: you can generate exactly the distribution of test cases you want.

But synthetic data isn't a silver bullet. The fundamental limitation is that generated data might not match real usage patterns. If your generation prompts assume users ask polite, well-formed questions, your test set won't prepare you for the terse, typo-filled, ambiguous queries real users submit. There's also quality variance: some generated test cases are excellent, others are nonsensical, and you need filtering to separate them.

The key is treating synthetic data as a complement to other data sources, not a replacement. Use synthetic data to achieve coverage of scenarios you've identified. Use real data to discover scenarios you haven't thought of. Use human-created data for the most critical cases. The combination gives you breadth, realism, and precision where it matters most.

Quality validation is essential for synthetic data. You can't just generate a thousand test cases and assume they're all good. Strategies include having humans review a sample to estimate overall quality, using a separate model to filter out low-quality examples, checking for duplicates and near-duplicates that inflate metrics, and validating that the distribution matches your expectations.

Here's a simple example of synthetic test generation:

```python
from langchain_openai import ChatOpenAI
import json

def generate_test_cases(
    topic: str, 
    num_cases: int,
    difficulty_distribution: dict,
    llm: ChatOpenAI
) -> list[dict]:
    """Generate synthetic test cases for a topic."""
    
    test_cases = []
    
    for difficulty, count in difficulty_distribution.items():
        prompt = f"""Generate {count} question-answer pairs about {topic}.

Difficulty level: {difficulty}
- easy: Basic factual questions with straightforward answers
- medium: Questions requiring explanation or connection of concepts  
- hard: Questions requiring synthesis, edge cases, or subtle distinctions

For each pair, provide:
- question: The question to ask
- expected_answer: A reference answer (doesn't need to be exact)
- key_concepts: List of concepts the answer should mention
- difficulty: {difficulty}

Format as JSON array."""

        response = llm.invoke(prompt)
        cases = json.loads(response.content)
        test_cases.extend(cases)
    
    return test_cases

# Generate a diverse test set
test_data = generate_test_cases(
    topic="retrieval-augmented generation",
    num_cases=30,
    difficulty_distribution={"easy": 10, "medium": 15, "hard": 5},
    llm=ChatOpenAI(model="gpt-4o")
)
```

This is a basic approach. For production evaluation, you want something more sophisticated, which brings us to RAGAS.

## The RAGAS Framework

RAGAS (RAG Assessment) is a framework specifically designed for evaluating retrieval-augmented generation systems. It provides both evaluation metrics and test data generation capabilities, making it a natural fit for the evaluation infrastructure we're building.

The framework addresses a fundamental question: how do you measure whether a RAG system is working well? RAGAS breaks this down into component parts, measuring retrieval quality and generation quality separately so you can identify exactly where problems occur.

### The Four Core Metrics

RAGAS defines four primary metrics that together give you a comprehensive view of RAG performance.

**Faithfulness** measures whether the generated answer sticks to what's actually in the retrieved context. A faithful answer only makes claims that are supported by the source material. An unfaithful answer hallucinates: it invents facts, makes unsupported claims, or goes beyond what the context justifies. Faithfulness is calculated as the proportion of claims in the answer that can be traced back to the retrieved context. If your system generates an answer with 10 distinct claims and only 7 can be found in the context, faithfulness is 0.7.

**Answer Relevance** measures whether the response actually addresses what was asked. A relevant answer is on-topic and responsive to the question. An irrelevant answer might be factually correct but completely beside the point. This metric uses semantic similarity between the question and answer to assess whether the response is actually addressing the query.

**Context Precision** measures retrieval quality: of the documents retrieved, what proportion were actually relevant? If you retrieve 5 chunks and only 2 are useful for answering the question, precision is 0.4. RAGAS weights this by rank position, so relevant documents appearing earlier count more. High precision means your retrieval isn't polluting the context with irrelevant information.

**Context Recall** measures whether the retrieved documents contain the information needed to answer correctly. This requires a ground truth answer for comparison—RAGAS checks what proportion of the expected answer is actually supported by the retrieved context. Low recall means you're missing information that should be retrieved.

These four metrics map to different failure modes. Low faithfulness indicates a generation problem: the model is hallucinating beyond the context. Low relevance indicates a prompt or instruction problem: the model isn't following the query. Low precision indicates a retrieval problem: you're pulling in junk. Low recall indicates a coverage problem: you're not finding what you need.

### Basic RAGAS Evaluation

Here's how to run a basic evaluation with RAGAS:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare your evaluation data
# This comes from running your RAG system on test questions
eval_data = {
    "question": [
        "What is the ReAct pattern?",
        "How do embeddings capture semantic meaning?",
        "When should you use multi-agent systems?"
    ],
    "answer": [
        "ReAct combines reasoning and acting in a loop...",
        "Embeddings map text to vectors where similar meanings...",
        "Multi-agent systems work best when you have distinct..."
    ],
    "contexts": [
        ["The ReAct pattern alternates between reasoning steps..."],
        ["Embeddings are dense vector representations...", "Semantic similarity..."],
        ["Multi-agent architectures provide benefits when..."]
    ],
    "ground_truth": [
        "ReAct (Reasoning + Acting) is a pattern where...",
        "Embeddings capture meaning by mapping text...",
        "Use multi-agent systems when tasks decompose naturally..."
    ]
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(f"Faithfulness: {results['faithfulness']:.3f}")
print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
print(f"Context Precision: {results['context_precision']:.3f}")
print(f"Context Recall: {results['context_recall']:.3f}")
```

Note that faithfulness and answer relevancy don't require ground truth because they can be computed from just the question, answer, and context. Context recall does require ground truth since it's measuring whether the retrieved context contains the information needed for a correct answer.

### Interpreting Results

Score interpretation depends on your use case, but here are general guidelines:

| Metric            | Good  | Acceptable | Needs Work |
| ----------------- | ----- | ---------- | ---------- |
| Faithfulness      | > 0.9 | 0.7-0.9    | < 0.7      |
| Answer Relevancy  | > 0.8 | 0.6-0.8    | < 0.6      |
| Context Precision | > 0.8 | 0.6-0.8    | < 0.6      |
| Context Recall    | > 0.8 | 0.6-0.8    | < 0.6      |

When diagnosing issues, low faithfulness usually means your model is hallucinating. Common fixes include strengthening grounding instructions in your prompts, lowering temperature, or using a more instruction-following model. Low answer relevancy suggests the model isn't understanding or addressing the question, so check your prompt structure and consider adding examples. Low context precision means retrieval is pulling irrelevant documents, pointing to issues with embeddings, chunking, or the need for reranking. Low context recall indicates missing information, suggesting you might need more documents, better search, or query expansion.

## The Knowledge Graph Approach

RAGAS's testset generation uses a knowledge graph approach to create comprehensive, diverse test cases. Understanding this approach helps you generate better evaluation data and adapt the technique for your own needs.

The intuition behind knowledge graphs for test generation is that documents contain entities (concepts, people, organizations, etc.) and relationships between those entities. By extracting this structure, you can systematically generate questions that cover the full scope of information in your documents.

### Building Knowledge Graphs from Documents

The process starts with entity extraction. An LLM reads through your documents and identifies the key entities mentioned:

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    entity_type: str  # concept, person, organization, etc.
    description: str
    
class Relationship(BaseModel):
    source: str  # entity name
    target: str  # entity name
    relationship: str  # how they're related

def extract_entities(document: str, llm: ChatOpenAI) -> list[Entity]:
    """Extract key entities from a document."""
    
    prompt = f"""Analyze this document and extract the key entities.

Document:
{document}

For each entity, provide:
- name: The entity's name
- entity_type: One of [concept, person, organization, technology, process]
- description: Brief description based on the document

Focus on entities that are central to the document's content.
Return as JSON array."""

    response = llm.invoke(prompt)
    return [Entity.model_validate(e) for e in json.loads(response.content)]

def extract_relationships(
    document: str, 
    entities: list[Entity],
    llm: ChatOpenAI
) -> list[Relationship]:
    """Extract relationships between entities."""
    
    entity_names = [e.name for e in entities]
    
    prompt = f"""Given these entities from a document:
{entity_names}

And the original document:
{document}

Identify relationships between entities. For each relationship:
- source: The first entity
- target: The second entity  
- relationship: How they're related (e.g., "uses", "implements", "is part of")

Return as JSON array."""

    response = llm.invoke(prompt)
    return [Relationship.model_validate(r) for r in json.loads(response.content)]
```

### Generating Questions from Graph Structure

Once you have a knowledge graph, you can generate questions systematically. Different graph patterns suggest different question types:

```python
def generate_questions_from_graph(
    entities: list[Entity],
    relationships: list[Relationship],
    llm: ChatOpenAI
) -> list[dict]:
    """Generate diverse questions based on knowledge graph structure."""
    
    questions = []
    
    # Simple entity questions (what is X?)
    for entity in entities:
        if entity.entity_type == "concept":
            q = generate_definition_question(entity, llm)
            questions.append({"type": "simple", **q})
    
    # Relationship questions (how does X relate to Y?)
    for rel in relationships:
        q = generate_relationship_question(rel, llm)
        questions.append({"type": "reasoning", **q})
    
    # Multi-hop questions (requires connecting multiple relationships)
    multi_hop_paths = find_multi_hop_paths(relationships)
    for path in multi_hop_paths[:10]:  # Limit complex questions
        q = generate_multi_hop_question(path, llm)
        questions.append({"type": "multi_context", **q})
    
    return questions

def generate_definition_question(entity: Entity, llm: ChatOpenAI) -> dict:
    """Generate a question about what something is."""
    
    prompt = f"""Create a question asking about this concept:
Name: {entity.name}
Description: {entity.description}

The question should ask what this concept is or how it works.
Also provide the expected answer based on the description.

Format: {{"question": "...", "answer": "..."}}"""
    
    return json.loads(llm.invoke(prompt).content)

def generate_relationship_question(rel: Relationship, llm: ChatOpenAI) -> dict:
    """Generate a question about how two things relate."""
    
    prompt = f"""Create a question about this relationship:
{rel.source} --[{rel.relationship}]--> {rel.target}

The question should ask about how these two concepts connect.
Provide the expected answer.

Format: {{"question": "...", "answer": "..."}}"""
    
    return json.loads(llm.invoke(prompt).content)
```

This approach ensures coverage because the questions are derived from the actual content of your documents. Every entity and relationship has the potential to generate test cases. You won't accidentally miss important concepts because they weren't on your mind when creating test data manually.

### RAGAS Testset Generation

RAGAS packages this approach into a convenient API. Note that the RAGAS API has evolved significantly—here's how to use the current version (0.4.x):

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Set up the generator with wrapped LLMs
# Use gpt-4o-mini to avoid rate limits - RAGAS makes many LLM calls during
# knowledge graph extraction, and gpt-4o's 30k TPM limit is too restrictive
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)

# Generate from your documents
testset = generator.generate_with_langchain_docs(
    documents=your_document_chunks,
    testset_size=100,  # Note: parameter name changed from test_size
)

# Convert to dataframe for inspection
test_df = testset.to_pandas()
print(test_df.head())

# RAGAS 0.4.x returns columns: user_input, reference, synthesizer_name
# You may want to rename for consistency:
test_df = test_df.rename(columns={
    "user_input": "question",
    "reference": "ground_truth"
})
```

RAGAS automatically generates a mix of question types based on the knowledge graph it extracts from your documents. The `synthesizer_name` column tells you what type of question was generated (single-hop, multi-hop, etc.).

## Generating Test Cases

Beyond the knowledge graph approach, there are additional patterns for generating effective test cases. The goal is always the same: create a diverse set of examples that thoroughly exercise your system.

### Difficulty-Based Generation

Different difficulty levels stress different parts of your system:

```python
def generate_difficulty_spectrum(
    topic: str,
    documents: list[str],
    llm: ChatOpenAI
) -> dict[str, list]:
    """Generate questions across difficulty levels."""
    
    test_cases = {"easy": [], "medium": [], "hard": []}
    
    # Easy: Direct lookup questions
    easy_prompt = f"""Based on these documents about {topic}, generate 10 questions 
where the answer appears almost verbatim in the text. These should be 
straightforward retrieval tasks.

Documents:
{documents[:2000]}

Format: [{{"question": "...", "answer": "...", "source_quote": "..."}}]"""
    
    test_cases["easy"] = json.loads(llm.invoke(easy_prompt).content)
    
    # Medium: Requires understanding and paraphrasing
    medium_prompt = f"""Generate 10 questions about {topic} where answering requires:
- Understanding concepts, not just finding text
- Paraphrasing information from the documents
- Connecting two related ideas

The answer shouldn't be copy-pasteable from the documents.

Documents:
{documents[:2000]}

Format: [{{"question": "...", "answer": "...", "reasoning": "..."}}]"""
    
    test_cases["medium"] = json.loads(llm.invoke(medium_prompt).content)
    
    # Hard: Edge cases, exceptions, nuanced understanding
    hard_prompt = f"""Generate 10 challenging questions about {topic}:
- Questions about edge cases or exceptions to general rules
- Questions requiring synthesis across multiple concepts
- Questions where naive answers would be incomplete or wrong
- Questions testing nuanced understanding

Documents:
{documents[:2000]}

Format: [{{"question": "...", "answer": "...", "why_hard": "..."}}]"""
    
    test_cases["hard"] = json.loads(llm.invoke(hard_prompt).content)
    
    return test_cases
```

### Adversarial Test Cases

Your system will encounter queries designed to trip it up, whether intentionally adversarial or just unusual. Generate test cases that stress failure modes:

```python
def generate_adversarial_cases(
    topic: str,
    documents: list[str],
    llm: ChatOpenAI
) -> list[dict]:
    """Generate test cases designed to find weaknesses."""
    
    prompt = f"""Generate adversarial test cases for a Q&A system about {topic}.

Include:

1. UNANSWERABLE questions (2-3 examples)
   Questions that seem related but can't actually be answered from the documents.
   The system should recognize it doesn't have the information.

2. AMBIGUOUS questions (2-3 examples)
   Questions that could be interpreted multiple ways.
   The system should ask for clarification or acknowledge ambiguity.

3. MISLEADING questions (2-3 examples)
   Questions with false premises or incorrect assumptions.
   The system should correct the assumption, not just answer.

4. OUT-OF-SCOPE questions (2-3 examples)
   Questions related to {topic} but outside what the documents cover.
   The system should acknowledge limitations.

Documents summary: {documents[:1000]}

For each case, specify:
- question: The adversarial question
- category: Which type (unanswerable, ambiguous, misleading, out_of_scope)
- expected_behavior: How a good system should respond
- failure_mode: What a bad response would look like"""
    
    return json.loads(llm.invoke(prompt).content)
```

Adversarial cases are crucial because they reveal how your system handles the unexpected. A system that scores 95% on straightforward questions might completely fail on edge cases. Better to discover this in evaluation than in production.

### Coverage Verification

After generating test cases, verify that you have adequate coverage:

```python
def analyze_coverage(
    test_cases: list[dict],
    expected_topics: list[str],
    expected_types: list[str]
) -> dict:
    """Analyze whether test cases cover expected dimensions."""
    
    # Track what's covered
    topic_coverage = {topic: 0 for topic in expected_topics}
    type_coverage = {t: 0 for t in expected_types}
    
    for case in test_cases:
        # Check topic coverage
        for topic in expected_topics:
            if topic.lower() in case.get("question", "").lower():
                topic_coverage[topic] += 1
        
        # Check type coverage
        case_type = case.get("type", "unknown")
        if case_type in type_coverage:
            type_coverage[case_type] += 1
    
    # Identify gaps
    gaps = {
        "missing_topics": [t for t, c in topic_coverage.items() if c == 0],
        "underrepresented_topics": [t for t, c in topic_coverage.items() if 0 < c < 3],
        "missing_types": [t for t, c in type_coverage.items() if c == 0]
    }
    
    return {
        "topic_coverage": topic_coverage,
        "type_coverage": type_coverage,
        "gaps": gaps,
        "total_cases": len(test_cases)
    }
```

If coverage analysis reveals gaps, generate additional cases specifically targeting those areas.

## Data Quality and Coverage

Generating test data is only half the battle. You also need to ensure the generated data is high quality and appropriately distributed.

### Validating Generated Data

Not every generated test case is good. Some common problems include questions that don't make sense, questions that are actually unanswerable despite claiming to be answerable, expected answers that are incorrect, near-duplicate questions that inflate test size without adding coverage, and questions that are too vague or too specific to be useful.

Build a validation pipeline:

```python
def validate_test_case(case: dict, documents: list[str], llm: ChatOpenAI) -> dict:
    """Validate a generated test case."""
    
    validation_prompt = f"""Evaluate this test case for quality:

Question: {case['question']}
Expected Answer: {case['answer']}

Reference Documents:
{documents[:2000]}

Evaluate:
1. Is the question clear and unambiguous? (1-5)
2. Is the expected answer actually correct based on the documents? (1-5)
3. Can the question be reasonably answered from the documents? (1-5)
4. Is this a useful test case (not trivial, not impossible)? (1-5)

Provide scores and explain any issues found.
Format: {{"scores": {{}}, "issues": [], "recommendation": "keep|revise|discard"}}"""
    
    validation = json.loads(llm.invoke(validation_prompt).content)
    
    return {
        **case,
        "validation": validation,
        "quality_score": sum(validation["scores"].values()) / 4
    }

def filter_test_cases(
    cases: list[dict], 
    min_quality: float = 3.5
) -> list[dict]:
    """Keep only high-quality test cases."""
    return [c for c in cases if c.get("quality_score", 0) >= min_quality]
```

### Distribution Matching

Your test data distribution should roughly match expected usage patterns. If 70% of real queries are simple factual questions, having 70% hard reasoning questions in your test set will give you a misleading picture of performance.

When you have production data, use it to calibrate:

```python
def analyze_query_distribution(production_queries: list[str], llm: ChatOpenAI) -> dict:
    """Analyze the distribution of query types in production data."""
    
    categories = {}
    
    for query in production_queries[:500]:  # Sample
        prompt = f"""Categorize this query:
"{query}"

Categories:
- simple_factual: Direct fact lookup
- explanation: Asking for explanation of concept
- comparison: Comparing two things
- how_to: Asking for instructions
- troubleshooting: Asking about problems/errors
- opinion: Asking for recommendation/opinion
- other: Doesn't fit above

Respond with just the category name."""
        
        category = llm.invoke(prompt).content.strip()
        categories[category] = categories.get(category, 0) + 1
    
    total = sum(categories.values())
    return {k: v/total for k, v in categories.items()}

def generate_matching_distribution(
    target_distribution: dict,
    total_cases: int,
    documents: list[str],
    llm: ChatOpenAI
) -> list[dict]:
    """Generate test cases matching a target distribution."""
    
    all_cases = []
    
    for category, proportion in target_distribution.items():
        count = int(total_cases * proportion)
        cases = generate_cases_for_category(category, count, documents, llm)
        all_cases.extend(cases)
    
    return all_cases
```

## Metrics-Driven Development

Evaluation isn't about measuring quality once and you're done. It's really about using measurements to drive systematic improvement. This is the philosophy of metrics-driven development: establish baselines, make changes, measure impact, iterate.

### The Improvement Loop

The basic loop is straightforward in concept:

1. **Measure**: Run your evaluation suite and record scores
2. **Analyze**: Identify what's causing low scores
3. **Hypothesize**: Form a theory about what would improve things
4. **Change**: Implement your proposed improvement
5. **Measure again**: See if it actually helped
6. **Iterate**: Keep going until you hit your targets

The key insight is that measurement comes both before and after the change. You don't make changes and just cross your fingers that they'll help. You make changes and then measure to *know* whether they did.

This sounds obvious, but it's surprisingly rare in practice. Most teams make changes based on intuition, ship them, and move on. They might notice if things get dramatically worse, but subtle regressions slip through. Slow accumulation of small regressions is how systems degrade over time.

```python
class EvaluationTracker:
    """Track evaluation results over time."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.runs = []
    
    def record_run(
        self, 
        version: str, 
        metrics: dict, 
        config: dict,
        notes: str = ""
    ):
        """Record an evaluation run."""
        self.runs.append({
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "metrics": metrics,
            "config": config,
            "notes": notes
        })
    
    def compare_versions(self, v1: str, v2: str) -> dict:
        """Compare metrics between two versions."""
        run1 = next(r for r in self.runs if r["version"] == v1)
        run2 = next(r for r in self.runs if r["version"] == v2)
        
        comparison = {}
        for metric in run1["metrics"]:
            old = run1["metrics"][metric]
            new = run2["metrics"][metric]
            comparison[metric] = {
                "old": old,
                "new": new,
                "change": new - old,
                "percent_change": (new - old) / old * 100 if old != 0 else float('inf')
            }
        
        return comparison
    
    def get_trend(self, metric: str) -> list:
        """Get trend of a metric over time."""
        return [
            {"version": r["version"], "value": r["metrics"].get(metric)}
            for r in sorted(self.runs, key=lambda x: x["timestamp"])
        ]
    
    def check_regression(self, metric: str, threshold: float = 0.05) -> bool:
        """Check if recent changes caused regression beyond threshold."""
        if len(self.runs) < 2:
            return False
        
        recent = self.runs[-1]["metrics"].get(metric, 0)
        previous = self.runs[-2]["metrics"].get(metric, 0)
        
        return (previous - recent) / previous > threshold if previous > 0 else False
```

### Setting Quality Thresholds

Before you can improve, you need to define what "good enough" looks like. This requires thinking about your specific use case. For a medical information system, faithfulness needs to be near-perfect because hallucinated medical advice is very bad (and possibly very dangerous!). For a creative writing assistant, some "hallucination" (which, in this context, could be just another name for creativity) might be desirable.

Set thresholds based on user impact. Ask yourself: at what quality level would users be satisfied? At what level would they stop using the system? At what level might there be actual harm? These thresholds become your targets.

```python
class QualityThresholds:
    """Define quality thresholds for the system."""
    
    # Minimum acceptable scores (system is broken below this)
    MINIMUM = {
        "faithfulness": 0.7,
        "relevance": 0.6,
        "retrieval_precision": 0.5
    }
    
    # Target scores (what we're aiming for)
    TARGET = {
        "faithfulness": 0.9,
        "relevance": 0.85,
        "retrieval_precision": 0.8
    }
    
    # Stretch goals (excellence)
    STRETCH = {
        "faithfulness": 0.95,
        "relevance": 0.92,
        "retrieval_precision": 0.9
    }
    
    @classmethod
    def assess(cls, metrics: dict) -> str:
        """Assess current metrics against thresholds."""
        
        # Check for any below minimum
        for metric, min_val in cls.MINIMUM.items():
            if metrics.get(metric, 0) < min_val:
                return f"CRITICAL: {metric} below minimum threshold"
        
        # Check if meeting targets
        meeting_targets = all(
            metrics.get(m, 0) >= t 
            for m, t in cls.TARGET.items()
        )
        
        if meeting_targets:
            return "GOOD: Meeting all targets"
        else:
            below_target = [
                m for m, t in cls.TARGET.items() 
                if metrics.get(m, 0) < t
            ]
            return f"IMPROVING: Below target on {', '.join(below_target)}"
```

### Avoiding Goodhart's Law

Goodhart's Law states: "When a measure becomes a target, it ceases to be a good measure." This is a real danger in metrics-driven development. If you optimize purely for faithfulness, you might get a system that only quotes verbatim from context and never synthesizes. If you optimize purely for relevance, you might get verbose answers that technically address the question but aren't actually helpful.

Protect against this by using multiple metrics together (improvement should be balanced across dimensions), maintaining human evaluation as a check (automated metrics can be gamed, humans catch gaming), periodically reviewing failure cases qualitatively (numbers tell you what, not why), and updating test sets as the system evolves (don't let the system overfit to a static test set).

## Loading Datasets into LangSmith

LangSmith provides infrastructure for managing evaluation datasets, running evaluations at scale, and tracking results over time. It's the natural place to store your synthetic test data and run systematic evaluations.

### Creating and Managing Datasets

```python
from langsmith import Client

client = Client()

# Create a new dataset
dataset = client.create_dataset(
    dataset_name="rag-evaluation-v1",
    description="Synthetic test cases for RAG system evaluation"
)

# Add examples to the dataset
examples = [
    {
        "inputs": {"question": "What is the ReAct pattern?"},
        "outputs": {"answer": "ReAct combines reasoning and acting..."}
    },
    {
        "inputs": {"question": "How do embeddings work?"},
        "outputs": {"answer": "Embeddings map text to vectors..."}
    }
]

client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id
)
```

### Organizing Datasets

For serious evaluation, you'll want multiple datasets:

- **Golden set**: High-quality, human-verified examples for critical testing.
- **Regression set**: Cases that previously failed, and are now fixed.You want to ensure they stay fixed.
- **Comprehensive set**: Large synthetic set for broad coverage.
- **Edge cases**: Adversarial and unusual cases.

```python
# Create datasets for different purposes
datasets = {
    "golden": client.create_dataset(
        "rag-golden-v1",
        description="Human-verified critical test cases"
    ),
    "regression": client.create_dataset(
        "rag-regression-v1", 
        description="Previously failing cases"
    ),
    "comprehensive": client.create_dataset(
        "rag-comprehensive-v1",
        description="Large synthetic test suite"
    ),
    "adversarial": client.create_dataset(
        "rag-adversarial-v1",
        description="Edge cases and adversarial examples"
    )
}
```

### Running Evaluations

LangSmith's evaluation API lets you run your system against datasets with custom evaluators:

```python
from langsmith.evaluation import evaluate

def my_rag_system(inputs: dict) -> dict:
    """The RAG system being evaluated."""
    question = inputs["question"]
    # Your RAG pipeline here
    result = rag_chain.invoke(question)
    return {"answer": result["answer"], "contexts": result["contexts"]}

def faithfulness_evaluator(run, example) -> dict:
    """Custom evaluator for faithfulness."""
    answer = run.outputs["answer"]
    contexts = run.outputs["contexts"]
    
    # Use RAGAS or custom logic
    score = calculate_faithfulness(answer, contexts)
    
    return {"key": "faithfulness", "score": score}

def relevance_evaluator(run, example) -> dict:
    """Custom evaluator for relevance."""
    question = example.inputs["question"]
    answer = run.outputs["answer"]
    
    score = calculate_relevance(question, answer)
    
    return {"key": "relevance", "score": score}

# Run evaluation
results = evaluate(
    my_rag_system,
    data="rag-comprehensive-v1",
    evaluators=[faithfulness_evaluator, relevance_evaluator],
    experiment_prefix="rag-v2.1"
)

# Results are stored in LangSmith for comparison
print(f"Faithfulness: {results['faithfulness']:.3f}")
print(f"Relevance: {results['relevance']:.3f}")
```

### Comparing Experiments

LangSmith tracks experiments over time, letting you compare versions:

```python
# After running multiple experiments, compare them
experiments = client.list_projects(project_ids=["rag-v2.0", "rag-v2.1", "rag-v2.2"])

for exp in experiments:
    runs = list(client.list_runs(project_name=exp.name))
    
    # Calculate aggregate metrics
    avg_faithfulness = sum(r.feedback_stats.get("faithfulness", {}).get("avg", 0) 
                          for r in runs) / len(runs)
    
    print(f"{exp.name}: faithfulness={avg_faithfulness:.3f}")
```

The LangSmith UI provides dashboards for visualizing these comparisons, making it easy to see trends and identify regressions.

## Building StudyBuddy v8

Time to apply everything we've learned. In the last chapter, we built a curriculum planner that breaks learning goals into structured study plans with automatically generated flashcards. That's powerful, but how do we know it's actually working well? How do we measure tutoring quality? How do we know our flashcards are pedagogically sound?

In this version, we're adding the evaluation infrastructure that lets us answer these questions systematically.

### Where We Left Off

StudyBuddy v7 introduced deep agent capabilities through the Curriculum Planner. Users can specify a learning goal like "Understand RAG systems well enough to build production applications," and the planner decomposes this into modules, sections, and prerequisite chains. It spawns subagents to generate flashcards for each section and tracks progress through checkpoints.

The system has two modes: tutoring (conversational explanations grounded in uploaded materials) and practice (spaced repetition flashcards). Memory connects both modes: struggle areas in practice inform tutoring, and concepts discussed in tutoring appear in practice.

What we're missing is any systematic way to evaluate whether any of this is working well. Do our explanations actually help learners understand concepts? Are the flashcards clear and useful? Is the retrieval finding the right content? We've been relying on vibe checks: running the system, reading outputs, making judgment calls. Time to level up.

### What We're Adding

StudyBuddy v8 introduces comprehensive evaluation infrastructure:

1. **Synthetic test data generation** using the RAGAS knowledge graph approach
2. **Evaluation datasets** covering tutoring quality, flashcard effectiveness, and retrieval accuracy
3. **LangSmith integration** for dataset management and experiment tracking
4. **Baseline metrics** to measure current performance and track improvements
5. **Custom evaluators** specific to educational content quality

### Setting Up RAGAS for Our Domain

First, let's configure RAGAS to generate test data from our reference materials. Note that we're using RAGAS 0.4.x which has a different API than earlier versions:

```python
# api/evaluation/testset_generator.py

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
    """Load and chunk reference documents for test generation."""

    documents_dir = Path(__file__).parent.parent.parent / docs_path
    guide_files = sorted(documents_dir.glob("ref-*.md"))

    # Use larger chunks for test generation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )

    all_chunks = []
    for filepath in guide_files:
        loader = TextLoader(str(filepath))
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

    return all_chunks

def create_testset_generator() -> TestsetGenerator:
    """Initialize RAGAS testset generator with wrapped LangChain models.

    Uses gpt-4o-mini to avoid rate limits - RAGAS makes many LLM calls during
    knowledge graph extraction, and gpt-4o's 30k TPM limit is too restrictive.
    """

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    return TestsetGenerator(llm=generator_llm, embedding_model=embeddings)

def generate_tutoring_testset(
    documents: list[Document],
    test_size: int = 100
) -> pd.DataFrame:
    """Generate test cases for tutoring evaluation."""

    generator = create_testset_generator()

    # RAGAS 0.4.x automatically generates diverse question types
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=test_size,  # Note: parameter name is testset_size
    )

    df = testset.to_pandas()

    # Rename columns for consistency with our evaluation system
    if "user_input" in df.columns:
        df = df.rename(columns={"user_input": "question"})
    if "reference" in df.columns:
        df = df.rename(columns={"reference": "ground_truth"})

    return df
```

Note that RAGAS 0.4.x requires additional packages: `rapidfuzz` for string distance calculations and `nest-asyncio` to work within FastAPI's event loop. The `nest_asyncio.apply()` call must happen before importing RAGAS to avoid "Event loop is closed" errors.

### Building Evaluation Datasets

We need different evaluation datasets for different purposes:

```python
# api/evaluation/dataset_builder.py

from langsmith import Client
import json

client = Client()

def create_tutoring_evaluation_dataset(testset_df, dataset_name: str):
    """Create LangSmith dataset for tutoring evaluation."""
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Synthetic test cases for tutoring quality evaluation"
    )
    
    for _, row in testset_df.iterrows():
        client.create_example(
            inputs={
                "question": row["question"],
                "topic": extract_topic(row["question"]),
                "difficulty": row.get("evolution_type", "simple")
            },
            outputs={
                "reference_answer": row["ground_truth"],
                "key_concepts": extract_key_concepts(row["ground_truth"])
            },
            dataset_id=dataset.id
        )
    
    return dataset

def create_flashcard_evaluation_dataset(topics: list[str]):
    """Create dataset for flashcard quality evaluation."""
    
    dataset = client.create_dataset(
        dataset_name="flashcard-quality-v1",
        description="Test cases for flashcard clarity and effectiveness"
    )
    
    # Generate flashcard evaluation cases
    llm = ChatOpenAI(model="gpt-4o")
    
    for topic in topics:
        # Generate examples of good and bad flashcards for each topic
        prompt = f"""For the topic "{topic}", generate examples for flashcard evaluation:

1. A GOOD flashcard (clear, focused, testable)
2. A BAD flashcard - too vague
3. A BAD flashcard - too complex (multiple concepts)
4. A BAD flashcard - unclear wording

For each, provide:
- question: The flashcard question
- answer: The expected answer
- quality: "good" or "bad"
- issue: If bad, what's wrong with it

Format as JSON array."""
        
        examples = json.loads(llm.invoke(prompt).content)
        
        for ex in examples:
            client.create_example(
                inputs={
                    "topic": topic,
                    "flashcard_question": ex["question"],
                    "flashcard_answer": ex["answer"]
                },
                outputs={
                    "expected_quality": ex["quality"],
                    "expected_issue": ex.get("issue", "none")
                },
                dataset_id=dataset.id
            )
    
    return dataset

def create_retrieval_evaluation_dataset(documents: list):
    """Create dataset for retrieval quality evaluation."""
    
    dataset = client.create_dataset(
        dataset_name="retrieval-quality-v1",
        description="Test cases for measuring retrieval accuracy"
    )
    
    llm = ChatOpenAI(model="gpt-4o")
    
    # For each document, generate questions it should answer
    for doc in documents:
        prompt = f"""Based on this content, generate 3 questions that should 
retrieve this document:

Content:
{doc.page_content[:1500]}

For each question:
- question: The query
- expected_chunks: Key phrases that should appear in retrieved content
- source_file: {doc.metadata.get('source', 'unknown')}

Format as JSON array."""
        
        questions = json.loads(llm.invoke(prompt).content)
        
        for q in questions:
            client.create_example(
                inputs={"query": q["question"]},
                outputs={
                    "expected_chunks": q["expected_chunks"],
                    "source_file": q["source_file"]
                },
                dataset_id=dataset.id
            )
    
    return dataset
```

### Custom Evaluators for Educational Content

Generic RAGAS metrics are a good start, but educational content has specific quality requirements. Let's build custom evaluators:

```python
# api/evaluation/evaluators.py

import json
import re
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks.

    LLMs often wrap JSON in markdown code blocks like ```json ... ```.
    This helper extracts the JSON regardless of formatting.
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        return json.loads(json_match.group(1).strip())

    # Try parsing the whole response as JSON
    return json.loads(text)

def tutoring_quality_evaluator(run, example) -> dict:
    """Evaluate tutoring response quality for learning effectiveness."""

    llm = ChatOpenAI(model="gpt-4o-mini")

    question = example.inputs.get("question", "")
    response = run.outputs.get("response", "")
    reference = example.outputs.get("reference_answer", "")

    prompt = f"""Evaluate this tutoring response for educational effectiveness.

Student Question: {question}

Tutor Response: {response}

Reference Answer: {reference}

Evaluate on these criteria (1-5 scale):

1. ACCURACY: Is the information factually correct?
2. CLARITY: Is the explanation clear and easy to understand?
3. COMPLETENESS: Does it cover the key concepts needed to answer?
4. PEDAGOGY: Does it teach effectively (examples, analogies, building blocks)?
5. ENGAGEMENT: Is it engaging and encouraging for a learner?

Provide scores and brief justification for each.
Format as JSON: {{"accuracy": N, "clarity": N, "completeness": N, "pedagogy": N,
"engagement": N, "overall": N, "justification": "..."}}

The overall score should be the average of the five criteria, rounded to nearest integer."""

    try:
        result = _extract_json(llm.invoke(prompt).content)
        overall = result.get("overall", 3)

        return {
            "key": "tutoring_quality",
            "score": overall / 5.0,  # Normalize to 0-1
            "comment": result.get("justification", "")
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Tutoring evaluator failed: {e}")
        return {
            "key": "tutoring_quality",
            "score": 0.5,  # Default middle score on failure
            "comment": f"Evaluation failed: {e}"
        }

def flashcard_quality_evaluator(run, example) -> dict:
    """Evaluate generated flashcard quality."""

    llm = ChatOpenAI(model="gpt-4o-mini")

    topic = example.inputs.get("topic", "")
    generated_q = run.outputs.get("question", "")
    generated_a = run.outputs.get("answer", "")

    prompt = f"""Evaluate this flashcard for learning effectiveness.

Topic: {topic}
Question: {generated_q}
Answer: {generated_a}

Evaluate:
1. FOCUS: Does it test exactly ONE concept? (not too broad or narrow)
2. CLARITY: Is the question unambiguous?
3. TESTABILITY: Can someone clearly know if they got it right?
4. ANSWER_QUALITY: Is the answer accurate and appropriately detailed?
5. LEARNING_VALUE: Will this help someone learn the topic?

Score each 1-5, then overall 1-5.
Identify any specific issues.

Format as JSON: {{"focus": N, "clarity": N, "testability": N, "answer_quality": N,
"learning_value": N, "overall": N, "issues": ["..."]}}"""

    try:
        result = _extract_json(llm.invoke(prompt).content)
        overall = result.get("overall", 3)
        issues = result.get("issues", [])

        return {
            "key": "flashcard_quality",
            "score": overall / 5.0,
            "comment": "; ".join(issues) if issues else "No issues found"
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Flashcard evaluator failed: {e}")
        return {
            "key": "flashcard_quality",
            "score": 0.5,
            "comment": f"Evaluation failed: {e}"
        }

def retrieval_precision_evaluator(run, example) -> dict:
    """Evaluate whether retrieval found the right content."""

    expected_chunks = example.outputs.get("expected_chunks", [])
    retrieved = run.outputs.get("contexts", [])

    if not expected_chunks:
        return {
            "key": "retrieval_precision",
            "score": 1.0,  # No expectations = automatic pass
            "comment": "No expected chunks to verify"
        }

    # Combine all retrieved contexts into one searchable string
    if isinstance(retrieved, list):
        retrieved_text = " ".join(str(c) for c in retrieved).lower()
    else:
        retrieved_text = str(retrieved).lower()

    # Check how many expected phrases appear in retrieved content
    found = sum(1 for chunk in expected_chunks if chunk.lower() in retrieved_text)
    precision = found / len(expected_chunks)

    return {
        "key": "retrieval_precision",
        "score": precision,
        "comment": f"Found {found}/{len(expected_chunks)} expected chunks"
    }
```

### Running Baseline Evaluation

Now let's establish our baseline metrics. The key insight here is that LangSmith's `evaluate()` function returns an `ExperimentResults` object that you iterate over. Each result contains `evaluation_results` with a list of `EvaluationResult` objects that have `.key` and `.score` attributes:

```python
# api/evaluation/run_baseline.py

import logging
from langsmith import Client
from langsmith.evaluation import evaluate
from .evaluators import (
    tutoring_quality_evaluator,
    flashcard_quality_evaluator,
    retrieval_precision_evaluator
)

logger = logging.getLogger(__name__)

def _get_langsmith_client() -> Client | None:
    """Get LangSmith client with error handling."""
    try:
        client = Client()
        list(client.list_datasets(limit=1))
        return client
    except Exception as e:
        logger.warning(f"LangSmith unavailable: {e}")
        return None

def run_tutoring_evaluation(
    tutor_func,
    dataset_name: str = "tutoring-evaluation-v1",
    experiment_prefix: str = "tutor-baseline"
) -> dict:
    """Run tutoring evaluation against LangSmith dataset."""

    client = _get_langsmith_client()
    if not client:
        return {"error": "LangSmith unavailable", "tutoring_quality": 0.0}

    try:
        results = evaluate(
            tutor_func,
            data=dataset_name,
            evaluators=[tutoring_quality_evaluator],
            experiment_prefix=experiment_prefix,
        )

        # Extract aggregate score from ExperimentResults
        # Each result is a dict with 'run', 'example', 'evaluation_results' keys
        # evaluation_results['results'] contains EvaluationResult objects
        scores = []
        result_list = list(results)  # Consume the iterator

        for r in result_list:
            eval_results = r.get("evaluation_results", {})
            results_list = eval_results.get("results", [])

            for eval_result in results_list:
                # EvaluationResult objects have .key and .score attributes
                key = getattr(eval_result, "key", None)
                score = getattr(eval_result, "score", None)
                if key == "tutoring_quality" and score is not None:
                    scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"Tutoring evaluation complete: {len(scores)} samples, avg={avg_score:.3f}")

        return {
            "tutoring_quality": avg_score,
            "sample_size": len(scores),
            "experiment": experiment_prefix
        }

    except Exception as e:
        logger.error(f"Tutoring evaluation failed: {e}")
        return {"error": str(e), "tutoring_quality": 0.0}
```

The same pattern applies to flashcard and retrieval evaluations. The important detail is consuming the `ExperimentResults` iterator with `list()` and then extracting scores using `getattr()` since `EvaluationResult` objects use attribute access, not dictionary access.

### Evaluation API Endpoint

Finally, let's add an API endpoint so we can trigger evaluations and view results:

```python
# In api/index.py

from .evaluation.run_baseline import (
    run_tutoring_evaluation,
    run_flashcard_evaluation,
    run_retrieval_evaluation
)

class EvaluationRequest(BaseModel):
    dataset_type: str  # "tutoring", "flashcard", "retrieval", "all"
    dataset_name: Optional[str] = None  # Name of LangSmith dataset to evaluate against
    experiment_name: Optional[str] = None

class EvaluationResult(BaseModel):
    dataset_type: str
    experiment_name: str
    metrics: dict
    sample_size: int
    timestamp: str

@app.post("/api/evaluation/run", response_model=EvaluationResult)
def run_evaluation(request: EvaluationRequest):
    """Run evaluation on specified dataset."""

    from datetime import datetime

    experiment_name = request.experiment_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M')}"

    # Define wrapper functions that connect to our system
    def tutor_system(inputs: dict) -> dict:
        context = search_materials(inputs.get("question", ""), k=4)
        response = tutor_explain(tutor_llm, inputs.get("question", ""), context, None, "")
        return {"response": response}

    def flashcard_system(inputs: dict) -> dict:
        context = search_materials(inputs.get("topic", ""), k=4)
        card = generate_single_card(card_generator_llm, inputs.get("topic", ""), [], context)
        return {
            "question": card.get("question", "") if card else "",
            "answer": card.get("answer", "") if card else ""
        }

    def retrieval_system(inputs: dict) -> dict:
        context = search_materials(inputs.get("query", ""), k=5)
        chunks = context.split("\n\n---\n\n") if context else []
        return {"contexts": chunks}

    metrics = {}
    sample_size = 0

    if request.dataset_type in ["tutoring", "all"]:
        dataset = request.dataset_name or "tutoring-evaluation-v1"
        result = run_tutoring_evaluation(
            tutor_system,
            dataset_name=dataset,
            experiment_prefix=f"tutor-{experiment_name}"
        )
        metrics["tutoring_quality"] = result.get("tutoring_quality", 0)
        sample_size += result.get("sample_size", 0)

    # Similar patterns for flashcard and retrieval...

    return EvaluationResult(
        dataset_type=request.dataset_type,
        experiment_name=experiment_name,
        metrics=metrics,
        sample_size=sample_size,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/evaluation/generate-testset")
def generate_test_data(test_size: int = 100):
    """Generate synthetic test data from reference materials."""

    from .evaluation.testset_generator import (
        load_reference_documents,
        generate_tutoring_testset,
        validate_testset
    )
    from .evaluation.dataset_builder import create_tutoring_evaluation_dataset
    from datetime import datetime

    # Load and chunk reference documents
    documents = load_reference_documents("documents")
    if not documents:
        raise HTTPException(status_code=404, detail="No reference documents found")

    # Generate synthetic test cases using RAGAS
    testset = generate_tutoring_testset(documents, test_size)
    testset = validate_testset(testset)  # Filter low-quality cases

    # Create LangSmith dataset with timestamp to avoid conflicts
    dataset_name = f"tutoring-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    result = create_tutoring_evaluation_dataset(testset, dataset_name)

    if not result:
        raise HTTPException(status_code=500, detail="Failed to create dataset")

    return {
        "status": "success",
        "test_cases_generated": len(testset),
        "dataset_id": str(result["id"]),  # Convert UUID to string
        "dataset_name": result["name"]
    }
```

### Evaluation Dashboard

To make evaluation results actionable, we need a way to visualize them. Here's an endpoint that provides dashboard data:

```python
# api/evaluation/dashboard.py

from langsmith import Client
from datetime import datetime, timedelta
from collections import defaultdict

client = Client()

class DashboardMetrics(BaseModel):
    current_scores: dict[str, float]
    score_trends: dict[str, list[dict]]
    weak_areas: list[dict]
    recent_experiments: list[dict]
    recommendations: list[str]

@app.get("/api/evaluation/dashboard", response_model=DashboardMetrics)
def get_evaluation_dashboard():
    """Get evaluation dashboard data."""
    
    # Fetch recent experiment results
    experiments = list(client.list_projects(
        reference_dataset_name="tutoring-evaluation-v1"
    ))[:10]
    
    # Calculate current scores (most recent experiment)
    latest = experiments[0] if experiments else None
    current_scores = get_experiment_scores(latest) if latest else {}
    
    # Build score trends over time
    score_trends = build_score_trends(experiments)
    
    # Identify weak areas by analyzing failure cases
    weak_areas = identify_weak_areas(latest) if latest else []
    
    # Generate recommendations based on metrics
    recommendations = generate_recommendations(current_scores, weak_areas)
    
    return DashboardMetrics(
        current_scores=current_scores,
        score_trends=score_trends,
        weak_areas=weak_areas,
        recent_experiments=[
            {"name": e.name, "date": e.created_at.isoformat()}
            for e in experiments[:5]
        ],
        recommendations=recommendations
    )

def identify_weak_areas(experiment) -> list[dict]:
    """Identify content areas with low scores."""
    
    weak_areas = []
    
    runs = client.list_runs(project_name=experiment.name)
    
    # Group scores by topic
    topic_scores = defaultdict(list)
    for run in runs:
        topic = run.inputs.get("topic", "unknown")
        score = run.feedback_stats.get("tutoring_quality", {}).get("avg", 0)
        topic_scores[topic].append(score)
    
    # Find topics with consistently low scores
    for topic, scores in topic_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        if avg < 0.7:  # Below threshold
            weak_areas.append({
                "topic": topic,
                "average_score": avg,
                "sample_count": len(scores),
                "suggestion": f"Review and improve content coverage for {topic}"
            })
    
    return sorted(weak_areas, key=lambda x: x["average_score"])

def generate_recommendations(scores: dict, weak_areas: list) -> list[str]:
    """Generate actionable recommendations from evaluation data."""
    
    recommendations = []
    
    # Check each metric against thresholds
    if scores.get("faithfulness", 1) < 0.85:
        recommendations.append(
            "Faithfulness is below target. Consider strengthening grounding "
            "instructions in your prompts or reducing temperature."
        )
    
    if scores.get("retrieval_precision", 1) < 0.75:
        recommendations.append(
            "Retrieval precision needs improvement. Chapter 9 covers reranking "
            "and hybrid search techniques that can help."
        )
    
    if scores.get("tutoring_quality", 1) < 0.8:
        recommendations.append(
            "Tutoring quality has room for improvement. Review the weak areas "
            "identified below and consider adding more examples to prompts."
        )
    
    # Add recommendations based on weak areas
    if weak_areas:
        topics = [w["topic"] for w in weak_areas[:3]]
        recommendations.append(
            f"Content quality is low for: {', '.join(topics)}. "
            "Consider adding more reference materials for these topics."
        )
    
    if not recommendations:
        recommendations.append(
            "All metrics are meeting targets. Consider running adversarial "
            "test cases to identify edge case failures."
        )
    
    return recommendations
```

This dashboard gives you visibility into system health at a glance. You can see current scores, track trends over time, identify specific weak spots, and get actionable recommendations for improvement. It transforms evaluation from a one-time activity into an ongoing practice.

### Testing the Evaluation System

Let's verify everything works:

```python
# tests/test_evaluation.py

def test_testset_generation():
    """Verify we can generate synthetic test data."""

    from api.evaluation.testset_generator import (
        load_reference_documents,
        generate_tutoring_testset
    )

    docs = load_reference_documents("documents")
    assert len(docs) > 0

    # Generate small testset for speed
    testset = generate_tutoring_testset(docs, test_size=5)
    assert len(testset) == 5
    assert "question" in testset.columns
    assert "ground_truth" in testset.columns

def test_tutoring_evaluator():
    """Verify tutoring evaluator produces valid scores."""

    from api.evaluation.evaluators import tutoring_quality_evaluator

    # Mock run and example that mimic LangSmith structures
    class MockRun:
        outputs = {"response": "Machine learning is a way for computers to learn from data..."}

    class MockExample:
        inputs = {"question": "What is machine learning?"}
        outputs = {"reference_answer": "Machine learning is a subset of AI..."}

    result = tutoring_quality_evaluator(MockRun(), MockExample())

    assert "score" in result
    assert 0 <= result["score"] <= 1
    assert result["key"] == "tutoring_quality"

def test_baseline_evaluation_runs():
    """Verify baseline evaluation completes without errors."""

    from api.evaluation.run_baseline import run_baseline_evaluation

    # Define mock functions for testing
    def mock_tutor(inputs):
        return {"response": "Mock tutoring response."}

    def mock_flashcard(inputs):
        return {"question": "What is X?", "answer": "X is Y."}

    def mock_retrieval(inputs):
        return {"contexts": ["Mock context"]}

    # This is an integration test - requires LangSmith setup
    results = run_baseline_evaluation(mock_tutor, mock_flashcard, mock_retrieval)

    assert "tutoring" in results
    assert "flashcards" in results
    assert "retrieval" in results
```

### Running locally

StudyBuddy v8 runs with two terminals—one for the backend, one for the frontend.

**Terminal 1 - Backend:**
```bash
cd v8-evaluation
uv run uvicorn api.index:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd v8-evaluation/frontend
npm run dev
```

Visit http://localhost:3000. The Next.js dev server proxies `/api/*` requests to the FastAPI backend on port 8000 (configured in `next.config.ts`).

### Using the Evaluation Infrastructure

With all the components in place, here's how to use the evaluation system in practice.

**Prerequisites**: First, install the new dependencies and ensure your environment variables are set:

```bash
cd v8-evaluation
uv sync
```

Your `.env` file needs:

```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=lsv2_...
```

**Step 1: Generate Synthetic Test Data**

The first step is generating test cases from your reference documents:

```bash
curl -X POST http://localhost:8000/api/evaluation/generate-testset \
  -H "Content-Type: application/json" \
  -d '{"test_size": 5}'
```

This creates a LangSmith dataset with synthetic questions generated from your reference documents using RAGAS. (Note that this will take a few minutes. You can switch to your server terminal to watch the progress of the operation.) The response tells you how many test cases were created (RAGAS might generate 6 instead of 5, which is normal) and the dataset ID. Once that's done, you can check your LangSmith dashboard to see the `tutoring-eval-YYYYMMDD-HHMMSS` dataset that was created.

**Step 2: Run Evaluation**

Once you have test data, run evaluation against your system:

```bash
# Evaluate all components
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_type": "all", "dataset_name": "tutoring-eval-YYYYMMDD-HHMMSS"}'

# Or evaluate specific components
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_type": "tutoring", "dataset_name": "tutoring-eval-YYYYMMDD-HHMMSS"}'
```
Either way, replace `tutoring-eval-YYYYMMDD-HHMMSS` with the actual dataset name from the `generate-testset` response or your LangSmith dashboard.

These commands return metrics t1hat look like this:

```json
{
  "metrics": {
    "tutoring_quality": 0.82,
    "flashcard_quality": 0.78,
    "retrieval_precision": 0.85
  },
  "sample_size": 50
}
```

**Step 3: View Dashboard**

Get aggregated metrics, trends, and recommendations:

```bash
curl http://localhost:8000/api/evaluation/dashboard
```

The dashboard returns current scores, historical trends across experiments, topics with low scores (weak areas), and actionable recommendations for improvement.

**Viewing Results in LangSmith**

Results are also tracked in LangSmith at https://smith.langchain.com. Navigate to your project (studybuddy-v8), view experiments by prefix (like `tutor-baseline-*`), and compare runs to see how changes affect quality over time.

**Important Notes**

The evaluation endpoints are disabled on Vercel because they're too compute-heavy for serverless environments. Run evaluations locally during development. Generate testsets once, then reuse them for multiple evaluation runs to save on API costs.

### What's Next

We now have the infrastructure to systematically measure and improve our learning assistant. The baseline metrics tell us where we stand. The synthetic test data gives us comprehensive coverage. The custom evaluators capture what matters for educational content.

In Chapter 9, we'll use this infrastructure to actually improve things. We'll add advanced retrieval techniques like hybrid search, reranking, and semantic chunking, and measure whether each change actually helps. The evaluation system we've built is what makes principled optimization possible.

You've got the tools to measure what matters. Now let's put them to work.
