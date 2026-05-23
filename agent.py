from typing import TypedDict, List

try:
    from openai import OpenAI

    import instructor

    from pydantic import BaseModel, Field

    from langgraph.graph import (
        StateGraph,
        END,
    )

except ImportError as e:
    raise ImportError(
        f"Import failed:\n{e}"
    )


from retrieval import RetrievalPipeline


# ------------------------------------------------------
# OLLAMA CONFIG
# ------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434/v1"

MODEL_NAME = "phi3"


# ------------------------------------------------------
# LANGGRAPH STATE
# ------------------------------------------------------

class AgentState(TypedDict):

    question: str

    documents: List[str]

    generation: str

    retries: int


# ------------------------------------------------------
# STRUCTURED OUTPUT MODEL
# ------------------------------------------------------

class GroundingCheck(BaseModel):

    is_grounded: bool = Field(
        description=(
            "Whether answer is grounded "
            "in provided context."
        )
    )

    reasoning: str = Field(
        description="Explanation of decision."
    )


# ------------------------------------------------------
# OLLAMA CLIENT
# ------------------------------------------------------

client = instructor.from_openai(
    OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)


# ------------------------------------------------------
# RETRIEVAL PIPELINE
# ------------------------------------------------------

retriever = RetrievalPipeline()


# ------------------------------------------------------
# NODE 1: RETRIEVE
# ------------------------------------------------------

def retrieve_node(state: AgentState):

    print("\n[RETRIEVE NODE]")

    docs = retriever.search(
        state["question"]
    )

    texts = []
    
    for d in docs:
        formatted = (
        f"[SOURCE: {d['chunk_id']}]\n"
        f"{d['text']}"
    )
        texts.append(formatted)

    

    return {
        "documents": texts
    }


# ------------------------------------------------------
# NODE 2: GENERATE
# ------------------------------------------------------
def generate_node(state: AgentState):

    print("\n[GENERATE NODE]")

    context = "\n\n".join(
        state["documents"]
    )

    prompt = f"""
You are a precise academic research assistant.

Answer ONLY using facts explicitly stated
in the provided context.

Do not speculate.
Do not exaggerate claims.
Do not repeat information.

Keep the answer concise,
factual, and under 200 words.

Cite relevant SOURCE IDs used
in the answer.

If the answer cannot be found,
say you do not know.

CONTEXT:
{context}

QUESTION:
{state["question"]}
"""

    try:

        response = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        ).chat.completions.create(
            model=MODEL_NAME,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual "
                        "academic assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = ""

        print("\n[STREAMING ANSWER]\n")

        for chunk in response:

            content = (
                chunk
                .choices[0]
                .delta
                .content
            )

            if content:

                print(
                    content,
                    end="",
                    flush=True,
                )

                answer += content

        print("\n")

    except Exception as e:

        answer = (
            f"Generation failed:\n{e}"
        )

    return {
        "generation": answer,
        "retries": state["retries"] + 1,
    }


# ------------------------------------------------------
# NODE 3: GROUNDING CHECK
# ------------------------------------------------------

def grounding_check(state: AgentState):

    print("\n[GROUNDING CHECK NODE]")

    context = "\n\n".join(
        state["documents"]
    )

    prompt = f"""
You are a strict evaluator.

Determine whether the answer is fully supported
by the provided context.

Return ONLY:

TRUE - if grounded

FALSE - if not grounded

CONTEXT:
{context}

ANSWER:
{state["generation"]}
"""

    try:

        response = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        ).chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        result = (
            response
            .choices[0]
            .message
            .content
            .strip()
            .upper()
        )

        print(f"\nGrounding Result: {result}")

        if "TRUE" in result:

            return END

        if state["retries"] >= 3:

            print(
                "\n[INFO] Max retries reached."
            )

            return END

        return "generate"

    except Exception as e:

        print(
            f"\n[ERROR] Grounding check failed:\n{e}"
        )

        return END

# ------------------------------------------------------
# BUILD GRAPH
# ------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node(
    "retrieve",
    retrieve_node
)

workflow.add_node(
    "generate",
    generate_node
)

workflow.set_entry_point(
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "generate"
)

workflow.add_conditional_edges(
    "generate",
    grounding_check,
)

app = workflow.compile()


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------

if __name__ == "__main__":

    question = input(
    "\nAsk your research question: "
)

    result = app.invoke(
        {
            "question": question,
            "documents": [],
            "generation": "",
            "retries": 0,
        }
    )

    print("\n========== FINAL ANSWER ==========\n")

    print(result["generation"])
    retriever.qdrant.close()
