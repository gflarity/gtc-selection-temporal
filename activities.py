from pydantic import BaseModel
import aiohttp
from temporalio import activity
from typing import Optional, List
from datetime import timedelta
import openai
import json
import asyncio
import string


# Pydantic model for Session
class Session(BaseModel):
    sessionID: str
    title: str
    abstract: str

# Pydantic model for ProcessSessionsInput
class ProcessSessionsInput(BaseModel):
    min_sessions: list[Session]
    new_sessions: list[Session]
    api_key: str


async def complete_with_schema(
    api_key: str,
    base_url: str,
    schema: dict,
    system_prompt: str,
    user_prompt: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
) -> tuple[str, str | None]:
    """
    Performs a completion using the provided schema, system prompt, and user prompt.
    The schema instruction is automatically appended to the system prompt.
    @param schema - The JSON schema object that the response should adhere to.
    @param system_prompt - The base system prompt (e.g., "You are a helpful AI assistant.").
    @param user_prompt - The user prompt to send to the model.
    @param model - The model to use for the completion (default: "meta-llama/Llama-3.3-70B-Instruct").
    @returns A promise that resolves to the parsed JSON response object.
    """
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    schema_str = json.dumps(schema)
    system_message = f"{system_prompt} Here's the json schema you need to adhere to: <schema>{schema_str}</schema>"

    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        stream=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise Exception("Failed to generate a response.")

    reasoning = None
    if hasattr(response.choices[0].message, 'reasoning_content'):
        reasoning = response.choices[0].message.reasoning_content

    return content, reasoning

# Cache for comparison results
cache = {}
async def compare_sessions(api_key: str, a: Session, b: Session) -> int:
    # Generate cache keys
    key_ab = f"{a.sessionID}_{b.sessionID}"
    key_ba = f"{b.sessionID}_{a.sessionID}"

    # Check cache
    if key_ab in cache:
        print(f"Cache hit for {a.title} vs. {b.title}")
        return cache[key_ab]
    elif key_ba in cache:
        print(f"Cache hit for {b.title} vs. {a.title}")
        return -cache[key_ba]

    # Define schema for comparison result
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"result": {"type": "integer", "enum": [1, -1]}},
        "required": ["result"],
        "additionalProperties": False,
    }

    system_prompt = "You are an expert at comparing GTC conference sessions given their titles and abstracts."
    model = "deepseek-ai/DeepSeek-R1"
    user_prompt = (
        f"You will analyze Titles & Abstracts of two GTC sessions (A and B) to determine which better emphasizes an academic discussions of techniques that lead to cost reduction and/or improved time/resource efficiency in AI/ML workflows, deployments, or applications."
        f"Prioritize abstacts that include evidence of concrete benefits over superficial buzz.\n"
        f"Avoid sessions that involve hyperbole/pricing-focused selling\n"
        f"Include sessions that seem to apply broadly (multiple domains or framework-agnostic) rather than niche/hardware-specific optimizations or verticals.\n"
        f"Penalize vague/generic phrases; reward specific frameworks, real-world examples, and caveats acknowledging limits.\n\n"
        f"Template for Analysis\n"
        f"Step-by-Step Instructions:\n\n"
        f"Analyze Criteria for Section A - Assign scores ((1-5): Cost/Efficiency Emphasis | Avoidance of Self-Promotion | Accessibility Generality | Supported Claims.\n\n"
        f"Analyze Criteria for Section B - Same framework.\n"
        f"(Compare relative strengths for each criteria).\n\n"
        f"Make Final Call. Consider:\n\n"
        f"• It must be related to AI/ML systems engineering, no using AI/ML to solve some problems such as autonomous driving, or cancer etc.\n"
        f"• Does A/B discuss actual financial metrics (e.g., 20%↑ inference speed) rather than ROI hype?\n"
        f"• If A focuses on custom ASIC chip design & B improves PyTorch pipeline design → B has wider ML impact.\n\n"
        f"VERDICT Format → {{-1 if A>B, 1 if B>=A}}: {{Return only \"-1\" or \"1\" without explanation.}}\n\n"
        f"Sessions Provided: "
        f"Title for paper a: ```{a.title}``` "
        f"Abstract for paper a: ```{a.abstract}``` "
        f"Title for paper b: ```{b.title}``` "
        f"Abstract for paper b: ```{b.abstract}```"
    )

    print(f"Comparing {a.title} with {b.title}")
    content, reasoning = await complete_with_schema(
        api_key,
        "https://api.centml.com/openai/v1",
        schema,
        system_prompt,
        user_prompt,
        model
    )

    print("reasoning", reasoning)
    print("content", content)
    obj = json.loads(content)
    result = obj["result"]
    cache[key_ab] = result
    return result

class FetchSessionsInput(BaseModel):
    from_offset: Optional[int]

@activity.defn
async def fetch_sessions(fetch_sessions_input: FetchSessionsInput) -> List[Session]:
    from_offset = fetch_sessions_input.from_offset
    url = 'https://events.rainfocus.com/api/search'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'origin': 'https://www.nvidia.com',
        'priority': 'u=1, i',
        'referer': 'https://www.nvidia.com/',
        'rfapiprofileid': 'kPEXqZyAH2yKiQIBjup0YsyR0slBWDne',
        'rfwidgetid': 'DrwI9RRokZ85dwAXIgogWYLFShMaC93k'
    }
    params = {
        "type": "session",
        "browserTimezone": "America/Toronto",
        "catalogDisplay": "list",
    }
    if from_offset is not None:
        params["from"] = str(from_offset)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=params) as response:
            response.raise_for_status()
            obj = await response.json()

    new_sessions: list[Session] = []
    if from_offset is None:
        for d in obj["sectionList"][0]["items"]:
            new_sessions.append(Session(**d))
    else:
        for d in obj["items"]:
            new_sessions.append(Session(**d))

    return new_sessions

async def async_filter(arr: list, predicate, concurrency: int = 1) -> list:
    index = 0
    results: list[bool] = [False] * len(arr)

    async def worker():
        nonlocal index
        while index < len(arr):
            current_index = index
            index += 1
            attempts = 0
            while attempts < 2:
                try:
                    results[current_index] = await predicate(arr[current_index])
                    break
                except Exception as error:
                    print(error)
                    print("Retrying item", current_index)
                    attempts += 1
            if attempts == 2:
                try:
                    results[current_index] = await predicate(arr[current_index])
                except Exception as error:
                    print(error)
                    print("Retrying item", current_index)

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)

    return [item for idx, item in enumerate(arr) if results[idx]]

class FilterSessionsInput(BaseModel):
    sessions: List[Session]
    api_key: str

@activity.defn
async def filter_sessions(filter_sessions_input: FilterSessionsInput) -> List[Session]:
    

    filtered_sessions = await async_filter(
        filter_sessions_input.sessions,
        predicate=lambda session: process_session_filter(filter_sessions_input.api_key, session),
        concurrency=3
    )
    return filtered_sessions

async def process_session_filter(api_key: str, session: Session) -> bool:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"result": {"type": "boolean"}},
        "required": ["result"],
        "additionalProperties": False,
    }
    system_prompt = "You are an expert at evaluating AI/ML conference sessions, specifically focusing on identifying sessions that discuss optimizations of AI/ML systems themselves."
    user_prompt = (
        f"Analyze the following GTC session title and abstract. Determine if it meets MANY of the following criteria for inclusion:\n\n"
        f"**Core Focus**: The session must explicitly address technical methods for optimizing the computational efficiency of AI/ML systems themselves. This includes:\n"
        f"- Reducing computational costs (e.g., energy, hardware, cloud expenses) of AI/ML models or training processes.\n"
        f"- Improving efficiency (e.g., faster training, optimized inference, reduced latency, smaller models).\n"
        f"- Techniques such as quantization, pruning, sparsity, distillation, parallelization, or novel architectures aimed at making AI/ML systems more efficient.\n\n"
        f"**Important**: The session must focus on improving the AI/ML techniques or systems, not on applying AI/ML to optimize other domains or processes.\n\n"
        f"**Technical Depth**: The session should:\n"
        f"- Mention frameworks/libraries (e.g., TensorFlow, PyTorch, CUDA) or tools (e.g., Triton, TensorRT) used in the optimization process.\n"
        f"- Describe algorithms, workflows, or provide measurable results (e.g., '40% fewer FLOPs,' '2x speedup on A100') related to AI/ML system optimization.\n"
        f"- Avoid vague claims (e.g., 'revolutionary,' 'industry-leading') without technical justification.\n\n"
        f"**Exclusion Rules**: Reject sessions that:\n"
        f"- Avoid workshop sessions, ie those that include 'Learn how to'"    
        f"- Avoid session that Focus on applying AI/ML to optimize other domains or processes, rather than optimizing the AI/ML systems themselves.\n"
        f"- Avoid sessions that are product demos, company announcements, or partnerships without technical detail on AI/ML optimization.\n"
        f"- Use excessive marketing language (e.g., 'transform your business,' 'exclusive solution').\n"
        f"- Lack concrete methodologies for AI/ML optimization (e.g., only high-level use cases, no benchmarks).\n\n"
        f"**Examples**:\n"
        f"**Included**:\n"
        f"Title: 'Dynamic Sparsity for Efficient Transformer Training'\n"
        f"Abstract: 'We present a PyTorch-based method to dynamically prune attention heads during training, reducing memory usage by 35% on GPT-3-scale models without accuracy loss.'\n"
        f"→ Rationale: Focuses on optimizing an AI/ML system (transformer training) with technical details (pruning, PyTorch, 35% memory reduction).\n\n"
        f"**Excluded**:\n"
        f"Title: 'Using AI to Optimize Energy Consumption in Data Centers'\n"
        f"Abstract: 'Learn how our AI-powered platform can reduce energy costs by predicting and optimizing data center cooling systems.'\n"
        f"→ Rationale: Focuses on applying AI to optimize data center energy use, not on optimizing the AI system itself.\n\n"
        f"**Another Excluded**:\n"
        f"Title: 'Accelerate AI with XYZ Corporation’s Cloud Platform'\n"
        f"Abstract: 'Discover how our industry-leading platform empowers teams to deploy models faster and cut costs!'\n"
        f"→ Rationale: Lacks technical details on AI/ML optimization methods; uses promotional language.\n\n"
        f"**Session to Evaluate**:\n"
        f"Title: ``` {session.title} ```\n"
        f"Abstract: ``` {session.abstract} ```\n"
        f"Based on the criteria above, should this session be included? Provide a brief justification.\n"
    )
    model = "deepseek-ai/DeepSeek-R1"
    content, reasoning = await complete_with_schema(
        api_key,
        "https://api.centml.com/openai/v1",
        schema,
        system_prompt,
        user_prompt,
        model
    )
    print("abstract", session.abstract)
    print("filter reasoning", reasoning)
    print("filter content", content)
    obj = json.loads(content)
    return obj["result"]

async def find_insertion_point(api_key: str, min_sessions: list[Session], new_session: Session) -> int:
    left = 0
    right = len(min_sessions)
    while left < right:
        mid = (left + right) // 2
        if await compare_sessions(api_key, new_session, min_sessions[mid]) == -1:
            right = mid
        else:
            left = mid + 1
    return left

@activity.defn
async def process_sessions(input: ProcessSessionsInput) -> List[Session]:
    min_sessions = input.min_sessions
    new_sessions = input.new_sessions
    api_key = input.api_key # Extract api_key from input
    for new_session in new_sessions:
        if len(min_sessions) == 0:
            min_sessions.append(new_session)
        elif len(min_sessions) < 10:
            pos = await find_insertion_point(api_key, min_sessions, new_session)
            min_sessions.insert(pos, new_session)
        else:
            if await compare_sessions(api_key, new_sessions[-1], min_sessions[-1]) == 1: # Corrected: compare new_session with min_sessions[-1]
                pos = await find_insertion_point(api_key, min_sessions, new_session)
                min_sessions.insert(pos, new_session)
                min_sessions.pop(-1)
    return min_sessions

def contains_non_standard_characters(text):
    standard_chars = string.ascii_letters + string.digits + string.punctuation + " "
    for char in text:
        if char not in standard_chars:
            return True
    return False


class FilterNonEnglishSessionsInput(BaseModel):
    sessions: List[Session]

@activity.defn
async def filter_non_english_sessions(filter_non_english_sessions_input: FilterNonEnglishSessionsInput) -> List[Session]:
    filtered_sessions = []
    for session in filter_non_english_sessions_input.sessions:
        if not contains_non_standard_characters(session.title) and not contains_non_standard_characters(session.abstract):
            filtered_sessions.append(session)
    return filtered_sessions