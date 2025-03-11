from pydantic import BaseModel
import aiohttp
from temporalio import activity
from typing import Optional, List
from datetime import timedelta
import CentML
import json
import asyncio
import string

# Note: 'timedelta' is imported but not used in this file. It may be a leftover from previous iterations.

class Session(BaseModel):
    """Represents a GTC conference session with essential metadata."""
    sessionID: str  # Unique identifier for the session
    title: str      # Title of the session
    abstract: str   # Abstract describing the session content

class ProcessSessionsInput(BaseModel):
    """Input model for the process_sessions activity."""
    min_sessions: list[Session]  # Current list of top sessions (sorted by relevance)
    new_sessions: list[Session]  # New sessions to evaluate and potentially add
    api_key: str                 # API key for CentML Serverless API

async def complete_with_schema(
    api_key: str,
    base_url: str,
    schema: dict,
    system_prompt: str,
    user_prompt: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
) -> tuple[str, str | None]:
    """
    Performs a LLM completion with a specified JSON schema.

    Constructs a system message incorporating the schema and requests a chat completion
    that adheres to it. Used as a utility for evaluating sessions via natural language processing.

    Args:
        api_key (str): The API key for authenticating with CentML Serverless API.
        base_url (str): The base URL for the CentML Serverless API endpoint.
        schema (dict): JSON schema that the response must follow.
        system_prompt (str): Base system prompt for the AI model.
        user_prompt (str): Specific prompt provided by the user.
        model (str, optional): Model identifier for CentML Serverless API. Defaults to "meta-llama/Llama-3.3-70B-Instruct".

    Returns:
        tuple[str, str | None]: A tuple containing:
            - The response content as a JSON string.
            - Optional reasoning content if provided by the API, else None.

    Raises:
        Exception: If the API fails to generate a response.
    """
    client = CentML.AsyncCentML(api_key=api_key, base_url=base_url)
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
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise Exception("Failed to generate a response.")

    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
    return content, reasoning

# Cache for storing comparison results to avoid redundant API calls
cache = {}

async def compare_sessions(api_key: str, a: Session, b: Session) -> int:
    """
    Compares two sessions to determine which better discusses AI/ML system optimizations.

    Uses CentML's Serverless API to evaluate session titles and abstracts based on criteria favoring
    academic, technical discussions over promotional content. Results are cached.

    Args:
        api_key (str): API key for CentML Serverless API.
        a (Session): First session to compare.
        b (Session): Second session to compare.

    Returns:
        int: Comparison result where:
            - -1 means 'a' is more relevant than 'b'.
            - 1 means 'b' is more relevant or equal to 'a'.
    """
    # Generate bidirectional cache keys
    key_ab = f"{a.sessionID}_{b.sessionID}"
    key_ba = f"{b.sessionID}_{a.sessionID}"

    # Check cache to avoid redundant comparisons
    if key_ab in cache:
        print(f"Cache hit for {a.title} vs. {b.title}")
        return cache[key_ab]
    elif key_ba in cache:
        print(f"Cache hit for {b.title} vs. {a.title}")
        return -cache[key_ba]

    # Schema for expected comparison result
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
        api_key, "https://api.centml.com/CentML/v1", schema, system_prompt, user_prompt, model
    )

    print("reasoning", reasoning)
    print("content", content)
    obj = json.loads(content)
    result = obj["result"]
    cache[key_ab] = result  # Cache the result
    return result

class FetchSessionsInput(BaseModel):
    """Input model for the fetch_sessions activity."""
    from_offset: Optional[int]  # Optional offset for pagination, None for initial fetch

@activity.defn
async def fetch_sessions(fetch_sessions_input: FetchSessionsInput) -> List[Session]:
    """
    Fetches sessions from the GTC conference API.

    Sends a POST request to retrieve session data, supporting pagination via an offset.
    Part of a Temporal.io workflow for session processing.

    Args:
        fetch_sessions_input (FetchSessionsInput): Input containing an optional offset.

    Returns:
        List[Session]: List of parsed Session objects from the API response.

    Raises:
        aiohttp.ClientResponseError: If the API request fails.
    """
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
    # Response structure differs based on whether it's the initial fetch or paginated
    if from_offset is None:
        for d in obj["sectionList"][0]["items"]:
            new_sessions.append(Session(**d))
    else:
        for d in obj["items"]:
            new_sessions.append(Session(**d))

    return new_sessions

async def async_filter(arr: list, predicate, concurrency: int = 1) -> list:
    """
    Asynchronously filters a list using a predicate function with concurrency.

    Employs multiple worker tasks to evaluate items concurrently, with retry logic for robustness.

    Args:
        arr (list): List of items to filter.
        predicate (callable): Async function returning a boolean for each item.
        concurrency (int, optional): Number of concurrent workers. Defaults to 1.

    Returns:
        list: Filtered list containing items where predicate returned True.
    """
    index = 0
    results: list[bool] = [False] * len(arr)

    async def worker():
        nonlocal index
        while index < len(arr):
            current_index = index
            index += 1
            attempts = 0
            while attempts < 2:  # Retry up to 2 times on failure
                try:
                    results[current_index] = await predicate(arr[current_index])
                    break
                except Exception as error:
                    print(error)
                    print("Retrying item", current_index)
                    attempts += 1
            if attempts == 2:  # Final attempt without retry
                try:
                    results[current_index] = await predicate(arr[current_index])
                except Exception as error:
                    print(error)
                    print("Retrying item", current_index)

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    return [item for idx, item in enumerate(arr) if results[idx]]

class FilterSessionsInput(BaseModel):
    """Input model for the filter_sessions activity."""
    sessions: List[Session]  # List of sessions to filter
    api_key: str             # API key for CentML Serverless API

@activity.defn
async def filter_sessions(filter_sessions_input: FilterSessionsInput) -> List[Session]:
    """
    Filters sessions for relevance to AI/ML system optimizations.

    Uses concurrent filtering to evaluate sessions against specific criteria via CentML API.
    Part of a Temporal.io workflow.

    Args:
        filter_sessions_input (FilterSessionsInput): Input with sessions and API key.

    Returns:
        List[Session]: Filtered list of relevant sessions.
    """
    filtered_sessions = await async_filter(
        filter_sessions_input.sessions,
        predicate=lambda session: process_session_filter(filter_sessions_input.api_key, session),
        concurrency=3  # Process 3 sessions concurrently
    )
    return filtered_sessions

async def process_session_filter(api_key: str, session: Session) -> bool:
    """
    Determines if a session meets criteria for AI/ML system optimization discussions.

    Evaluates a session's title and abstract using CentML's Serverless API against predefined criteria.

    Args:
        api_key (str): API key for CentML Serverless API.
        session (Session): Session to evaluate.

    Returns:
        bool: True if the session meets the criteria, False otherwise.
    """
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
        f"- Avoid workshop sessions, ie those that include 'Learn how to'\n"
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
        api_key, "https://api.centml.com/CentML/v1", schema, system_prompt, user_prompt, model
    )
    print("abstract", session.abstract)
    print("filter reasoning", reasoning)
    print("filter content", content)
    obj = json.loads(content)
    return obj["result"]

async def find_insertion_point(api_key: str, min_sessions: list[Session], new_session: Session) -> int:
    """
    Finds the insertion point for a new session in a sorted list using binary search.

    Assumes min_sessions is sorted in descending order of relevance (best first).
    Uses compare_sessions to maintain order.

    Args:
        api_key (str): API key for CentML Serverless API.
        min_sessions (list[Session]): Sorted list of top sessions.
        new_session (Session): Session to insert.

    Returns:
        int: Index where new_session should be inserted.
    """
    left = 0
    right = len(min_sessions)
    while left < right:
        mid = (left + right) // 2
        if await compare_sessions(api_key, new_session, min_sessions[mid]) == -1:  # new_session > mid
            right = mid
        else:
            left = mid + 1
    return left

@activity.defn
async def process_sessions(input: ProcessSessionsInput) -> List[Session]:
    """
    Maintains a list of the top 10 most relevant sessions.

    Evaluates new sessions and inserts them into a sorted list if they rank among the top 10.
    Part of a Temporal.io workflow.

    Note: Contains a potential bug in the comparison condition. Should likely check if
    new_session > min_sessions[-1] (i.e., compare_sessions returns -1) to include it,
    but currently checks == 1 and has a typo using new_sessions[-1].

    Args:
        input (ProcessSessionsInput): Input with current top sessions, new sessions, and API key.

    Returns:
        List[Session]: Updated list of top sessions (max 10).
    """
    min_sessions = input.min_sessions
    new_sessions = input.new_sessions
    api_key = input.api_key

    for new_session in new_sessions:
        if len(min_sessions) == 0:
            min_sessions.append(new_session)
        elif len(min_sessions) < 10:
            pos = await find_insertion_point(api_key, min_sessions, new_session)
            min_sessions.insert(pos, new_session)
        else:
            # Bug: Should be 'new_session' not 'new_sessions[-1]', and condition should be == -1
            # Current: if min_sessions[-1] >= new_session, which incorrectly triggers insertion
            # Should be: if new_session > min_sessions[-1] (i.e., == -1)
            if await compare_sessions(api_key, new_sessions[-1], min_sessions[-1]) == 1:
                pos = await find_insertion_point(api_key, min_sessions, new_session)
                min_sessions.insert(pos, new_session)
                min_sessions.pop(-1)  # Remove least relevant session
    return min_sessions

def contains_non_standard_characters(text: str) -> bool:
    """
    Checks if text contains non-standard (non-ASCII) characters.

    Standard characters include ASCII letters, digits, punctuation, and space.

    Args:
        text (str): Text to evaluate.

    Returns:
        bool: True if non-standard characters are present, False otherwise.
    """
    standard_chars = string.ascii_letters + string.digits + string.punctuation + " "
    return any(char not in standard_chars for char in text)

class FilterNonEnglishSessionsInput(BaseModel):
    """Input model for the filter_non_english_sessions activity."""
    sessions: List[Session]  # List of sessions to filter

@activity.defn
async def filter_non_english_sessions(filter_non_english_sessions_input: FilterNonEnglishSessionsInput) -> List[Session]:
    """
    Filters out sessions likely containing non-English text.

    Removes sessions with titles or abstracts containing non-standard characters.
    Part of a Temporal.io workflow.

    Args:
        filter_non_english_sessions_input (FilterNonEnglishSessionsInput): Input with sessions to filter.

    Returns:
        List[Session]: List of sessions likely in English.
    """
    filtered_sessions = [
        session for session in filter_non_english_sessions_input.sessions
        if not (contains_non_standard_characters(session.title) or contains_non_standard_characters(session.abstract))
    ]
    return filtered_sessions