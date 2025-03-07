from dataclasses import dataclass
import aiohttp  # aiohttp is needed in the activity file
from temporalio import activity
from typing import Optional, List, Dict
from datetime import timedelta
import openai
import json
import os

CENTML_API_KEY = os.getenv("CENTML_API_KEY")

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
    if hasattr(response.choices[0].message, 'reasoning_content'): # Changed from "reasoning_content" in response.choices[0].message
        reasoning = response.choices[0].message.reasoning_content

    return content, reasoning

class Session:
    def __init__(self, sessionID: str, title: str, abstract: str):
        self.sessionID = sessionID
        self.title = title
        self.abstract = abstract

    def to_dict(self):
        return {"sessionID": self.sessionID, "title": self.title, "abstract": self.abstract}

    @classmethod
    def from_dict(cls, d):
        return cls(d["sessionID"], d["title"], d["abstract"])

# Initialize a cache to store comparison results
cache = {} # Changed from Map to dict
async def compare_sessions(a: Session, b: Session) -> int: # Changed anonymous function to named function
    # Generate keys for both comparison orders
    key_ab = f"{a.sessionID}_{b.sessionID}" # Changed from template literals to f-strings
    key_ba = f"{b.sessionID}_{a.sessionID}" # Changed from template literals to f-strings

    # Check if the result is cached for a vs. b
    if key_ab in cache: # Changed from cache.has(keyAB) to key_ab in cache
        print(f"Cache hit for {a.title} vs. {b.title}") # Changed from console.log to print
        return cache[key_ab] # Changed from cache.get(keyAB)! to cache[key_ab]
    # Check if the result is cached for b vs. a and negate it
    elif key_ba in cache: # Changed from cache.has(keyBA) to key_ba in cache
        print(f"Cache hit for {b.title} vs. {a.title}") # Changed from console.log to print
        return -cache[key_ba] # Changed from cache.get(keyBA)! to cache[key_ba]
    # If not cached, perform the comparison
    else:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "result": {
                    "type": "integer",
                    "enum": [1, -1],
                },
            },
            "required": ["result"],
            "additionalProperties": False,
        }

        system_prompt = (
            "You are an expert at comparing GTC conference sessions given their titles and abstracts."
        )
        model = "deepseek-ai/DeepSeek-R1"
        user_prompt = (
            f"You will analyze Titles & Abstracts of two GTC sessions (A and B) to determine which better emphasizes cost reduction, efficiency improvements, and aligns with the following success criteria:\n\n"
            f"Greater Impact/Optimization Focus: More actionable strategies addressing measurable cost reduction and time/resource efficiency in AI/ML workflows, deployments, or applications. Prioritizes metrics/evidence of concrete benefits over superficial buzz.\n"
            f"Less Self-Promotion: Avoids hyperbole/pricing-focused selling; highlights challenges/successes usable across models/tools/organizations.\n"
            f"Broader Applicability: Solutions/insights apply broadly (multiple domains or framework-agnostic) vs. niche/hardware-specific optimizations or verticals.\n"
            f"⤷ For authenticity: Penalize vague/generic phrases; reward specific frameworks, real-world examples, and caveats acknowledging limits.\n\n"
            f"Template for Analysis\n"
            f"Step-by-Step Instructions:\n\n"
            f"Analyze Criteria for Section A - Assign scores ((1-5): Cost/Efficiency Emphasis | Avoidance of Self-Promotion | Accessibility Generality | Supported Claims.\n\n"
            f"Analyze Criteria for Section B - Same framework.\n"
            f"(Compare relative strengths for each criteria).\n\n"
            f"Make Final Call. Consider:\n\n"
            f"• It must be related to AI/ML systems engineering, no using AI/ML to solve some problem.\n"
            f"• Does A/B discuss actual financial metrics (e.g., 20%↑ inference speed) rather than ROI hype?\n"
            f"• Did one omit practical implementation roadblocks? (= possible overselling sign).\n"
            f"• If A focuses on custom ASIC chip design & B improves PyTorch pipeline design → B has wider ML impact.\n\n"
            f"VERDICT Format → {{-1 if A>B, 1 if B>=A}}: {{Return only \"-1\" or \"1\" without explanation.}}\n\n"
            f"Sessions Provided: " +
            "Title for paper a: ```" +
            a.title +
            "``` " +
            "Abstract for paper a: ```" +
            a.abstract +
            "``` " +
            "Title for paper b: ```" +
            b.title +
            "``` " +
            "Abstract for paper b: ```" +
            b.abstract +
            "```"
        )

        print(f"Comparing {a.title} with {b.title}") # Changed from console.log to print
        content, reasoning = await complete_with_schema(
            CENTML_API_KEY, # Replace with actual API key or get from environment variables
            "https://api.centml.com/openai/v1",
            schema,
            system_prompt,
            user_prompt,
            model
        )

        print("reasoning", reasoning) # Changed from console.log to print
        print("content", content) # Changed from console.log to print
        obj = json.loads(content)
        result = obj["result"]

        # Cache the result for a vs. b
        cache[key_ab] = result # Changed from cache.set(keyAB, result) to cache[key_ab] = result
        return result


@activity.defn
async def fetch_sessions(from_offset: Optional[int]) -> List[Dict]:
    # aiohttp and request logic belongs HERE, in the ACTIVITY
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
        
    new_sessions: list = []    
    if from_offset is None:
        for d in obj["sectionList"][0]["items"]:

            new_sessions.append(Session.from_dict(d).to_dict())
    else:
        for d in obj["items"]:
            new_sessions.append(Session.from_dict(d).to_dict())

    return new_sessions

async def find_insertion_point(min_sessions: list[Session], new_session: Session) -> int:
    left = 0
    right = len(min_sessions)
    while left < right:
        mid = (left + right) // 2
        if await compare_sessions(new_session, min_sessions[mid]) == 1:
            right = mid
        else:
            left = mid + 1
    return left

@dataclass
class ProcessSessionsInput:
    min_sessions: list[dict]
    new_sessions: list[dict]

@activity.defn
async def process_sessions(input: ProcessSessionsInput) -> List[Dict]:
    min_sessions = [Session.from_dict(d) for d in input.min_sessions]
    new_sessions = [Session.from_dict(d) for d in input.new_sessions]
    for new_session in new_sessions:    
        if len(min_sessions) == 0:
            min_sessions.append(new_session)
        elif len(min_sessions) < 10:
            # Insert at the correct position to maintain sort
            pos = await find_insertion_point(min_sessions, new_session)
            min_sessions.insert(pos, new_session)
        else:
            # Only add if smaller than the largest
            if await compare_sessions(new_session, min_sessions[-1]) == 1:
                pos = await find_insertion_point(min_sessions, new_session)
                min_sessions.insert(pos, new_session)
                min_sessions.pop(-1)
    return [s.to_dict() for s in min_sessions]

