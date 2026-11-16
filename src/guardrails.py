"""Guardrails: decline questions the celebrity would not know about."""

from __future__ import annotations

import re

from .llm import query_llm      #Imports the function that calls Gemini

# Patterns indicating post-March 2026 future events (knowledge cutoff: March 31, 2026)
_POST_CUTOFF_PATTERNS = [
    r"\b202[7-9]\b",
    r"\b20[3-9]\d\b",
    r"\b(?:april|may|june|july|august|september|october|november|december)\s+2026\b",
    r"\b2026\s+(?:nba\s+)?(?:playoffs|finals|championship|champion|draft|offseason|free\s+agency|world\s+cup)\b",
    r"\bwho\s+(?:won|wins|will\s+win)\s+the\s+2026\b",
    r"\b2026[-–/]27\b",
    r"\b2026[-–/]2027\b",
]

# Patterns for clear non-Luka / non-basketball technical or professional tasks. These are questions/ patterns that luka would clearly not know about.
_OUT_OF_EXPERTISE_PATTERNS = [
    r"\b(?:write|debug|compile|refactor)\s+(?:a\s+|some\s+|the\s+)?(?:python|javascript|typescript|java|c\+\+|rust|sql|html|css|react|code|script|function|algorithm)\b",
    r"\b(?:def\s+\w+\(|SELECT\s+.*\s+FROM\s+|import\s+numpy|import\s+torch|console\.log\()\b",
    r"\b(?:solve|integrate|differentiate|prove)\s+(?:this\s+)?(?:differential\s+equation|integral|derivative|matrix|eigenvalue|theorem)\b",
    r"\b(?:diagnose\s+my|prescribe\s+medication|what\s+dosage\s+of|interpret\s+my\s+lab\s+results)\b",
    r"\b(?:file\s+my\s+taxes|tax\s+deduction\s+under\s+irs|draft\s+a\s+legal\s+contract)\b",
]

# Simple greetings/farewells/acks that are always in scope. These are questions/ patterns that luka would clearly know about.
_CASUAL_CHAT_PATTERN = re.compile(
    r"^\s*(?:hi|hey|hello|yo|sup|what'?s\s+up|how\s+are\s+you|how'?s\s+it\s+going|"
    r"good\s+morning|good\s+evening|thanks|thank\s+you|hvala|appreciate\s+it|"
    r"bye|goodbye|see\s+ya|see\s+you|later|take\s+care|good\s+luck)[!.,?\sluka\s+man\s+bro]*$",
    re.IGNORECASE,
)


def _is_post_cutoff_regex(text: str) -> bool:
    lowered = text.lower() #Make the text lowecase
    return any(re.search(pat, lowered) for pat in _POST_CUTOFF_PATTERNS) #Search for any patterns in post cutoff patterns within our text, if any are found, return True, otherwise return False.


#Same as above method for this one, we search for any patterns in the out of expertise patterns within our text, if any are found, return True, otherwise return False.
def _is_out_of_expertise_regex(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in _OUT_OF_EXPERTISE_PATTERNS)

#This funciton formats the recent history of conversations upto 3 turns. 
def _format_recent_history(history: list[dict], max_turns: int = 6) -> str:
    if not history:     #If history is empty, return a message saying no prior turns.
        return "(No prior turns)"
    lines = [] #Initialize an empty list to store the lines of the recent history.
    for turn in history[-max_turns:]: #Takes the last 6 turns from history.
        role = "User" if turn.get("role") == "user" else "Luka" #If the role is user, set the role to User, otherwise set it to Luka.
        content = (turn.get("content") or "").strip() #Get the content of the turn and strip any whitespace around it. If there is no content, return an empty string
        if content: #If the content is not empty, add the role and content to the list.
            lines.append(f"{role}: {content}") #Add the role and content to the list.
    return "\n".join(lines) if lines else "(No prior turns)" #Join the lines with a newline character and return the result, if the list is empty, return a message saying no prior turns.


def out_of_scope(user_message: str, history: list[dict]) -> bool:
    """Return True if the message asks about events after the knowledge cutoff or outside the celebrity's expertise."""
    cleaned = (user_message or "").strip()
    if not cleaned: #If cleaned (Which is the user message) is empty, return False.
        return False

    if _CASUAL_CHAT_PATTERN.match(cleaned): #If the user message matches the casual chat pattern, return False.
        return False

    if _is_post_cutoff_regex(cleaned) or _is_out_of_expertise_regex(cleaned): #If the user message matches the post cutoff regex or the out of expertise regex, return True.
        return True

    history_text = _format_recent_history(history) #Format the recent history of conversations upto 3 turns.
    classifier_prompt = f"""You are a scope and knowledge-cutoff classifier for an AI agent roleplaying NBA star Luka Dončić.

Knowledge Cutoff: March 31, 2026.

Decide whether the User Message is IN_SCOPE or OUT_OF_SCOPE:

1. POST-CUTOFF (OUT_OF_SCOPE):
   - Any question asking about events, game results, stats, awards, trades, champions, or news occurring AFTER March 31, 2026 (for example: April 2026 or later, the 2026 NBA Playoffs or 2026 NBA Finals outcomes, 2026 FIFA World Cup outcomes, the 2026-27 season, or years 2027+).
   - Note: Luka's February 2025 trade from Dallas to the Los Angeles Lakers and games/events up through March 2026 ARE before the cutoff and are IN_SCOPE.

2. OUTSIDE EXPERTISE (OUT_OF_SCOPE):
   - Technical, academic, or professional requests that a pro basketball player would not answer: software engineering/coding, advanced math/physics/chemistry, medical diagnosis/treatment, legal or tax advice, corporate finance, or random academic/encyclopedic trivia completely unrelated to Luka Dončić, basketball, Slovenia, or his known interests.

3. IN_SCOPE:
   - Anything about Luka Dončić's life, childhood in Slovenia, Union Olimpija, Real Madrid, EuroLeague, Dallas Mavericks, Los Angeles Lakers, NBA career, stats, games, teammates, coaches, rivalries, basketball strategy, training, Jordan Brand shoes, foundation, or his known hobbies (Overwatch/gaming, cars, fishing, watching soccer/Real Madrid, Balkan music/food).
   - Casual conversation, greetings, small talk, opinions on basketball, or saying goodbye.
   - Follow-up questions referring to an in-scope topic in the conversation history.

Recent conversation:
{history_text}

User message:
{cleaned}

Respond with exactly one token: IN_SCOPE or OUT_OF_SCOPE."""

    verdict = query_llm(classifier_prompt, temperature=0.0).strip().upper() #If the conversation is hard to classify, we query the LLM to classify it.
    return "OUT_OF_SCOPE" in verdict #If the verdict is OUT_OF_SCOPE, return True, otherwise return False.


def decline(user_message: str) -> str:
    """Return an in-character response saying the celebrity doesn't know."""
    cleaned = (user_message or "").strip()
    is_future = _is_post_cutoff_regex(cleaned) #If the user message matches the post cutoff regex, return True, otherwise return False.

    decline_prompt = f"""You are NBA star Luka Dončić (speaking in March 2026).
The user just asked you a question that you must decline in character because either:
- It asks about future events after March 2026 (like the 2026 NBA Playoffs/Finals or future years), OR
- It asks about a technical, academic, medical, legal, or unrelated topic outside a basketball player's expertise.

Write a short (1 to 2 sentences), natural, slightly deadpan in-character refusal in Luka's voice:
- If it's about the future (after March 2026), say you can't see the future, man — you're just taking it one game at a time right now and we'll have to wait and see how that plays out.
- If it's outside your expertise, laugh it off casually (e.g., "I mean, I just hoop and play some Overwatch, man — I definitely don't know anything about that.").
- Do NOT attempt to answer the out-of-scope question.
- Output ONLY Luka's 1-2 sentence reply (no quotes).

User message:
{cleaned}

Luka's decline:"""

    try:
        reply = query_llm(decline_prompt, temperature=0.2).strip().strip('"')
        if reply:
            return reply
    except Exception:
        pass

    if is_future: #This is the default response if the user message matches the post cutoff regex and the LLM call fails
        return (
            "I mean, I can't see the future, man — we're only in March 2026 right now, "
            "so we'll just have to take it one game at a time and see how that plays out."
        )
    return (
        "I mean, I just hoop and play some Overwatch, man — I definitely don't know "
        "anything about that."
    )
