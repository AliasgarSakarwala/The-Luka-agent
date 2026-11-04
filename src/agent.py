"""Agent: wire guardrails, content, style, and sign-off into one turn."""
#Agent.py work is to communicate with the user and LLM and make sure the conversation is flowing smoothy by getting all the information"
#From the necessary places and adding it to the conversation"

"Basically orchestrate the conversation between the user and the LLM (Coordinating multiple modules)"
from __future__ import annotations #Helps with type hinting and other features. It is completely optional but recommended to help debugging and other developers read


import re #Regular expressions for pattern matching

#All these imports start with a dot because they are in the same directory as agent.py. Basically importing the other functions from the other files in the src directory.
from .content_module import content_step
from .conversations import Conversation, save_conversation
from .guardrails import decline, out_of_scope
from .llm import query_llm
from .memory_store import MemoryStore
from .style_module import stylize

#In python, developers use uppsercase keywords for constants. 
SIGNOFF_TEXT = "Appreciate it, man. Take care — Hvala and see you!" #Signoff text we will use to sign off the conversation.

# A list containing 4 strings, ech representing a different pattern that could be used to identify an explicit farewell.
# This is regex syntax, where \b stands for "word boundary". This is used to match the beginning or end of a word.
# An example of word boundary is "bye" in "byeby". Next, '?:/ is use to make the pattern non-capturing.
#Non capturing means that the pattern is not captured and will not be returned as a match. For example "byebye" will match "bye" but not "byebye".
#Next, the patterns are separated by | which means "or". So, the pattern will match if any of the patterns are found.
#\s+ means one or more spaces is obserbed between the words. So "see ya" will match "see ya" but not "seeyou".
# In the code below, adding ?: at the beginning tells the regex to not give thatt pair a number when grouping and the parentheses is used to 
#Group all the patterns together. However ? after a text tells the regex to make it non-capturing. So, "see ya" will match "see ya" but not "seeyou".
_EXPLICIT_FAREWELL_PATTERNS = [
    r"\b(?:bye|goodbye|good\s+bye|see\s+ya|see\s+you(?:\s+later)?|cya|ttyl|talk\s+to\s+you\s+later|catch\s+you\s+later|peace\s+out|take\s+care)\b",
    r"\b(?:gotta\s+go|gotta\s+run|i\s+have\s+to\s+go|i\s+need\s+to\s+go|i'?m\s+heading\s+out|i'?m\s+signing\s+off|i'?m\s+done(?:\s+for\s+now)?)\b",
    r"\b(?:that'?s\s+all(?:\s+for\s+now|\s+i\s+had)?|nothing\s+else\s+from\s+me|no\s+more\s+questions|that\s+wraps\s+it\s+up)\b",
    r"\b(?:have\s+a\s+good\s+(?:one|night|day|rest\s+of\s+your\s+day))\b",
]

#he job over here is to identify if the user is asking a followup question.
#There are 2 options here, either a question mark or a word that is a followup question.
#r" is the special string that tells the regex to treat the string as a raw string. So, it will not escape any characters.
#/? Matching to a literal ?/ Remeber, a plain ?> is optional grouping so we need to use a backslash to escape it.
#Like above, \b is a word boundary and is used to match the beginning or end of a word.
#re.IGNORECASE is used to make the regex case-insensitive. So, "What" will match "what" and "What".
#re.compile turns the pattern into a reusable object once so that the code can call it multiple times without recompiling the pattern each time.
#So we basically can just call _FOLLOWUP_QUESTION_CUES.search(cleaned) instead of re.compile(r"(?:\?|\b(?:what|who|when|where|why|how|which|can\s+you|could\s+you|tell\s+me|also|one\s+more\s+question|another\s+question)\b)", re.IGNORECASE).search(cleaned)

_FOLLOWUP_QUESTION_CUES = re.compile(
    r"(?:\?|\b(?:what|who|when|where|why|how|which|can\s+you|could\s+you|tell\s+me|also|one\s+more\s+question|another\s+question)\b)",
    re.IGNORECASE,
)

#This is the same as above but for a pure farewell. As you can see it starts with a ^ and ends with $. These are called anchors.
#This differs hugely from the above because using the anchors it tells the regex to match the beginning or end of the string, not just finding
#A word in the string. So we need to match that entire string within the sentence
#This works in blocks too so we can use | to match one of the phrases. eg "Thanks Luka thats all" is not part of one pattern below but since its
#words match all the pattens below, it will pass.
_PURE_FAREWELL_PATTERN = re.compile(
    r"^\s*(?:thanks(?:\s+so\s+much|\s+a\s+lot|\s+for\s+chatting|\s+for\s+your\s+time)?|"
    r"thank\s+you(?:\s+so\s+much|\s+for\s+chatting|\s+for\s+your\s+time)?|"
    r"appreciate\s+it|hvala|good\s+luck(?:\s+this\s+season|\s+next\s+game|\s+with\s+the\s+lakers)?|"
    r"bye|goodbye|see\s+ya|see\s+you(?:\s+later)?|take\s+care|have\s+a\s+good\s+one|"
    r"that'?s\s+all|gotta\s+go|gotta\s+run|catch\s+you\s+later|peace|luka|man|bro|[,!.—\-\s])+$",
    re.IGNORECASE,
)


def _is_pure_farewell(user_message: str) -> bool:
    """Return True when the message is solely a farewell/thank-you with no new question."""
    cleaned = (user_message or "").strip() #Is user_message is empty, choose an empty string. .strip() removes any whitespace at the beginning or end of the string.
    if not cleaned or "?" in cleaned:   #If the cleaned string is empty or contains a question mark, return False.
        return False
    if _FOLLOWUP_QUESTION_CUES.search(cleaned): #If the cleaned string contains a follow up question, return false
        return False
    return bool(_PURE_FAREWELL_PATTERN.match(cleaned)) #If the cleaned string matches the pure farewell pattern, return True else False


#This is one of the more sophisticated functions in the code that determines if the conversation should end or not.
def should_end(user_message: str, history: list[dict]) -> bool:
    """Determine if the conversation should end."""
    cleaned = (user_message or "").strip()
    if not cleaned:
        return False

    lowered = cleaned.lower() #Converts the cleaned string to lowercase. This is because we wuold pass our string to search with a case-insensitive regex.
    has_followup_question = bool(_FOLLOWUP_QUESTION_CUES.search(cleaned)) #If the cleaned string contains a follow up question, return true else false
    #This is a generator expressions that checks if the cleaned string matches ANY of the explicit farewell patterns.
    #Over here pat represents each of the patterns in the _EXPLICIT_FAREWELL_PATTERNS list. re.search is a function that searches the string for the pattern
    #If atleast one pattern matches, the generator expression returns True else False.
    has_explicit_farewell = any(re.search(pat, lowered) for pat in _EXPLICIT_FAREWELL_PATTERNS)

    if has_explicit_farewell and not has_followup_question:  #If the user is saying goodbye and not asking a followup question, return True
        return True

    # If the user just says a short thank-you/good-luck wrap-up without any new question,
    # check with the LLM whether they are closing the conversation.
    wrapup_cues = any(
        phrase in lowered
        for phrase in (
            "thanks",
            "thank you",
            "hvala",
            "appreciate your time",
            "thanks for chatting",
            "good luck",
            "bye",
            "see ya",
            "see you",
            "that's all",
            "thats all",
        )
    )
    if not wrapup_cues: #if the user is not saying a wrapup cue, return False
        return False

    if has_followup_question and not has_explicit_farewell: #If the user is asking a followup question and not saying goodbye, return False, Obvisouly
        return False

    recent_lines = []
    for turn in (history or [])[-4:]: #Over here history is a list of dictionaries containing the previous messages. -4: means list slicing which means get the last 4 itens within the list
        role = "User" if turn.get("role") == "user" else "Luka" #If its the users turn (Identified by the role in the dictionary being user) then set the role to User else Luka
        content = (turn.get("content") or "").strip() #Get the content of the turn and remove any whitespace at the beginning or end of the string.
        if content:
            recent_lines.append(f"{role}: {content}") #For each turn, append the role and content to the recent_lines list.
    history_text = "\n".join(recent_lines) if recent_lines else "(No prior turns)" #Join the recent_lines list with a newline character and if the list is empty, return "(No prior turns)"

    #This would be the prompt that we will pass to the LLM to determine if the conversation should end or not.
    prompt = f"""Determine whether the user is ending/closing the conversation with Luka Dončić in their latest message.

Rules:
- Reply END if the user is saying goodbye, signing off, or indicating they are done chatting (e.g., "Thanks Luka, bye!", "That's all I had, appreciate your time", "Good luck next game, gotta run").
- Reply CONTINUE if the user is still asking a question, continuing the topic, or just saying a polite "thanks" while continuing to chat.

Recent conversation:
{history_text}

Latest user message:
{cleaned}

Reply with one word only: END or CONTINUE."""

    try:
        verdict = query_llm(prompt, temperature=0.0).strip().upper() #query llm is imported from llm.py, it sends the prompt to the function, setting the temperature to 0.0
        #Setting temperature to 0.0 requests the LLM to generate the most likely answer, not the most creative one. The response is saved to verdict variable
        #Stripped from whitespace and converted to uppercase.
        return verdict.startswith("END") #If the verdict starts with END, return True else False
    except Exception: #Incase the API call fails
        return has_explicit_farewell


def add_signoff(response: str) -> str:
    """Add a signoff to the response."""
    base = (response or "").strip() #If response is empty, return empty string else strip whitespace around response
    if not base:
        return SIGNOFF_TEXT #Sign off text defined at the top

    lowered = base.lower()
    if "hvala and see you" in lowered or SIGNOFF_TEXT.lower() in lowered:
        return base #If the base string contains "hvala and see you" or the sign off text, return the base string

    return f"{base} {SIGNOFF_TEXT}"  #Else return the base string with the sign off text appended to it.

#Everything we have learned so far in this file, it all comes down to this function. 
#MemoryStore asnd Conversation are objects that are imported from the memory_store.py and conversations.py files respectively.
#So Conversation and MemoryStore can have multiple attributers which we can access by their object classifications
def respond(memory: MemoryStore, conversation: Conversation, user_message: str) -> str:
    """Run one turn of the agent, add it to the conversation, and mark the conversation ended if it is over."""
    if conversation.ended:
        raise ValueError("Conversation has already ended and cannot accept new messages.") #If the conversation is ended, raise an error because we cannot converse anymore

    cleaned_message = (user_message or "").strip() #The user message is cleaned by strip or is an emptyy string, which in the next line will raise an error
    if not cleaned_message:
        raise ValueError("User message cannot be empty.")

    history = list(conversation.turns) #The history is a list of dictionaries containing the previous messages, we created this list in the conversations.py file.
    ending = should_end(cleaned_message, history) #Gets a booleaan value to confirm if the conversation should end or not.

    if ending and _is_pure_farewell(cleaned_message): #If the conversation should end and the user is explicitly saying goodbye
        retrieved_chunks: list[dict] = [] #Then we set the retrieved_chunks to an empty list. 
        content_plan = (
            "- Warmly thank the user for the conversation and wish them well in Luka Dončić's relaxed voice."
        ) #content plan is a string that shows what Lukas reply is supposed to be before we stylize it to his voice
        reply = "It was great talking with you, man."
        declined = False
    elif out_of_scope(cleaned_message, history):  #This method was imported from guardrails.py file. It checks if the user message is out of scope or not.
        retrieved_chunks = [] #Then we set the retrieved_chunks to an empty list. 
        content_plan = "- Decline out-of-scope or post-cutoff question in Luka Dončić's voice."
        reply = decline(cleaned_message) #This method was imported from guardrails.py file. It declines the user message in Luka's voice.
        declined = True #Then we set the declined to True.
    else:
        step = content_step(memory=memory, user_message=cleaned_message, history=history) #contet step method was imported from content_module.py file where it retrives
        #memory chunks and asks the LLM to create a content plan (turn the chunks into a response)
        retrieved_chunks = step.get("retrieved_chunks", []) #Python syntax for get a list otherwise return an empty list.
        content_plan = step.get("content_plan", "")
        reply = stylize(content_plan) #Stylize then styles the response in Luka's voice.
        declined = False #Then we set the declined to False.

    if ending:
        reply = add_signoff(reply)
        conversation.ended = True

    conversation.turns.append(  #After the message was generated, we append it to our conversaiton list 
        {
            "role": "user",
            "content": cleaned_message,
        }
    )
    conversation.turns.append(
        {
            "role": "assistant",
            "content": reply,
            "retrieved_chunks": retrieved_chunks,
            "content_plan": content_plan,
            "declined": declined,
        }
    )
    save_conversation(conversation) #Finally we save the conversation and return the reply to the user.
    return reply
