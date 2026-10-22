"""Which change-verb matches in a task actually ask for a change (rules only, no model).

routing.change_verbs, question_pipeline.is_question and clarify.classify each search a task for
change verbs. A bare word match misroutes two kinds of question (2026-10-18):

- "What port does the server use?": "port" is a noun there, not "port X to Go".
- "Why is add wrong?", "what does add return?": a question about a function named add, update,
  remove... is a question, while "add a function ..." or "can you add tests?" is a change.

So in a question sentence (one that starts with a question word or ends in "?") a verb counts
only in request position: at the sentence start, after a request lead ("please", "can you",
"I want you to", "need to", "how do I"), or followed by an object ("a", "the", "support", "tests").
Preceded by a determiner ("the update", "which port") it is a noun. "port" is held to this in
every sentence, since as a noun it is common in statements too ("the port is 8080"); every other
verb in a statement counts as before. Callers blank code spans first (routing.strip_code), so a
backticked `add()` never counts.
"""

from __future__ import annotations

import re

# a sentence ends at . ! ? followed by space, or a line break (a dot inside a path does not, nor
# "e.g." / "i.e." / "etc." / "vs.")
_SENTENCE_END = re.compile(
    r"(?<=[.!?])(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\betc\.)(?<!\bvs\.)\s+|\n", re.IGNORECASE)
# at the real start (after a bullet or quote), not "..., which keeps"
_QUESTION_START = re.compile(
    r"^[\s\"'“(*>#-]*(?:\d+[.)]\s+)?(?:what|which|how|where|why|who|whose|does|do(?!\s+not\b)"
    r"|did|is|are|was|were|can|could|would|should|will|has|have"
    r"|when\s+(?:do|does|did|is|are|was|were|will|should|can|would))\b", re.IGNORECASE)
# what may come right before a verb that is being requested
_LEAD = re.compile(
    r"(?:^|\b(?:please|kindly|to|(?:can|could|would|will)\s+(?:you|we|someone|vega)"
    r"(?:\s+please)?|help\s+(?:me|us)|let's|let\s+us|go\s+ahead\s+and"
    r"|(?:how|where|when)\s+(?:do|can|should|would|could)\s+(?:i|we)"
    r"|should\s+(?:i|we)|(?:i|we)\s+(?:want|need|would\s+like|'d\s+like)))\W*$",
    re.IGNORECASE)
# an object after the verb: "add a ...", "fix the ...", "add support", "write tests"
_OBJECT = re.compile(
    r"^\s+(?:a|an|the|some|this|that|these|those|my|our|its|their|your|his|her|new|more|another"
    r"|it|them|one|all|each|every|any|support|tests?|unit\s+tests?|docs?|docstrings?|type\s+hints?"
    r"|logging|comments?|error\s+handling|validation|retries|caching|[\w./-]+\.\w{1,8})\b",
    re.IGNORECASE)
# a verb or number right after it makes it a noun: "add returns", "PORT must be", "port 80"
_NOUN_AFTER = re.compile(
    r"^\s+(?:is|are|was|were|must|should|does|do|did|will|can|could|would|has|had|stays|returns?"
    r"|numbers?|\d)\b", re.IGNORECASE)
# a determiner or question word right before it makes it a noun: "the update", "which port"
_NOUN_BEFORE = re.compile(
    r"\b(?:the|a|an|what|which|whose|this|that|its|their|our|my|your|default|same|server|tcp|udp"
    r"|http|https|serial|listening|local|remote|open|free|\w+'s)\W*$", re.IGNORECASE)


def _requested(sentence: str, start: int, end: int, word: str) -> bool:
    question = sentence.rstrip().endswith("?") or bool(_QUESTION_START.match(sentence))
    if not question and word != "port":
        return True
    before, after = sentence[:start], sentence[end:]
    if _NOUN_BEFORE.search(before) or _NOUN_AFTER.match(after):
        return False
    return bool(_LEAD.search(before) or _OBJECT.match(after))


def requested(pattern: re.Pattern[str], text: str) -> list[str]:
    """The pattern's matches in `text` (code already blanked) that ask for a change, lowercased,
    in order."""
    found = []
    for sentence in _SENTENCE_END.split(text):
        for match in pattern.finditer(sentence):
            word = match.group(0).lower()
            if _requested(sentence, match.start(), match.end(), word):
                found.append(word)
    return found
