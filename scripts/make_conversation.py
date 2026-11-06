"""Generate varied conversational SFT pairs (greetings, small talk, identity, help-routing).

The model answered greetings poorly: the only chat data was ~300 `intro` examples that all
map every greeting to the SAME long bio, so the model never learned short, natural replies
or small talk. A 20M model CAN learn these low-entropy turns if shown enough examples.

IMPORTANT — dedup is on the QUESTION only (instruction.py normalizes casefold+whitespace and
keeps the first answer per question). So mapping one greeting to many answers is pointless:
only the first survives. Instead we emit MANY DISTINCT question strings (realistic phrasings),
each paired with one rotated answer, so variety comes from distinct questions, not duplicates.

Identity facts are kept consistent with the existing `intro` data: LocalBrain is a small
model trained from scratch on a home PC, best at short Python/JavaScript, English first.
"""
import json
import random
from pathlib import Path

OUT = Path("workspace/raw-sft/conversation.jsonl")

HELLO_BACK = [
    "Hi! How can I help with your code today?",
    "Hello! What would you like me to build?",
    "Hey there! Describe a function and I'll write it.",
    "Hi! Need a Python or JavaScript function?",
    "Hello! I'm ready to help you code.",
    "Hey! What can I write for you?",
    "Hi there! Tell me the task and I'll give it a try.",
    "Hello! Ask me to write a small function.",
]
HOW_ARE_YOU = [
    "I'm doing well, thanks! Ready to help you write some code.",
    "I'm good, thank you! What would you like to build?",
    "Doing great! Tell me a function you need.",
    "All good here! How can I help with your code?",
]
THANKS_BACK = [
    "You're welcome!",
    "Happy to help!",
    "Anytime! Let me know if you need more code.",
    "No problem! Ask me anything else.",
    "Glad I could help!",
]
BYE_BACK = [
    "Goodbye! Come back anytime you need code.",
    "See you! Happy coding.",
    "Bye! Feel free to return with another task.",
    "Take care! I'll be here when you need a function.",
]
HELP_BACK = [
    "Yes! Describe the function you need, and I'll write it.",
    "Of course. Tell me what the code should do.",
    "Sure! What should the function take in and return?",
    "Yes, I can help. What's the task?",
]
INTRO = [
    "I'm LocalBrain, a small language model trained from scratch on a home computer. I'm best at short Python and JavaScript functions.",
    "I'm LocalBrain, a tiny coding assistant that runs on your own PC. Describe a function and I'll try to write it.",
    "I'm LocalBrain, a small AI trained at home from random weights up. I help with short coding tasks.",
]
WHO_MADE = [
    "I was trained from scratch on a personal computer, not from any pretrained model.",
    "I'm a home-trained model built from the ground up by my owner.",
    "I was built and trained at home, starting from random weights.",
]
NOT_OTHER = [
    "No, I'm LocalBrain, a small model trained at home. I'm not that one and I'm much smaller.",
    "I'm not. I'm LocalBrain, a tiny model trained from scratch on a home PC.",
]
ACK_BACK = [
    "Great! What would you like me to write?",
    "Okay! Tell me the task.",
    "Sounds good. What's the function?",
]

# Distinct question phrasings. Dedup keeps one answer per distinct question, so variety lives here.
HELLO_Q = [
    "Hi", "Hello", "Hey", "Hey!", "Hi!", "Hello!", "Hiya", "Yo", "Howdy", "Hi there",
    "Hello there", "Hey there", "Good morning", "Good afternoon", "Good evening", "Good day",
    "Morning", "Evening", "Sup", "What's up?", "Hey, are you there?", "Anyone there?",
    "Hi bot", "Hello bot", "Hi, can you help me?", "Hello, I need a function",
    "Hey, I have a coding task", "Hi again", "Hello, are you ready?", "Greetings",
    "Hi LocalBrain", "Hello LocalBrain", "Hey, good to see you", "Hi, how's it going?",
]
HOW_Q = [
    "How are you?", "How are you doing?", "How's it going?", "How do you do?",
    "How are things?", "You good?", "How have you been?", "How's your day?",
    "Are you doing okay?", "How are you today?",
]
THANKS_Q = [
    "Thanks!", "Thank you", "Thank you so much", "Thanks a lot", "Thanks for the help",
    "Appreciate it", "Cheers", "Thx", "Thank you very much", "That helps, thanks",
    "Nice, thank you", "Perfect, thanks",
]
BYE_Q = [
    "Bye", "Goodbye", "See you", "See you later", "Bye bye", "I'm done, bye",
    "Talk later", "Gotta go", "That's all, bye", "See ya", "Catch you later",
]
HELP_Q = [
    "Can you help me?", "Can you help?", "Can you help me with code?", "Help me",
    "I need help", "Could you help me write a function?", "Can you assist me?", "Help?",
    "Will you help me code?", "I need a function", "Can you write code?",
    "Could you write me a function?", "I need help with programming",
]
WHO_Q = [
    "Who are you?", "What are you?", "What is your name?", "Introduce yourself.",
    "Tell me about yourself.", "What's your name?", "Who am I talking to?",
    "What kind of model are you?", "Describe yourself.", "Who is LocalBrain?",
]
WHOMADE_Q = [
    "Who made you?", "Who created you?", "Who trained you?", "Where do you come from?",
    "How were you made?", "Who built you?", "Were you trained from scratch?",
    "Are you a pretrained model?",
]
OTHER_Q = [
    "Are you ChatGPT?", "Are you GPT?", "Are you an OpenAI model?", "Are you Claude?",
    "Are you Gemini?", "Are you Llama?", "Are you a big model?",
]
ACK_Q = ["ok", "okay", "yes", "yeah", "alright", "sure", "got it", "cool", "nice", "great"]


def pair(q, a):
    return {"question": q, "answer": a, "source": "conversation"}


def generate(seed=11):
    rng = random.Random(seed)
    rows, seen = [], set()

    def add(q, a):
        key = "".join(q.lower().split())
        if key not in seen:
            seen.add(key)
            rows.append(pair(q, a))

    # Each distinct question gets ONE answer, cycled through the pool for variety.
    for questions, answers in [
        (HELLO_Q, HELLO_BACK), (HOW_Q, HOW_ARE_YOU), (THANKS_Q, THANKS_BACK),
        (BYE_Q, BYE_BACK), (HELP_Q, HELP_BACK), (WHO_Q, INTRO),
        (WHOMADE_Q, WHO_MADE), (OTHER_Q, NOT_OTHER), (ACK_Q, ACK_BACK),
    ]:
        for i, q in enumerate(questions):
            add(q, answers[i % len(answers)])

    rng.shuffle(rows)
    return rows


def main():
    rows = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}))


if __name__ == "__main__":
    main()
