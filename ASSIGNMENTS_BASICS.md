# Assignments: Chat Basics

These assignments are for beginners learning `chat.py` and `chat_streaming.py`.
Complete these to practice what you learned in the basic class!

---

## Assignment B1: Personalize Your Chatbot (Easy)

**Goal:** Make the chatbot your own!

**Requirements:**
1. Open `chat.py`
2. Change the chatbot name from "Mistral" to your own name (e.g., "Jarvis", "Max", "Luna")
3. Update the welcome message
4. Change the prompt from "You:" to something else (e.g., "Human:", ">>", your name)

**What to modify:**
- `print_header()` function
- The welcome message in `main()`
- The `input("You: ")` line

**Example Output:**
```
============================================================
   LUNA - Your Friendly AI Companion!
============================================================

Hello! I'm Luna, your AI friend. Ask me anything!

Human>> What is Python?
Luna: Python is a programming language...
```

**Deliverable:** Screenshot of your personalized chatbot running

---

## Assignment B2: Add a Help Command (Easy)

**Goal:** Add a 'help' command that shows available commands

**Requirements:**
1. When user types 'help', show a list of all commands
2. Include: help, clear, quit/exit
3. Add a brief description for each command

**Hint:** Add this in the "Handle Special Commands" section:
```python
if user_input.lower() == "help":
    print("\nAvailable Commands:")
    print("  help  - Show this help message")
    print("  clear - Start a new conversation")
    print("  quit  - Exit the chatbot")
    print()
    continue
```

**Deliverable:** Screenshot showing the help command working

---

## Assignment B3: Add a System Prompt (Medium)

**Goal:** Give your AI a personality using a system prompt!

**What is a System Prompt?**
A system prompt is a hidden instruction that tells the AI how to behave.
For example: "You are a pirate. Speak like a pirate in all responses."

**Requirements:**
1. Add a system message at the start of the conversation
2. Choose a personality (examples below)
3. Test that the AI follows the personality

**Personality Ideas:**
- Pirate: "You are a friendly pirate. Use pirate speak like 'Ahoy!' and 'Arrr!'"
- Teacher: "You are a patient teacher. Explain things simply with examples."
- Chef: "You are a professional chef. Relate everything to cooking."
- Poet: "You are a poet. Respond in rhymes when possible."

**Code to Add:**
```python
from langchain_core.messages import SystemMessage

# Add at the start of main(), before the while loop:
system_prompt = "You are a friendly pirate. Use pirate language!"
messages.append({"role": "system", "content": system_prompt})
```

**Example Output:**
```
You: What is Python?
AI: Ahoy, matey! Python be a mighty fine programming language,
    it be! As smooth as sailin' the seven seas, arrr!
```

**Deliverable:** Working chatbot with a creative personality

---

## Assignment B4: Message Counter (Easy)

**Goal:** Show how many messages have been exchanged

**Requirements:**
1. Count total messages in the conversation
2. Display the count after each AI response
3. Reset count when user types 'clear'

**Example Output:**
```
You: Hello!
AI: Hi there! How can I help you?
[Messages in conversation: 2]

You: What is 2+2?
AI: 2+2 equals 4.
[Messages in conversation: 4]
```

**Hint:** The messages list already tracks this! Use `len(messages)`

**Deliverable:** Screenshot showing message counter

---

## Assignment B5: Try Different Models (Easy)

**Goal:** Test different AI models with Ollama

**Requirements:**
1. Pull a different model: `ollama pull llama3.2:1b` (smaller, faster)
2. Change `MODEL_NAME` in chat.py
3. Chat with both models and compare responses
4. Write down differences you notice

**Models to Try:**
```bash
ollama pull llama3.2:1b    # Smaller, faster
ollama pull llama3.2       # Default (already have)
ollama pull gemma2:2b      # Google's model
```

**Deliverable:**
- Screenshot of chat with different model
- Written comparison: Which model did you prefer? Why?

---

## Assignment B6: Conversation Starter (Medium)

**Goal:** Add random conversation starters

**Requirements:**
1. When the chat starts, suggest 3 random things the user could ask
2. User can type a number (1, 2, or 3) to select a starter
3. Or they can type their own question

**Example Output:**
```
Hello! I'm your AI assistant. Here are some things you can ask:

  1. "Tell me a fun fact"
  2. "Explain how computers work"
  3. "Help me write a poem"

Type 1, 2, 3 to select, or ask your own question!

You: 1
AI: Here's a fun fact: Honey never spoils! Archaeologists have found...
```

**Hint:**
```python
import random

starters = [
    "Tell me a fun fact",
    "Explain how computers work",
    "Help me write a poem",
    "What's the weather like on Mars?",
    "Teach me a new word",
]

# Show 3 random starters
selected = random.sample(starters, 3)
for i, starter in enumerate(selected, 1):
    print(f"  {i}. \"{starter}\"")
```

**Deliverable:** Working conversation starter feature

---

## Assignment B7: Save Conversation to File (Medium)

**Goal:** Save your conversation to a text file

**Requirements:**
1. Add a 'save' command
2. When user types 'save', write the conversation to a file
3. File should be readable (not code, just the chat)

**Example Saved File (conversation.txt):**
```
=== Conversation saved on 2024-01-15 ===

You: What is Python?
AI: Python is a programming language known for its simplicity...

You: Give me an example
AI: Here's a simple Python example: print("Hello, World!")
```

**Hint:**
```python
from datetime import datetime

if user_input.lower() == "save":
    with open("conversation.txt", "w") as f:
        f.write(f"=== Conversation saved on {datetime.now().strftime('%Y-%m-%d')} ===\n\n")
        for msg in messages:
            role = "You" if msg["role"] == "user" else "AI"
            f.write(f"{role}: {msg['content']}\n\n")
    print("[Conversation saved to conversation.txt]")
    continue
```

**Deliverable:** The saved conversation.txt file

---

## Assignment B8: Response Time Tracker (Medium)

**Goal:** Show how long the AI takes to respond

**Requirements:**
1. Measure time before and after AI responds
2. Display the response time in seconds
3. Show if it was fast (< 2s), medium (2-5s), or slow (> 5s)

**Example Output:**
```
You: Explain quantum computing
AI: Quantum computing is a type of computation that...

[Response time: 3.2 seconds - Medium]
```

**Hint:**
```python
import time

# Before calling the AI
start_time = time.time()

# Call AI here...
response = send_message_to_mistral(messages)

# After response
end_time = time.time()
duration = end_time - start_time

# Categorize speed
if duration < 2:
    speed = "Fast"
elif duration < 5:
    speed = "Medium"
else:
    speed = "Slow"

print(f"[Response time: {duration:.1f} seconds - {speed}]")
```

**Deliverable:** Screenshot showing response times

---

## Assignment B9: Compare Streaming vs Non-Streaming (Easy)

**Goal:** Understand the difference between streaming and non-streaming

**Requirements:**
1. Run `chat.py` and ask a long question
2. Run `chat_streaming.py` and ask the same question
3. Note the difference in how responses appear
4. Write a short paragraph explaining:
   - What is streaming?
   - When would you prefer streaming?
   - When would you prefer non-streaming?

**Test Question:** "Explain the history of computers in detail"

**Deliverable:** Written comparison (3-5 sentences)

---

## Assignment B10: Create Your Own Chatbot Theme (Hard)

**Goal:** Create a completely themed chatbot experience

**Requirements:**
Combine multiple assignments to create a themed chatbot:
1. Custom name and welcome message (B1)
2. System prompt with personality (B3)
3. Help command (B2)
4. At least one more feature of your choice

**Theme Ideas:**
- **Space Explorer Bot:** Space-themed, talks about planets, uses space terminology
- **Cooking Assistant:** Chef personality, suggests recipes, uses cooking terms
- **Study Buddy:** Teacher personality, explains things simply, gives quizzes
- **Fitness Coach:** Motivational, gives exercise tips, tracks goals

**Deliverable:**
- Your themed `my_chatbot.py` file
- Screenshot of it running
- Brief description of your theme

---

## Submission Checklist

Complete at least **5 assignments** from B1-B10:

| # | Assignment | Difficulty | Done? |
|---|------------|------------|-------|
| B1 | Personalize Chatbot | Easy | [ ] |
| B2 | Help Command | Easy | [ ] |
| B3 | System Prompt | Medium | [ ] |
| B4 | Message Counter | Easy | [ ] |
| B5 | Try Different Models | Easy | [ ] |
| B6 | Conversation Starter | Medium | [ ] |
| B7 | Save to File | Medium | [ ] |
| B8 | Response Time | Medium | [ ] |
| B9 | Streaming Comparison | Easy | [ ] |
| B10 | Complete Theme | Hard | [ ] |

**Bring to next class:**
- Your modified Python files
- Screenshots of working features
- Any questions you have!

---

## Need Help?

1. **Read the comments** in `chat.py` - they explain everything!
2. **Start simple** - get it working, then improve
3. **Test often** - run your code after each small change
4. **Ask questions** - bring them to the next class!

Good luck and have fun!