# Assignments: Tools & Memory Concepts

Complete these assignments to practice what you learned. Bring your solutions to the next class!

---

## Part 1: Tools Assignments

### Assignment T1: Add a New Tool (Easy)

**Goal:** Add a `temperature_converter` tool to `tools_demo.py`

**Requirements:**
- Create a tool that converts temperature between Celsius and Fahrenheit
- The tool should accept: `value` (number) and `convert_to` ("celsius" or "fahrenheit")
- Add it to the tools list so the AI can use it

**Example Usage:**
```
You: Convert 100 degrees Fahrenheit to Celsius
AI: [Uses temperature_converter tool]
AI: 100°F is 37.78°C
```

**Hints:**
- Formula: `C = (F - 32) * 5/9` and `F = C * 9/5 + 32`
- Look at how `calculator` tool is defined
- Don't forget to add it to the `tools` list!

**Deliverable:** Modified `tools_demo.py` with working temperature converter

---

### Assignment T2: Create a String Tool (Easy)

**Goal:** Create a `string_helper` tool

**Requirements:**
- Create a tool with operations: "uppercase", "lowercase", "reverse", "length"
- It should take a `text` parameter and an `operation` parameter

**Example Usage:**
```
You: Reverse the word "hello"
AI: [Uses string_helper tool]
AI: The reversed word is "olleh"
```

**Deliverable:** Show the tool working with at least 2 different operations

---

### Assignment T3: Weather Lookup Tool (Medium)

**Goal:** Create a fake weather lookup tool

**Requirements:**
- Create a `get_weather` tool that takes a `city` parameter
- Since we don't have a real API, return fake/hardcoded weather data
- Include: temperature, condition (sunny/rainy/cloudy), humidity

**Example:**
```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    # Fake weather data
    weather_data = {
        "new york": {"temp": 72, "condition": "sunny", "humidity": 45},
        "london": {"temp": 58, "condition": "rainy", "humidity": 80},
        # Add more cities...
    }
    # Your code here...
```

**Example Usage:**
```
You: What's the weather in London?
AI: [Uses get_weather tool]
AI: London is currently rainy, 58°F with 80% humidity.
```

**Deliverable:** Working weather tool with at least 5 cities

---

### Assignment T4: Multi-Tool Agent (Hard)

**Goal:** Create a new file `my_tools_agent.py` with multiple tools

**Requirements:**
Create an agent with these 3 tools:
1. `calculator` - math operations
2. `get_current_date` - returns today's date
3. `unit_converter` - converts units (km to miles, kg to pounds, etc.)

**Test Questions:**
- "What is 15 miles in kilometers?"
- "What's today's date and what is 365 * 24?"
- "Convert 100 kg to pounds and add 50"

**Bonus:** Add a 4th tool of your choice!

**Deliverable:** New file with all tools working

---

## Part 2: Memory Assignments

### Assignment M1: Conversation Counter (Easy)

**Goal:** Modify `memory_basics.py` to count messages

**Requirements:**
- After each message, show: "This is message #X in our conversation"
- The count should persist across the conversation
- When you type 'forget', the count should reset to 0

**Example:**
```
You: Hello!
AI: Hi there!
[This is message #2 in our conversation]

You: How are you?
AI: I'm doing well!
[This is message #4 in our conversation]
```

**Deliverable:** Modified script showing message counts

---

### Assignment M2: Conversation Summary (Medium)

**Goal:** Add a 'summary' command to `langgraph_memory.py`

**Requirements:**
- When user types 'summary', show a summary of the conversation
- Include: number of messages, topics discussed (you can keep it simple)

**Example:**
```
You: My name is John
AI: Nice to meet you, John!

You: I work as a teacher
AI: Teaching is a great profession!

You: summary
[Conversation Summary]
- Messages: 4
- You mentioned: your name (John), your job (teacher)
```

**Hint:** You can access all messages from the state

**Deliverable:** Working 'summary' command

---

### Assignment M3: User Profile Memory (Medium)

**Goal:** Create a simple user profile that persists

**Requirements:**
- Create `user_profile.py` that remembers user facts
- Store: name, age, location, interests
- Commands: 'profile' (show saved info), 'clear profile' (reset)

**Example:**
```
You: My name is Sarah and I'm 25 years old
AI: Nice to meet you Sarah! I've noted that you're 25.

You: I live in Chicago and love painting
AI: Chicago is great! I'll remember you enjoy painting.

You: profile
[Your Profile]
- Name: Sarah
- Age: 25
- Location: Chicago
- Interests: painting
```

**Hint:** Use a dictionary to store the profile data alongside messages

**Deliverable:** New file with working profile feature

---

### Assignment M4: Multi-User Chat System (Hard)

**Goal:** Create a system that handles multiple users with separate memories

**Requirements:**
- Create `multi_user_chat.py`
- User can switch between users: 'switch user Alice', 'switch user Bob'
- Each user has their own conversation history
- Command 'users' shows all active users

**Example:**
```
[Current user: Alice]
You: I love pizza

[Switching to Bob]
You: switch user Bob
[Current user: Bob]
You: I love sushi

You: switch user Alice
[Current user: Alice]
You: What do I love?
AI: You mentioned that you love pizza!
```

**Hint:** Use different `thread_id` for each user in LangGraph

**Deliverable:** Working multi-user system

---

### Assignment M5: Persistent Memory (Hard - Bonus)

**Goal:** Make memory persist even after restarting the program

**Requirements:**
- Save conversation history to a JSON file
- When program starts, load previous conversations
- User can type 'save' to manually save, 'load' to load

**Hint:** Use Python's `json` module to save/load

**Deliverable:** Memory that survives program restart

---

## Submission Guidelines

### What to Bring to Next Class

1. **Your code files** - All modified/new Python files
2. **Screenshots** - Show your tools/memory working
3. **Challenges faced** - Note any difficulties you encountered
4. **Questions** - Write down anything you didn't understand

### Grading Criteria

| Criteria | Points |
|----------|--------|
| Code runs without errors | 30% |
| Requirements met | 40% |
| Code is readable/commented | 20% |
| Bonus features added | 10% |

### Minimum Requirements

- Complete at least **2 Tool assignments** (T1-T4)
- Complete at least **2 Memory assignments** (M1-M5)
- Be ready to explain your code!

---

## Starter Templates

### Tool Template

```python
from langchain_core.tools import tool

@tool
def my_new_tool(param1: str, param2: int) -> str:
    """
    Description of what this tool does.

    Args:
        param1: What this parameter is for
        param2: What this parameter is for

    Returns:
        What the tool returns
    """
    # Your logic here
    result = f"Processed {param1} with {param2}"
    return result
```

### Memory Template (Simple)

```python
# Simple memory - just a list!
memory = []

def add_to_memory(role: str, content: str):
    memory.append({"role": role, "content": content})

def get_memory():
    return memory

def clear_memory():
    global memory
    memory = []
```

### LangGraph Thread Template

```python
# Different thread_id = different conversation
config_user1 = {"configurable": {"thread_id": "user-1"}}
config_user2 = {"configurable": {"thread_id": "user-2"}}

# Use with agent.invoke()
result = agent.invoke({"messages": [...]}, config=config_user1)
```

---

## Need Help?

If you get stuck:

1. **Re-read the demo files** - They have lots of comments!
2. **Check error messages** - They often tell you what's wrong
3. **Start simple** - Get basic version working, then add features
4. **Ask in class** - Bring your questions!

Good luck! See you in the next class!
