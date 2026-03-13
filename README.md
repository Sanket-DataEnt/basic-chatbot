# My Chat - Your First AI Application

Welcome! This project is designed for **beginners who want to learn about Generative AI** by building a simple chat application that runs locally on your computer.

## What You'll Learn

- How Large Language Models (LLMs) process conversations
- How to send prompts to an AI and receive responses
- How chat history provides context for conversations
- Basic patterns for building AI-powered applications

## Quick Start (TL;DR)

```bash
# 1. Install Ollama from https://ollama.ai/download

# 2. Start Ollama (in a terminal, keep it running)
ollama serve

# 3. Download a model (in a new terminal, ~2GB)
ollama pull llama3.2

# 4. Clone/download this project, then navigate to it
cd basic-chatbot

# 5. Install Python dependencies
pip install -e .
# Or if using uv: uv sync

# 6. Run the chat app
python chat.py
# Or if using uv: uv run python chat.py
```

---

## Detailed Setup Guide

### Step 1: Install Python (3.12 or newer)

Check if Python is installed:
```bash
python --version
```

If not installed or version is below 3.12:
- **macOS:** `brew install python`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)
- **Linux:** `sudo apt install python3` (Ubuntu/Debian) or `sudo dnf install python3` (Fedora)

### Step 2: Install Ollama

Ollama lets you run AI models locally on your computer (no cloud, no API costs).

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### Step 3: Start the Ollama Server

Open a terminal and run:
```bash
ollama serve
```

Keep this terminal open - it needs to stay running!

### Step 4: Download an AI Model

In a **new** terminal, download the Llama 3.2 model (~2GB):
```bash
ollama pull llama3.2
```

This only needs to be done once.

> **Note:** You can also use Mistral (`ollama pull mistral`, ~4GB). Just update `MODEL_NAME` in `chat.py`.

### Step 5: Get This Project

If you have Git:
```bash
git clone <repository-url>
cd basic-chatbot
```

Or download and extract the ZIP file, then navigate to the folder.

### Step 6: Install Python Dependencies

**Option A - Using pip:**
```bash
pip install -e .
```

**Option B - Using uv (faster):**

First install uv if you don't have it:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install dependencies:
```bash
uv sync
```

### Step 7: Run the Chat App

**If you used pip:**
```bash
python chat.py
```

**If you used uv:**
```bash
uv run python chat.py
```

Or activate the virtual environment first:
```bash
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
python chat.py
```

You should see a welcome message, and then you can start chatting!

### Commands

| Command | What it does |
|---------|--------------|
| Type any question | Chat with the AI |
| `clear` | Reset conversation history |
| `quit` or `exit` | Exit the application |
| `Ctrl+C` | Force quit |

## Example Conversation

```
You: What is Python?

Mistral: Python is a high-level, interpreted programming language known
for its clear syntax and readability. It was created by Guido van Rossum
and first released in 1991...

You: Can you show me a simple example?

Mistral: Of course! Here's a simple Python program that prints "Hello, World!":

    print("Hello, World!")

You can save this in a file called hello.py and run it with: python hello.py
```

Notice how Mistral remembered the context of your first question when answering the second one!

## Understanding the Code

### Key Concepts

#### 1. Messages and Roles

Every message has a **role** (who's speaking) and **content** (what they said):

```python
{"role": "user", "content": "What is Python?"}      # You
{"role": "assistant", "content": "Python is..."}     # The AI
```

#### 2. Conversation History

We send ALL previous messages to the AI, not just the latest one. This is how the AI "remembers" the conversation:

```python
messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "Show me an example"},  # The AI knows this relates to Python!
]
```

#### 3. The Chat Loop

```
User types message
       |
       v
Add to history --> Send to AI --> Get response --> Display --> Add response to history
       ^                                                              |
       |______________________________________________________________|
```

## Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`

### "Model not found" error
- Download the model: `ollama pull llama3.2` (or `ollama pull mistral` if using Mistral)

### Slow responses
- First response may be slow as the model loads into memory
- Subsequent responses should be faster

### Out of memory
- Close other applications
- Try a smaller quantized model: `ollama pull llama3.2:1b` (1B parameter version)

## Next Steps

Once you're comfortable with this app, try:

1. **Try streaming** - Run `python chat_streaming.py` to see responses appear word-by-word (like ChatGPT!)
2. **Modify the system prompt** - Add a system message to give the AI a personality
3. **Experiment with other models** - Try `ollama pull llama3` and change `MODEL_NAME`
4. **Build a web interface** - Use Flask or Streamlit to create a browser-based chat

## Project Structure

```
basic-chatbot/
├── pyproject.toml       # Project dependencies
├── chat.py              # Main chat application (start here!)
├── chat_streaming.py    # Streaming version - see words appear one by one!
├── tools_demo.py        # Learn how AI can use tools (calculator example)
├── memory_basics.py     # Beginner-friendly memory introduction
├── memory_demo.py       # Advanced memory types demo
├── langgraph_memory.py  # LangGraph agent with memory (recommended!)
├── ASSIGNMENTS.md       # Practice assignments for Tools & Memory
└── README.md            # This file
```

## Tools Demo

The `tools_demo.py` script demonstrates how to give an AI the ability to use **tools** (functions).

```bash
uv run python tools_demo.py
```

**What are Tools?**
- Normally, an LLM can only generate text
- With tools, you define Python functions the AI can "call"
- Example: A calculator tool lets the AI perform accurate math

**How it works:**
1. You define a function with `@tool` decorator
2. You "bind" the tool to the model
3. When needed, the AI generates a "tool call" request
4. Your code executes the tool and returns the result
5. The AI uses the result in its final answer

This uses **LangChain** and **LangGraph** - popular frameworks for building AI applications.

## Memory Demos

### Start Here: Memory Basics

The `memory_basics.py` script is a **beginner-friendly introduction** to memory:

```bash
uv run python memory_basics.py
```

**What you'll learn:**
- Part 1: See AI WITHOUT memory (the problem)
- Part 2: See AI WITH memory (the solution)
- Part 3: Why different memory types exist
- Part 4: Build your own memory (hands-on!)

**The key insight:** AI doesn't "remember" - WE remember FOR it by sending all previous messages!

### Advanced: Memory Types Demo

After understanding the basics, try `memory_demo.py` for a deeper dive:

```bash
uv run python memory_demo.py
```

**Memory Types Demonstrated:**

| Type | Description | Best For |
|------|-------------|----------|
| **Buffer Memory** | Stores ALL messages | Short conversations, exact recall |
| **Window Memory** | Keeps last N messages only | Long chats, fixed token budget |
| **Summary Memory** | Summarizes old messages | Long conversations with context |
| **Entity Memory** | Remembers facts about people/things | Personal assistants, CRM apps |

## LangGraph Memory (Recommended!)

The `langgraph_memory.py` script shows how to build a **real agent** with memory using LangGraph:

```bash
uv run python langgraph_memory.py
```

**Why LangGraph?**
- Industry-standard framework for building AI agents
- Built-in memory management with checkpointers
- Supports complex flows with multiple nodes

**Key Concepts Demonstrated:**

| Concept | What It Means |
|---------|---------------|
| **Graph** | A flowchart - nodes connected by edges |
| **State** | The "memory" that flows through the graph |
| **Node** | A function that does something (like calling AI) |
| **Checkpointer** | Saves state between calls (enables memory!) |
| **Thread ID** | Conversation ID - different threads = separate conversations |

**The demo includes:**
1. Agent WITHOUT memory (see the problem)
2. Agent WITH memory (see the solution)
3. Multiple threads (separate conversations)
4. Interactive chat mode

## Resources

- [Ollama Documentation](https://ollama.ai/)
- [Mistral AI](https://mistral.ai/)
- [Python Ollama Library](https://github.com/ollama/ollama-python)

Happy learning!
