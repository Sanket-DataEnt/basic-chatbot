"""
Memory Concepts Demo with LangChain and Ollama

This script demonstrates different types of MEMORY in AI applications.
Memory is what allows an AI to "remember" things across a conversation
or even across multiple sessions.

=== WHY DOES AI NEED MEMORY? ===
Without memory, every message to an AI is like talking to someone with
amnesia - they forget everything you just said! Memory solves this.

=== TYPES OF MEMORY WE'LL EXPLORE ===

1. CONVERSATION BUFFER MEMORY (Short-term)
   - Stores all messages in the conversation
   - Simple but uses lots of tokens for long conversations
   - Like remembering every word of a conversation

2. CONVERSATION SUMMARY MEMORY
   - Summarizes old messages to save space
   - Keeps recent messages + summary of older ones
   - Like remembering the "gist" of what was discussed

3. WINDOW MEMORY
   - Only keeps the last N messages
   - Simple and predictable token usage
   - Like only remembering the last few things said

4. ENTITY MEMORY
   - Extracts and remembers facts about specific entities (people, places)
   - "User's name is John, lives in NYC, likes pizza"
   - Like keeping notes about people you meet

=== PREREQUISITES ===
1. Ollama running: ollama serve
2. Model downloaded: ollama pull llama3.2
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dataclasses import dataclass, field
import json

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "llama3.2"


def get_model():
    """Create and return the Ollama model."""
    return ChatOllama(model=MODEL_NAME, temperature=0.7)


# =============================================================================
# MEMORY TYPE 1: CONVERSATION BUFFER MEMORY
# =============================================================================
# This is the simplest form of memory - just keep ALL messages!

@dataclass
class ConversationBufferMemory:
    """
    CONVERSATION BUFFER MEMORY

    === HOW IT WORKS ===
    - Stores every message (human and AI) in a list
    - When calling the AI, we send ALL previous messages
    - The AI sees the complete conversation history

    === PROS ===
    - Simple to implement
    - AI has full context of everything said
    - Perfect for short conversations

    === CONS ===
    - Uses more tokens as conversation grows
    - Can hit context window limits on long conversations
    - More expensive (if using paid APIs)

    === BEST FOR ===
    - Short conversations (< 10-20 exchanges)
    - When you need exact recall of everything said
    """
    messages: list = field(default_factory=list)

    def add_user_message(self, content: str):
        """Add a message from the user."""
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        """Add a message from the AI."""
        self.messages.append(AIMessage(content=content))

    def get_messages(self) -> list:
        """Get all messages to send to the AI."""
        return self.messages.copy()

    def clear(self):
        """Clear all memory."""
        self.messages = []

    def get_token_estimate(self) -> int:
        """Rough estimate of tokens used (4 chars ~ 1 token)."""
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4


def demo_buffer_memory():
    """Demonstrate Conversation Buffer Memory."""
    print("\n" + "="*70)
    print(" DEMO 1: CONVERSATION BUFFER MEMORY")
    print("="*70)
    print("""
    This memory keeps ALL messages in the conversation.
    Watch how the AI remembers everything we discussed!
    """)

    model = get_model()
    memory = ConversationBufferMemory()

    # Simulate a conversation
    conversation = [
        "Hi! My name is Alice and I'm a software engineer.",
        "I've been working on a Python project about machine learning.",
        "What's my name and what am I working on?",  # Test if AI remembers!
    ]

    system_msg = SystemMessage(content="You are a helpful assistant. Be concise.")

    for user_input in conversation:
        print(f"\nYou: {user_input}")
        memory.add_user_message(user_input)

        # Send ALL messages to the AI
        messages = [system_msg] + memory.get_messages()
        response = model.invoke(messages)

        print(f"AI: {response.content}")
        memory.add_ai_message(response.content)

        print(f"   [Memory: {len(memory.messages)} messages, ~{memory.get_token_estimate()} tokens]")

    print("\n--- Memory Contents ---")
    for i, msg in enumerate(memory.messages):
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  {i+1}. [{role}]: {preview}")


# =============================================================================
# MEMORY TYPE 2: WINDOW MEMORY
# =============================================================================
# Only keep the last N messages - simple and predictable!

@dataclass
class WindowMemory:
    """
    WINDOW MEMORY (Last K Messages)

    === HOW IT WORKS ===
    - Only keeps the last K messages
    - Older messages are automatically removed
    - Like a "sliding window" over the conversation

    === PROS ===
    - Predictable token usage
    - Never hits context limits
    - Simple to implement

    === CONS ===
    - Loses old context completely
    - AI forgets things from earlier in conversation
    - May seem "forgetful" to users

    === BEST FOR ===
    - Long conversations where recent context matters most
    - Applications with strict token budgets
    - Simple chatbots without complex recall needs
    """
    window_size: int = 6  # Keep last 6 messages (3 exchanges)
    messages: list = field(default_factory=list)

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))
        self._trim()

    def _trim(self):
        """Keep only the last window_size messages."""
        if len(self.messages) > self.window_size:
            # Remove oldest messages
            self.messages = self.messages[-self.window_size:]

    def get_messages(self) -> list:
        return self.messages.copy()

    def clear(self):
        self.messages = []


def demo_window_memory():
    """Demonstrate Window Memory."""
    print("\n" + "="*70)
    print(" DEMO 2: WINDOW MEMORY (Last K Messages)")
    print("="*70)
    print("""
    This memory only keeps the LAST 4 messages (2 exchanges).
    Watch how the AI "forgets" older information!
    """)

    model = get_model()
    memory = WindowMemory(window_size=4)  # Only keep last 4 messages

    conversation = [
        "My favorite color is blue.",
        "My favorite food is pizza.",
        "My favorite movie is Inception.",
        "What is my favorite color?",  # AI might forget this!
    ]

    system_msg = SystemMessage(content="You are a helpful assistant. Be concise.")

    for user_input in conversation:
        print(f"\nYou: {user_input}")
        memory.add_user_message(user_input)

        messages = [system_msg] + memory.get_messages()
        response = model.invoke(messages)

        print(f"AI: {response.content}")
        memory.add_ai_message(response.content)

        print(f"   [Window Memory: {len(memory.messages)}/{memory.window_size} messages kept]")

    print("\n--- What's in Memory Now ---")
    print("   (Notice: oldest messages were removed!)")
    for i, msg in enumerate(memory.messages):
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"  {i+1}. [{role}]: {preview}")


# =============================================================================
# MEMORY TYPE 3: SUMMARY MEMORY
# =============================================================================
# Summarize old messages to save space while keeping context

@dataclass
class SummaryMemory:
    """
    CONVERSATION SUMMARY MEMORY

    === HOW IT WORKS ===
    - Keeps recent messages in full
    - Periodically summarizes older messages
    - Sends: [summary of old stuff] + [recent messages]

    === PROS ===
    - Retains context from long conversations
    - Uses fewer tokens than buffer memory
    - Balances recall with efficiency

    === CONS ===
    - Loses specific details in summarization
    - Requires extra LLM calls to create summaries
    - More complex to implement

    === BEST FOR ===
    - Long conversations needing historical context
    - Balancing cost with context retention
    - Applications where "gist" is enough
    """
    summary: str = ""
    recent_messages: list = field(default_factory=list)
    max_recent: int = 4  # Keep last 4 messages in full

    def add_user_message(self, content: str):
        self.recent_messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.recent_messages.append(AIMessage(content=content))

    def should_summarize(self) -> bool:
        """Check if we have too many recent messages."""
        return len(self.recent_messages) > self.max_recent

    def create_summary(self, model) -> str:
        """Use the LLM to summarize older messages."""
        if len(self.recent_messages) <= 2:
            return self.summary

        # Get messages to summarize (all but the last 2)
        to_summarize = self.recent_messages[:-2]

        # Create summary prompt
        conversation_text = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in to_summarize
        ])

        if self.summary:
            prompt = f"""Previous summary: {self.summary}

New conversation to add to summary:
{conversation_text}

Create an updated, concise summary of the key points discussed:"""
        else:
            prompt = f"""Conversation:
{conversation_text}

Create a concise summary of the key points discussed:"""

        response = model.invoke([HumanMessage(content=prompt)])
        new_summary = response.content

        # Keep only last 2 messages
        self.recent_messages = self.recent_messages[-2:]
        self.summary = new_summary

        return new_summary

    def get_messages(self) -> list:
        """Get messages to send to AI, including summary context."""
        messages = []
        if self.summary:
            messages.append(SystemMessage(
                content=f"Summary of earlier conversation: {self.summary}"
            ))
        messages.extend(self.recent_messages)
        return messages

    def clear(self):
        self.summary = ""
        self.recent_messages = []


def demo_summary_memory():
    """Demonstrate Summary Memory."""
    print("\n" + "="*70)
    print(" DEMO 3: SUMMARY MEMORY")
    print("="*70)
    print("""
    This memory SUMMARIZES older messages to save tokens.
    Watch how it compresses the conversation while keeping context!
    """)

    model = get_model()
    memory = SummaryMemory(max_recent=4)

    conversation = [
        "Hi! I'm Bob, I'm 30 years old and I live in San Francisco.",
        "I work as a data scientist at a tech startup.",
        "My hobbies include hiking, photography, and playing guitar.",
        "I'm planning a trip to Japan next month.",
        "Can you remind me about myself and my plans?",
    ]

    system_msg = SystemMessage(content="You are a helpful assistant. Be concise.")

    for i, user_input in enumerate(conversation):
        print(f"\nYou: {user_input}")
        memory.add_user_message(user_input)

        # Check if we should summarize
        if memory.should_summarize():
            print("   [Creating summary of older messages...]")
            summary = memory.create_summary(model)
            print(f"   [Summary: {summary[:80]}...]")

        messages = [system_msg] + memory.get_messages()
        response = model.invoke(messages)

        print(f"AI: {response.content}")
        memory.add_ai_message(response.content)

        print(f"   [Recent messages: {len(memory.recent_messages)}, Has summary: {bool(memory.summary)}]")


# =============================================================================
# MEMORY TYPE 4: ENTITY MEMORY
# =============================================================================
# Extract and remember facts about specific entities

@dataclass
class EntityMemory:
    """
    ENTITY MEMORY

    === HOW IT WORKS ===
    - Extracts entities (people, places, things) from conversation
    - Stores facts/attributes about each entity
    - Provides relevant entity info when mentioned again

    === PROS ===
    - Remembers specific facts efficiently
    - Can recall info about entities mentioned long ago
    - Structured storage of information

    === CONS ===
    - Requires entity extraction (extra LLM calls)
    - May miss implicit references
    - More complex implementation

    === BEST FOR ===
    - Personal assistants that track user preferences
    - Customer service bots remembering customer details
    - Any app needing to remember facts about specific things
    """
    entities: dict = field(default_factory=dict)  # entity_name -> list of facts
    recent_messages: list = field(default_factory=list)

    def add_user_message(self, content: str):
        self.recent_messages.append(HumanMessage(content=content))
        # Keep only last 6 messages
        if len(self.recent_messages) > 6:
            self.recent_messages = self.recent_messages[-6:]

    def add_ai_message(self, content: str):
        self.recent_messages.append(AIMessage(content=content))
        if len(self.recent_messages) > 6:
            self.recent_messages = self.recent_messages[-6:]

    def extract_entities(self, model, text: str):
        """Use LLM to extract entities and facts from text."""
        prompt = f"""Extract any entities (people, places, organizations) and facts about them from this text.
Return as JSON: {{"entity_name": ["fact1", "fact2"]}}
If no entities found, return: {{}}

Text: {text}

JSON:"""

        response = model.invoke([HumanMessage(content=prompt)])

        try:
            # Try to parse JSON from response
            content = response.content.strip()
            # Handle markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            extracted = json.loads(content)

            # Merge with existing entities
            for entity, facts in extracted.items():
                entity_lower = entity.lower()
                if entity_lower not in self.entities:
                    self.entities[entity_lower] = []
                for fact in facts:
                    if fact not in self.entities[entity_lower]:
                        self.entities[entity_lower].append(fact)

            return extracted
        except (json.JSONDecodeError, IndexError):
            return {}

    def get_relevant_entities(self, text: str) -> str:
        """Get entity info relevant to the current message."""
        relevant = []
        text_lower = text.lower()

        for entity, facts in self.entities.items():
            if entity in text_lower:
                relevant.append(f"{entity.title()}: {', '.join(facts)}")

        return "\n".join(relevant) if relevant else ""

    def get_messages(self, current_input: str) -> list:
        """Get messages with relevant entity context."""
        messages = []

        entity_context = self.get_relevant_entities(current_input)
        if entity_context:
            messages.append(SystemMessage(
                content=f"Known information:\n{entity_context}"
            ))

        messages.extend(self.recent_messages)
        return messages

    def clear(self):
        self.entities = {}
        self.recent_messages = []


def demo_entity_memory():
    """Demonstrate Entity Memory."""
    print("\n" + "="*70)
    print(" DEMO 4: ENTITY MEMORY")
    print("="*70)
    print("""
    This memory extracts and stores FACTS about ENTITIES (people, places).
    Watch how it builds a knowledge base as we chat!
    """)

    model = get_model()
    memory = EntityMemory()

    conversation = [
        "Sarah is my sister. She lives in Boston and works as a doctor.",
        "My friend Mike is a chef in New York. He owns a Italian restaurant.",
        "Tell me what you know about Sarah.",
        "What about Mike?",
    ]

    system_msg = SystemMessage(content="You are a helpful assistant with memory of people mentioned. Be concise.")

    for user_input in conversation:
        print(f"\nYou: {user_input}")

        # Extract entities from user input
        if any(word in user_input.lower() for word in ["is", "works", "lives", "owns"]):
            print("   [Extracting entities...]")
            extracted = memory.extract_entities(model, user_input)
            if extracted:
                print(f"   [Found: {list(extracted.keys())}]")

        memory.add_user_message(user_input)

        messages = [system_msg] + memory.get_messages(user_input)
        response = model.invoke(messages)

        print(f"AI: {response.content}")
        memory.add_ai_message(response.content)

    print("\n--- Entity Knowledge Base ---")
    for entity, facts in memory.entities.items():
        print(f"  {entity.title()}:")
        for fact in facts:
            print(f"    - {fact}")


# =============================================================================
# COMPARISON DEMO
# =============================================================================
def demo_comparison():
    """Compare different memory types side by side."""
    print("\n" + "="*70)
    print(" MEMORY TYPES COMPARISON")
    print("="*70)
    print("""
    +-----------------------+------------+------------+--------------+
    | Memory Type           | Token Use  | Recall     | Complexity   |
    +-----------------------+------------+------------+--------------+
    | Buffer (Full History) | High       | Perfect    | Low          |
    | Window (Last K)       | Fixed/Low  | Limited    | Low          |
    | Summary               | Medium     | Good       | Medium       |
    | Entity                | Low-Medium | Selective  | High         |
    +-----------------------+------------+------------+--------------+

    WHEN TO USE EACH:

    Buffer Memory:
      - Short conversations
      - When exact recall matters
      - Prototyping and simple apps

    Window Memory:
      - Long conversations
      - Real-time chat applications
      - When recent context is enough

    Summary Memory:
      - Long conversations needing history
      - Customer support with context
      - Meeting assistants

    Entity Memory:
      - Personal assistants
      - CRM-style applications
      - When tracking facts about people/things
    """)


# =============================================================================
# INTERACTIVE MODE
# =============================================================================
def interactive_mode():
    """Let user try different memory types."""
    print("\n" + "="*70)
    print(" INTERACTIVE MODE")
    print("="*70)
    print("""
    Choose a memory type to chat with:
    1. Buffer Memory (remembers everything)
    2. Window Memory (remembers last 4 messages)
    3. Summary Memory (summarizes old messages)

    Type 'switch' to change memory type
    Type 'clear' to reset memory
    Type 'quit' to exit
    """)

    model = get_model()
    memories = {
        "1": ("Buffer", ConversationBufferMemory()),
        "2": ("Window", WindowMemory(window_size=4)),
        "3": ("Summary", SummaryMemory(max_recent=4)),
    }

    current = "1"
    memory_name, memory = memories[current]
    print(f"Using: {memory_name} Memory\n")

    system_msg = SystemMessage(content="You are a helpful assistant. Be concise.")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                memory.clear()
                print("[Memory cleared!]\n")
                continue

            if user_input.lower() == "switch":
                print("\nChoose: 1=Buffer, 2=Window, 3=Summary")
                choice = input("Enter choice: ").strip()
                if choice in memories:
                    current = choice
                    memory_name, memory = memories[current]
                    memory.clear()
                    print(f"[Switched to {memory_name} Memory]\n")
                continue

            # Add message and get response
            memory.add_user_message(user_input)

            # Handle summary memory specially
            if isinstance(memory, SummaryMemory) and memory.should_summarize():
                print("   [Summarizing...]")
                memory.create_summary(model)

            messages = [system_msg] + memory.get_messages()
            response = model.invoke(messages)

            print(f"AI: {response.content}")
            memory.add_ai_message(response.content)

            # Show memory status
            if isinstance(memory, ConversationBufferMemory):
                print(f"   [{len(memory.messages)} messages, ~{memory.get_token_estimate()} tokens]\n")
            elif isinstance(memory, WindowMemory):
                print(f"   [{len(memory.messages)}/{memory.window_size} messages in window]\n")
            elif isinstance(memory, SummaryMemory):
                print(f"   [{len(memory.recent_messages)} recent, summary: {bool(memory.summary)}]\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              MEMORY CONCEPTS DEMO - How AI Remembers                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  This demo shows different ways to give an AI "memory":              ║
║                                                                      ║
║  1. Buffer Memory   - Keep ALL messages (simple but expensive)       ║
║  2. Window Memory   - Keep last N messages (simple, may forget)      ║
║  3. Summary Memory  - Summarize old messages (balanced approach)     ║
║  4. Entity Memory   - Remember facts about people/things             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    print("Choose an option:")
    print("  1. Run all demos (recommended for learning)")
    print("  2. Interactive mode (try memories yourself)")
    print("  3. Exit")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        try:
            demo_buffer_memory()
            input("\n[Press Enter to continue to next demo...]")

            demo_window_memory()
            input("\n[Press Enter to continue to next demo...]")

            demo_summary_memory()
            input("\n[Press Enter to continue to next demo...]")

            demo_entity_memory()
            input("\n[Press Enter to continue to comparison...]")

            demo_comparison()

        except KeyboardInterrupt:
            print("\n\nDemo interrupted!")
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure Ollama is running: ollama serve")

    elif choice == "2":
        interactive_mode()

    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()
