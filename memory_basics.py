"""
Memory Basics - Understanding Why AI Needs Memory

This is a BEGINNER-FRIENDLY script that explains memory concepts
step by step. Run this before memory_demo.py!

We'll answer ONE simple question:
  "Why can't AI remember what I just said?"

And show you exactly how to fix it!
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Setup
MODEL_NAME = "llama3.2"
model = ChatOllama(model=MODEL_NAME, temperature=0.7)


def print_section(title):
    """Helper to print section headers."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# =============================================================================
# PART 1: THE PROBLEM - AI WITHOUT MEMORY
# =============================================================================

def part1_without_memory():
    """Show what happens when AI has NO memory."""
    print_section("PART 1: AI WITHOUT MEMORY (The Problem)")

    print("""
    Let's see what happens when we DON'T give the AI any memory.
    We'll send two separate messages and see if it remembers...
    """)

    input("Press Enter to start...\n")

    # First message
    print("--- Message 1 ---")
    print("You: My name is Alice.\n")

    response1 = model.invoke([
        HumanMessage(content="My name is Alice.")
    ])
    print(f"AI: {response1.content}\n")

    input("Press Enter to send the next message...\n")

    # Second message - WITHOUT the first message's context
    print("--- Message 2 (sent WITHOUT previous context) ---")
    print("You: What is my name?\n")

    response2 = model.invoke([
        HumanMessage(content="What is my name?")
    ])
    print(f"AI: {response2.content}\n")

    print("""
    ^^^ NOTICE: The AI doesn't know your name!

    WHY? Because each call to the AI is INDEPENDENT.
    The AI doesn't automatically remember previous messages.
    It's like talking to someone with amnesia!

    Let's fix this...
    """)


# =============================================================================
# PART 2: THE SOLUTION - AI WITH MEMORY
# =============================================================================

def part2_with_memory():
    """Show how memory solves the problem."""
    print_section("PART 2: AI WITH MEMORY (The Solution)")

    print("""
    Now let's give the AI MEMORY.

    The secret? Send ALL previous messages with each new request!

    Instead of just sending:
        [new message]

    We send:
        [message 1, message 2, message 3, ..., new message]

    This way, the AI can "see" the entire conversation!
    """)

    input("Press Enter to start...\n")

    # This list IS our memory - it stores the conversation
    memory = []

    # First message
    print("--- Message 1 ---")
    print("You: My name is Alice.\n")

    memory.append(HumanMessage(content="My name is Alice."))

    # Send ALL messages (currently just one)
    response1 = model.invoke(memory)
    print(f"AI: {response1.content}")

    # Save AI's response to memory too!
    memory.append(AIMessage(content=response1.content))

    print(f"\n[Memory now has {len(memory)} messages]\n")

    input("Press Enter to send the next message...\n")

    # Second message - WITH the previous context
    print("--- Message 2 (sent WITH previous context) ---")
    print("You: What is my name?\n")

    memory.append(HumanMessage(content="What is my name?"))

    # Send ALL messages (now we have 3: user, ai, user)
    response2 = model.invoke(memory)
    print(f"AI: {response2.content}")

    memory.append(AIMessage(content=response2.content))

    print(f"\n[Memory now has {len(memory)} messages]\n")

    print("""
    ^^^ SUCCESS! The AI remembered your name!

    THE KEY INSIGHT:
    Memory is just a LIST of messages that we send every time.
    The AI doesn't "remember" - WE remember FOR it!
    """)

    # Show what's in memory
    print("\n--- What's in our memory list? ---")
    for i, msg in enumerate(memory, 1):
        role = "You" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i}. [{role}]: {msg.content[:50]}...")


# =============================================================================
# PART 3: WHY DIFFERENT MEMORY TYPES EXIST
# =============================================================================

def part3_why_memory_types():
    """Explain why we need different memory strategies."""
    print_section("PART 3: WHY DO WE NEED DIFFERENT MEMORY TYPES?")

    print("""
    You might think: "Just keep ALL messages forever!"

    But there's a problem: AI has a LIMITED CONTEXT WINDOW.

    Think of it like this:
    - The AI can only "read" about 4,000-128,000 words at once
    - If your conversation is longer, old messages get cut off!
    - Also, more messages = more cost (if using paid APIs)

    So we need STRATEGIES to manage memory efficiently:

    +-----------------+----------------------------------------+
    | Strategy        | How it works                           |
    +-----------------+----------------------------------------+
    | Keep Everything | Store all messages (simple but costly) |
    | Window          | Keep only last N messages              |
    | Summary         | Summarize old messages                 |
    +-----------------+----------------------------------------+

    Let's see the difference...
    """)

    input("Press Enter to see a demo...\n")

    # Simulate a conversation
    print("Simulating a long conversation about different topics...\n")

    messages = [
        ("You", "My favorite color is blue."),
        ("AI", "Nice! Blue is a calming color."),
        ("You", "My favorite food is pizza."),
        ("AI", "Pizza is delicious! What toppings?"),
        ("You", "I like pepperoni."),
        ("AI", "Classic choice!"),
        ("You", "My favorite movie is Inception."),
        ("AI", "Great film! Christopher Nolan is amazing."),
    ]

    print("Full conversation:")
    for role, content in messages:
        print(f"  {role}: {content}")

    print("\n" + "-" * 40)
    print("\nWith WINDOW MEMORY (last 4 messages only):")
    for role, content in messages[-4:]:
        print(f"  {role}: {content}")

    print("\n  --> AI would NOT remember: favorite color, favorite food")
    print("  --> AI WOULD remember: favorite movie")

    print("\n" + "-" * 40)
    print("\nWith SUMMARY MEMORY:")
    print("  [Summary]: User likes blue, pizza with pepperoni")
    for role, content in messages[-2:]:
        print(f"  {role}: {content}")
    print("\n  --> AI remembers key facts in condensed form!")


# =============================================================================
# PART 4: HANDS-ON - BUILD YOUR OWN MEMORY
# =============================================================================

def part4_build_memory():
    """Interactive demo where user builds memory step by step."""
    print_section("PART 4: BUILD YOUR OWN MEMORY (Interactive)")

    print("""
    Now YOU try it! Chat with the AI and watch the memory grow.

    Commands:
      - Type anything to chat
      - Type 'memory' to see what's stored
      - Type 'forget' to clear memory and start fresh
      - Type 'quit' to exit
    """)

    # Our simple memory - just a list!
    memory = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower() == 'memory':
                print("\n--- Memory Contents ---")
                if not memory:
                    print("  (empty)")
                for i, msg in enumerate(memory, 1):
                    role = "You" if isinstance(msg, HumanMessage) else "AI"
                    preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                    print(f"  {i}. [{role}]: {preview}")
                print(f"\nTotal: {len(memory)} messages")
                continue

            if user_input.lower() == 'forget':
                memory = []
                print("\n[Memory cleared! AI will forget everything.]")
                continue

            # Add user message to memory
            memory.append(HumanMessage(content=user_input))

            # Send ENTIRE memory to AI
            print("\n[Sending all messages to AI...]")
            response = model.invoke(memory)

            print(f"\nAI: {response.content}")

            # Add AI response to memory
            memory.append(AIMessage(content=response.content))

            print(f"\n[Memory: {len(memory)} messages]")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         MEMORY BASICS - A Beginner's Guide                 ║
    ╠════════════════════════════════════════════════════════════╣
    ║  This script will teach you ONE thing:                     ║
    ║                                                            ║
    ║  WHY AI needs memory and HOW to give it memory             ║
    ║                                                            ║
    ║  We'll go step by step:                                    ║
    ║    Part 1: See AI WITHOUT memory (the problem)             ║
    ║    Part 2: See AI WITH memory (the solution)               ║
    ║    Part 3: Why different memory types exist                ║
    ║    Part 4: Build your own memory (hands-on!)               ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    print("Choose what to learn:")
    print("  1. Start from beginning (recommended)")
    print("  2. Jump to hands-on demo")
    print("  3. Exit")

    choice = input("\nEnter choice (1/2/3): ").strip()

    try:
        if choice == "1":
            part1_without_memory()
            input("\nPress Enter to continue to Part 2...")

            part2_with_memory()
            input("\nPress Enter to continue to Part 3...")

            part3_why_memory_types()
            input("\nPress Enter to continue to Part 4 (hands-on)...")

            part4_build_memory()

        elif choice == "2":
            part4_build_memory()

        else:
            print("Goodbye!")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    main()
