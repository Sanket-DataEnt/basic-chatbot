"""
LangGraph Memory Demo - The Simplest Possible Agent

This script shows how to build an AI agent with MEMORY using LangGraph.
We'll build the simplest possible agent - just ONE node that chats!

=== WHAT IS LANGGRAPH? ===
LangGraph lets you build AI applications as "graphs" (flowcharts):
- NODES = Boxes that do things (like "call the AI")
- EDGES = Arrows connecting the boxes
- STATE = Data that flows through the graph

=== WHAT IS A CHECKPOINTER? ===
A checkpointer SAVES the state after each step.
This is how LangGraph gives your agent MEMORY!

Without checkpointer: AI forgets everything between calls
With checkpointer: AI remembers the entire conversation

=== PREREQUISITES ===
1. Ollama running: ollama serve
2. Model downloaded: ollama pull llama3.2
"""

# =============================================================================
# IMPORTS
# =============================================================================
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict

# This special function tells LangGraph how to combine message lists
from langgraph.graph.message import add_messages


# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "llama3.2"


# =============================================================================
# STEP 1: DEFINE THE STATE
# =============================================================================
# State is like a "container" that holds data as it flows through the graph.
# Think of it as the agent's "memory" or "notepad".

class State(TypedDict):
    """
    Our agent's state - what it "remembers".

    We only need ONE thing: the list of messages in the conversation.

    The `Annotated[..., add_messages]` part tells LangGraph:
    "When updating messages, APPEND new ones to the list, don't replace."
    """
    messages: Annotated[list, add_messages]


# =============================================================================
# STEP 2: DEFINE THE NODE
# =============================================================================
# A node is a function that:
# 1. Takes the current state
# 2. Does something (like calling the AI)
# 3. Returns updates to the state

def chat_node(state: State) -> dict:
    """
    The chat node - this is where the AI magic happens!

    It's just a function that:
    1. Gets the conversation history from state
    2. Sends it to the AI
    3. Returns the AI's response

    That's it! LangGraph handles the rest.
    """
    # Create the AI model
    model = ChatOllama(model=MODEL_NAME, temperature=0.7)

    # Get messages from state and call the AI
    # The state["messages"] contains the ENTIRE conversation history!
    response = model.invoke(state["messages"])

    # Return the response - LangGraph will add it to messages automatically
    # (because we used `add_messages` in our State definition)
    return {"messages": [response]}


# =============================================================================
# STEP 3: BUILD THE GRAPH
# =============================================================================
# Now we connect the pieces into a flowchart!

def build_graph():
    """
    Build our simple chat graph.

    The graph looks like this:

        [START] --> [chat_node] --> [END]

    That's it! The simplest possible graph with just ONE node.
    """
    # Create a new graph with our State type
    graph_builder = StateGraph(State)

    # Add our chat node
    # "chat" is the name, chat_node is the function
    graph_builder.add_node("chat", chat_node)

    # Connect the pieces:
    # START --> chat --> END
    graph_builder.add_edge(START, "chat")  # Start goes to chat
    graph_builder.add_edge("chat", END)     # Chat goes to end

    return graph_builder


# =============================================================================
# STEP 4: ADD MEMORY (THE CHECKPOINTER)
# =============================================================================
# The checkpointer is what gives our agent MEMORY!
# It saves the state after each run so we can continue the conversation.

def create_agent_with_memory():
    """
    Create our agent WITH memory.

    The secret sauce: MemorySaver()

    This checkpointer saves the conversation state in memory.
    Each conversation has a "thread_id" - like a conversation ID.
    """
    # Build the graph
    graph_builder = build_graph()

    # Create the memory saver - THIS IS THE KEY!
    memory = MemorySaver()

    # Compile the graph WITH the checkpointer
    # This "activates" the graph and adds memory capability
    agent = graph_builder.compile(checkpointer=memory)

    return agent


def create_agent_without_memory():
    """
    Create our agent WITHOUT memory (for comparison).
    """
    graph_builder = build_graph()

    # Compile WITHOUT a checkpointer - no memory!
    agent = graph_builder.compile()

    return agent


# =============================================================================
# STEP 5: DEMO - SEE MEMORY IN ACTION
# =============================================================================

def demo_without_memory():
    """Show what happens WITHOUT memory."""
    print("\n" + "=" * 60)
    print("  DEMO 1: Agent WITHOUT Memory")
    print("=" * 60)
    print("""
    Let's see what happens when we DON'T use a checkpointer.
    The agent will FORGET everything between calls!
    """)

    agent = create_agent_without_memory()

    # First message
    print("--- Call 1 ---")
    print("You: My name is Alice.\n")

    result1 = agent.invoke({
        "messages": [HumanMessage(content="My name is Alice.")]
    })
    print(f"AI: {result1['messages'][-1].content}\n")

    input("Press Enter to send the next message...\n")

    # Second message - WITHOUT memory, this is a NEW conversation
    print("--- Call 2 ---")
    print("You: What is my name?\n")

    result2 = agent.invoke({
        "messages": [HumanMessage(content="What is my name?")]
    })
    print(f"AI: {result2['messages'][-1].content}\n")

    print("""
    ^^^ The AI FORGOT your name!

    WHY? Because without a checkpointer, each invoke() starts fresh.
    The agent has no memory of previous conversations.
    """)


def demo_with_memory():
    """Show what happens WITH memory."""
    print("\n" + "=" * 60)
    print("  DEMO 2: Agent WITH Memory")
    print("=" * 60)
    print("""
    Now let's use a checkpointer (MemorySaver).
    The agent will REMEMBER the conversation!

    KEY: We use the same "thread_id" for both messages.
    Think of thread_id as a conversation ID.
    """)

    agent = create_agent_with_memory()

    # The config with thread_id - THIS IS IMPORTANT!
    config = {"configurable": {"thread_id": "conversation-1"}}

    # First message
    print("--- Call 1 (thread_id: conversation-1) ---")
    print("You: My name is Alice.\n")

    result1 = agent.invoke(
        {"messages": [HumanMessage(content="My name is Alice.")]},
        config=config  # <-- Same thread_id
    )
    print(f"AI: {result1['messages'][-1].content}\n")

    input("Press Enter to send the next message...\n")

    # Second message - WITH memory, using SAME thread_id
    print("--- Call 2 (thread_id: conversation-1) ---")
    print("You: What is my name?\n")

    result2 = agent.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config  # <-- Same thread_id = same conversation!
    )
    print(f"AI: {result2['messages'][-1].content}\n")

    print("""
    ^^^ The AI REMEMBERED your name!

    WHY? Because the checkpointer saved the state after Call 1.
    When we used the same thread_id in Call 2, it loaded that state.

    THE KEY INSIGHT:
    - Same thread_id = Continue the same conversation
    - Different thread_id = Start a new conversation
    """)


def demo_multiple_threads():
    """Show how different thread_ids = different conversations."""
    print("\n" + "=" * 60)
    print("  DEMO 3: Multiple Conversations (Different Thread IDs)")
    print("=" * 60)
    print("""
    Let's prove that different thread_ids are SEPARATE conversations.
    We'll have TWO conversations at the same time!
    """)

    agent = create_agent_with_memory()

    # Two different thread configs
    alice_config = {"configurable": {"thread_id": "alice-conversation"}}
    bob_config = {"configurable": {"thread_id": "bob-conversation"}}

    # Alice introduces herself
    print("--- Alice's Conversation ---")
    print("Alice: My name is Alice and I love pizza.\n")
    agent.invoke(
        {"messages": [HumanMessage(content="My name is Alice and I love pizza.")]},
        config=alice_config
    )

    # Bob introduces himself
    print("--- Bob's Conversation ---")
    print("Bob: My name is Bob and I love sushi.\n")
    agent.invoke(
        {"messages": [HumanMessage(content="My name is Bob and I love sushi.")]},
        config=bob_config
    )

    input("Press Enter to ask each person about themselves...\n")

    # Ask Alice about herself
    print("--- Back to Alice ---")
    print("Alice: What's my name and favorite food?\n")
    result_alice = agent.invoke(
        {"messages": [HumanMessage(content="What's my name and favorite food?")]},
        config=alice_config  # <-- Alice's thread
    )
    print(f"AI (to Alice): {result_alice['messages'][-1].content}\n")

    # Ask Bob about himself
    print("--- Back to Bob ---")
    print("Bob: What's my name and favorite food?\n")
    result_bob = agent.invoke(
        {"messages": [HumanMessage(content="What's my name and favorite food?")]},
        config=bob_config  # <-- Bob's thread
    )
    print(f"AI (to Bob): {result_bob['messages'][-1].content}\n")

    print("""
    ^^^ Each conversation is SEPARATE!

    - Alice's thread remembers Alice's info
    - Bob's thread remembers Bob's info
    - They don't mix!

    This is how real apps work:
    - Each user gets their own thread_id
    - Their conversations stay private and separate
    """)


# =============================================================================
# INTERACTIVE CHAT
# =============================================================================

def interactive_chat():
    """Let the user chat with the memory-enabled agent."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE CHAT")
    print("=" * 60)
    print("""
    Now you try! Chat with the memory-enabled agent.

    Commands:
      - Type anything to chat
      - 'new' = Start a new conversation (new thread_id)
      - 'quit' = Exit
    """)

    agent = create_agent_with_memory()
    thread_num = 1
    config = {"configurable": {"thread_id": f"thread-{thread_num}"}}

    print(f"[Started conversation: thread-{thread_num}]\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower() == 'new':
                thread_num += 1
                config = {"configurable": {"thread_id": f"thread-{thread_num}"}}
                print(f"\n[Started NEW conversation: thread-{thread_num}]")
                print("[The AI will forget everything from before!]\n")
                continue

            # Send message to agent
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # Get the AI's response (last message)
            ai_response = result["messages"][-1].content
            print(f"\nAI: {ai_response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        LANGGRAPH MEMORY - Simple Agent with Memory           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  This demo shows how to give a LangGraph agent MEMORY        ║
    ║  using a checkpointer.                                       ║
    ║                                                              ║
    ║  KEY CONCEPTS:                                               ║
    ║  • Graph = Flowchart (nodes connected by edges)              ║
    ║  • State = The data flowing through (messages)               ║
    ║  • Checkpointer = Saves state (gives memory!)                ║
    ║  • Thread ID = Conversation ID (different = separate)        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print("Choose a demo:")
    print("  1. Run all demos (recommended for learning)")
    print("  2. Interactive chat only")
    print("  3. Exit")

    choice = input("\nEnter choice (1/2/3): ").strip()

    try:
        if choice == "1":
            demo_without_memory()
            input("\nPress Enter for next demo...")

            demo_with_memory()
            input("\nPress Enter for next demo...")

            demo_multiple_threads()
            input("\nPress Enter for interactive chat...")

            interactive_chat()

        elif choice == "2":
            interactive_chat()

        else:
            print("Goodbye!")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    main()
