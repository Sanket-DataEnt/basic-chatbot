"""
Beginner-Friendly Chat Application with Mistral via Ollama

This is an educational chat application designed to help you understand
how Large Language Models (LLMs) work. We'll use Mistral, an open-source
LLM, running locally through Ollama.

=== WHAT YOU'LL LEARN ===
1. How to send messages to an AI model
2. How chat history provides context for conversations
3. How to handle AI responses (streaming vs. complete)
4. Basic error handling for AI applications

=== PREREQUISITES ===
Before running this app, make sure you have:
1. Installed Ollama: brew install ollama (macOS) or https://ollama.ai/download
2. Started Ollama server: ollama serve
3. Downloaded Mistral: ollama pull mistral

Let's get started!
"""

# =============================================================================
# IMPORTS
# =============================================================================
# The 'ollama' library is the official Python client for Ollama.
# It handles all the communication between our Python code and the
# Ollama server running on your computer.
import ollama

# 'sys' is a built-in Python module. We use it here to exit the program
# gracefully and to flush output (make sure text appears immediately).
import sys


# =============================================================================
# CONFIGURATION
# =============================================================================
# This is the name of the AI model we want to use.
# Mistral is a great choice for beginners because:
# - It's open-source and free
# - It runs well on most computers
# - It provides good quality responses
MODEL_NAME = "llama3.2"  # Using llama3.2 (already downloaded). Change to "mistral" after running: ollama pull mistral


# =============================================================================
# MAIN CHAT FUNCTION
# =============================================================================
def send_message_to_mistral(messages: list[dict]) -> str:
    """
    Send a conversation to Mistral and get a response.

    === WHAT IS A MESSAGE? ===
    In LLM chat applications, messages have two parts:
    - "role": Who is speaking? ("user" = you, "assistant" = the AI)
    - "content": What did they say?

    Example:
        {"role": "user", "content": "What is Python?"}
        {"role": "assistant", "content": "Python is a programming language..."}

    === WHAT IS CHAT HISTORY? ===
    We send ALL previous messages, not just the latest one. This gives
    the AI "memory" of the conversation. Without history, each message
    would be like talking to someone with amnesia!

    Args:
        messages: A list of all messages in the conversation so far.
                  Each message is a dictionary with "role" and "content".

    Returns:
        The AI's response as a string.
    """
    # === MAKING THE API CALL ===
    # ollama.chat() sends our messages to the Mistral model.
    # The model processes the conversation and generates a response.
    #
    # Parameters explained:
    # - model: Which AI model to use (we set this to "mistral")
    # - messages: The conversation history
    # - stream: If True, we get words one at a time (like ChatGPT typing)
    #           If False, we wait for the complete response
    #           We use False here to keep things simple for beginners.

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=False  # Get the complete response at once
    )

    # === EXTRACTING THE RESPONSE ===
    # The response from Ollama is a dictionary containing lots of info.
    # We only need the actual text content, which is nested inside:
    # response -> message -> content
    #
    # Example response structure:
    # {
    #     "message": {
    #         "role": "assistant",
    #         "content": "Here's my answer..."
    #     },
    #     "done": True,
    #     ... (other metadata)
    # }

    return response["message"]["content"]


def clear_screen():
    """Clear the terminal screen and show a fresh header."""
    # ANSI escape code to clear screen - works on most terminals
    print("\033[2J\033[H", end="")
    print_header()


def print_header():
    """Display a welcome header with instructions."""
    print("=" * 60)
    print("   MISTRAL CHAT - Your First AI Conversation!")
    print("=" * 60)
    print()
    print("Commands:")
    print("  - Type your question and press Enter to chat")
    print("  - Type 'clear' to start a new conversation")
    print("  - Type 'quit' or 'exit' to leave")
    print()
    print("-" * 60)
    print()


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """
    The main chat loop.

    === HOW CHAT APPLICATIONS WORK ===
    1. Display a prompt asking for user input
    2. Get the user's message
    3. Add it to the conversation history
    4. Send the history to the AI
    5. Display the AI's response
    6. Add the response to history
    7. Repeat!

    The key insight is that we maintain a HISTORY of all messages.
    This is how the AI "remembers" what you talked about earlier.
    """

    # === CONVERSATION HISTORY ===
    # This list will store all messages exchanged between you and the AI.
    # Each message is a dictionary with "role" and "content".
    # We start with an empty list - a fresh conversation!
    messages: list[dict] = []

    # Clear screen and show welcome message
    clear_screen()

    print("Hello! I'm Mistral, an AI assistant running on your computer.")
    print("Ask me anything, and I'll do my best to help!")
    print()

    # === THE CHAT LOOP ===
    # This loop runs forever until the user decides to quit.
    # It's the heart of any chat application!
    while True:
        try:
            # --- Step 1: Get User Input ---
            # The input() function shows "You: " and waits for you to type.
            # .strip() removes extra whitespace from the beginning and end.
            user_input = input("You: ").strip()

            # --- Step 2: Handle Special Commands ---
            # Check if the user wants to quit or clear the conversation
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! Thanks for chatting!")
                sys.exit(0)

            if user_input.lower() == "clear":
                # Reset the conversation history
                messages = []
                clear_screen()
                print("Conversation cleared! Let's start fresh.")
                print()
                continue  # Go back to the start of the loop

            # Skip empty input
            if not user_input:
                continue

            # --- Step 3: Add User Message to History ---
            # We create a message dictionary and add it to our history.
            # Role is "user" because YOU are sending this message.
            messages.append({
                "role": "user",
                "content": user_input
            })

            # --- Step 4: Get AI Response ---
            # Show a loading indicator since this might take a moment
            print("\nMistral: ", end="", flush=True)

            # Send the ENTIRE conversation history to get a contextual response
            response = send_message_to_mistral(messages)

            # --- Step 5: Display the Response ---
            print(response)
            print()  # Add a blank line for readability

            # --- Step 6: Add AI Response to History ---
            # We also save the AI's response to history.
            # Role is "assistant" because the AI is responding.
            # This way, in future messages, the AI knows what it said before!
            messages.append({
                "role": "assistant",
                "content": response
            })

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nInterrupted! Goodbye!")
            sys.exit(0)

        except ollama.ResponseError as e:
            # === ERROR HANDLING ===
            # This happens when Ollama returns an error.
            # Common causes:
            # - The model isn't downloaded (run: ollama pull mistral)
            # - The model name is misspelled
            print(f"\n[Error from Ollama]: {e}")
            print("Make sure you have pulled the model: ollama pull mistral")
            # Remove the failed message from history so we can try again
            if messages and messages[-1]["role"] == "user":
                messages.pop()

        except EOFError:
            # Handle end of input (e.g., when piping input)
            print("\n\nEnd of input. Goodbye!")
            sys.exit(0)

        except Exception as e:
            # Catch any other errors
            # Most common: Ollama server isn't running
            print(f"\n[Error]: {e}")
            print("\nTroubleshooting tips:")
            print("1. Is Ollama running? Start it with: ollama serve")
            print("2. Is Mistral downloaded? Run: ollama pull mistral")
            print("3. Check if port 11434 is available")
            # Remove the failed message from history
            if messages and messages[-1]["role"] == "user":
                messages.pop()


# =============================================================================
# ENTRY POINT
# =============================================================================
# This is a Python convention. The code inside only runs when you
# execute this file directly (python chat.py), not when importing it.
if __name__ == "__main__":
    main()
