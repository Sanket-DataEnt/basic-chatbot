"""
Streaming Chat Application with Mistral via Ollama

This is the STREAMING version of the chat application. Instead of waiting
for the complete response, you'll see words appear one by one - just like
ChatGPT!

=== WHAT IS STREAMING? ===
When an LLM generates text, it produces one word (token) at a time.
- Non-streaming: Wait until ALL words are generated, then show everything
- Streaming: Show each word AS IT'S GENERATED (feels much more responsive!)

This file demonstrates the streaming approach. Compare with chat.py to see
the difference!

=== PREREQUISITES ===
1. Installed Ollama: brew install ollama (macOS) or https://ollama.ai/download
2. Started Ollama server: ollama serve
3. Downloaded a model: ollama pull llama3.2 (or ollama pull mistral)
"""

import ollama
import sys


# Configuration - change this to use a different model
MODEL_NAME = "llama3.2"  # Or "mistral" if you've pulled it


def send_message_streaming(messages: list[dict]) -> str:
    """
    Send a conversation to the AI and STREAM the response word by word.

    === HOW STREAMING WORKS ===
    1. We call ollama.chat() with stream=True
    2. Instead of getting one big response, we get a "generator"
    3. A generator yields small pieces (chunks) one at a time
    4. Each chunk contains a few words (tokens)
    5. We print each chunk immediately, creating a "typing" effect

    This makes the AI feel much more responsive because you see
    output immediately instead of waiting for the full response.

    Args:
        messages: The conversation history

    Returns:
        The complete response as a string (assembled from all chunks)
    """
    # === STREAMING API CALL ===
    # The key difference: stream=True
    # This returns an iterator instead of a complete response
    stream = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True  # Enable streaming!
    )

    # === COLLECTING AND DISPLAYING THE RESPONSE ===
    # We'll build up the full response while also printing each chunk
    full_response = ""

    # Iterate through each chunk as it arrives
    # This loop runs once for each "piece" of the response
    for chunk in stream:
        # Each chunk has a structure like:
        # {
        #     "message": {"role": "assistant", "content": "word"},
        #     "done": False  # True on the last chunk
        # }

        # Extract the text content from this chunk
        chunk_content = chunk["message"]["content"]

        # Print immediately WITHOUT a newline (end="")
        # flush=True forces Python to display it right away
        print(chunk_content, end="", flush=True)

        # Add to our complete response
        full_response += chunk_content

    # Print a newline after the complete response
    print()

    return full_response


def clear_screen():
    """Clear the terminal screen and show a fresh header."""
    print("\033[2J\033[H", end="")
    print_header()


def print_header():
    """Display a welcome header with instructions."""
    print("=" * 60)
    print("   STREAMING CHAT - Watch the AI Think!")
    print("=" * 60)
    print()
    print("This version shows responses WORD BY WORD as they're generated.")
    print()
    print("Commands:")
    print("  - Type your question and press Enter to chat")
    print("  - Type 'clear' to start a new conversation")
    print("  - Type 'quit' or 'exit' to leave")
    print()
    print("-" * 60)
    print()


def main():
    """
    The main chat loop with streaming responses.

    The structure is almost identical to chat.py, but we use
    send_message_streaming() instead of send_message_to_mistral().
    This gives us the real-time "typing" effect!
    """
    # Conversation history
    messages: list[dict] = []

    # Clear screen and show welcome
    clear_screen()

    print("Hello! I'm an AI assistant running locally on your computer.")
    print("Watch how I generate responses word by word!")
    print()

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! Thanks for chatting!")
                sys.exit(0)

            if user_input.lower() == "clear":
                messages = []
                clear_screen()
                print("Conversation cleared! Let's start fresh.")
                print()
                continue

            if not user_input:
                continue

            # Add user message to history
            messages.append({
                "role": "user",
                "content": user_input
            })

            # Show the AI label, then stream the response
            print("\nAI: ", end="", flush=True)

            # === THE KEY DIFFERENCE ===
            # This function streams the response word by word!
            response = send_message_streaming(messages)

            print()  # Blank line for readability

            # Add response to history
            messages.append({
                "role": "assistant",
                "content": response
            })

        except KeyboardInterrupt:
            print("\n\nInterrupted! Goodbye!")
            sys.exit(0)

        except ollama.ResponseError as e:
            print(f"\n[Error from Ollama]: {e}")
            print(f"Make sure you have pulled the model: ollama pull {MODEL_NAME}")
            if messages and messages[-1]["role"] == "user":
                messages.pop()

        except EOFError:
            print("\n\nEnd of input. Goodbye!")
            sys.exit(0)

        except Exception as e:
            print(f"\n[Error]: {e}")
            print("\nTroubleshooting tips:")
            print("1. Is Ollama running? Start it with: ollama serve")
            print(f"2. Is the model downloaded? Run: ollama pull {MODEL_NAME}")
            print("3. Check if port 11434 is available")
            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    main()
