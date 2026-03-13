"""
Tools Demo with LangGraph and Ollama

This script demonstrates how to give an AI model the ability to use TOOLS.
Think of tools as "superpowers" you give to the AI - functions it can call
to perform specific tasks.

=== WHAT ARE TOOLS? ===
Normally, an LLM can only generate text. But what if you ask it:
"What is 3847 * 2918?"

The LLM might guess (and get it wrong!). But with TOOLS, we can:
1. Define a "calculator" function
2. Tell the AI about this tool
3. When the AI needs to calculate, it CALLS the tool
4. The tool returns the real answer
5. The AI uses that answer in its response

=== HOW IT WORKS ===
1. We define tools as Python functions with the @tool decorator
2. We "bind" these tools to the model (tell the model about them)
3. When we ask a question, the model decides if it needs a tool
4. If yes, it generates a "tool call" (which tool + what arguments)
5. We execute the tool and return the result
6. The model uses the result to form its final answer

=== PREREQUISITES ===
1. Ollama running: ollama serve
2. Model downloaded: ollama pull llama3.2
3. Install dependencies: uv add langchain langgraph langchain-ollama
"""

# =============================================================================
# IMPORTS
# =============================================================================
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import json

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "llama3.2"  # Same model we used in chat.py


# =============================================================================
# STEP 1: DEFINE TOOLS
# =============================================================================
# Tools are just Python functions with the @tool decorator.
# The decorator tells LangChain to:
# - Extract the function name, description, and parameters
# - Format them so the AI model understands what the tool does

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """
    Perform a mathematical calculation.

    Use this tool when you need to do math operations like addition,
    subtraction, multiplication, or division.

    Args:
        operation: The math operation to perform.
                   Must be one of: "add", "subtract", "multiply", "divide"
        a: The first number
        b: The second number

    Returns:
        The result of the calculation as a string
    """
    # === THIS IS THE ACTUAL TOOL LOGIC ===
    # When the AI "calls" this tool, this code runs!

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Cannot divide by zero!"
        result = a / b
    else:
        return f"Error: Unknown operation '{operation}'. Use add, subtract, multiply, or divide."

    return f"{a} {operation} {b} = {result}"


# =============================================================================
# STEP 2: SET UP THE MODEL WITH TOOLS
# =============================================================================
def create_model_with_tools():
    """
    Create an Ollama model and bind tools to it.

    === WHAT IS "BINDING TOOLS"? ===
    When we bind tools to a model, we're essentially telling the model:
    "Hey, here are some functions you can use. Here's what each one does,
    and here's how to call them."

    The model doesn't actually RUN the tools - it just knows how to
    REQUEST that we run them. We handle the actual execution.
    """
    # Create the base model (same as chat.py, but using LangChain wrapper)
    model = ChatOllama(
        model=MODEL_NAME,
        temperature=0  # Lower temperature = more deterministic responses
    )

    # === BIND TOOLS TO THE MODEL ===
    # This tells the model about our calculator tool
    # The model will now know it CAN use this tool when needed
    tools = [calculator]
    model_with_tools = model.bind_tools(tools)

    return model_with_tools, tools


# =============================================================================
# STEP 3: CREATE A SIMPLE AGENT LOOP
# =============================================================================
def run_agent(user_question: str):
    """
    Run a simple agent that can use tools to answer questions.

    === THE AGENT LOOP ===
    1. Send the question to the model
    2. Check if the model wants to use a tool
    3. If yes: run the tool, send result back, repeat
    4. If no: we have our final answer!

    This is a simplified version of what LangGraph does automatically.
    """
    print(f"\n{'='*60}")
    print(f"Question: {user_question}")
    print(f"{'='*60}\n")

    # Create model with tools
    model, tools = create_model_with_tools()
    tools_by_name = {t.name: t for t in tools}

    # Start with a system message and the user's question
    messages = [
        SystemMessage(content="""You are a helpful assistant with access to a calculator tool.
When asked to perform calculations, ALWAYS use the calculator tool to get accurate results.
Don't try to calculate in your head - use the tool!"""),
        HumanMessage(content=user_question)
    ]

    # === THE AGENT LOOP ===
    max_iterations = 5  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        # Ask the model
        response = model.invoke(messages)
        print(f"Model response type: {type(response).__name__}")

        # === CHECK FOR TOOL CALLS ===
        # If the model wants to use a tool, response.tool_calls will have data
        if response.tool_calls:
            print(f"Model wants to use {len(response.tool_calls)} tool(s)!")

            # Add the model's response to our message history
            messages.append(response)

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"\n  Tool: {tool_name}")
                print(f"  Arguments: {json.dumps(tool_args, indent=4)}")

                # === EXECUTE THE TOOL ===
                # This is where we actually run our Python function!
                tool_func = tools_by_name[tool_name]
                result = tool_func.invoke(tool_args)

                print(f"  Result: {result}")

                # === SEND RESULT BACK TO MODEL ===
                # We create a ToolMessage with the result
                # The model will use this to form its final answer
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )
        else:
            # === NO TOOL CALLS = FINAL ANSWER ===
            print("\nModel provided final answer (no tool calls)")
            print(f"\n{'='*60}")
            print("FINAL ANSWER:")
            print(f"{'='*60}")
            print(response.content)
            return response.content

    print("Max iterations reached!")
    return None


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Demonstrate tool usage with various questions.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║           TOOLS DEMO - Teaching AI to Use Functions          ║
╠══════════════════════════════════════════════════════════════╣
║  This demo shows how an AI can use a "calculator" tool to    ║
║  perform accurate mathematical calculations.                  ║
║                                                              ║
║  Watch how the AI:                                           ║
║  1. Recognizes when it needs to calculate                    ║
║  2. Calls the calculator tool with the right arguments       ║
║  3. Uses the result to answer your question                  ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Test questions that require the calculator
    test_questions = [
        "What is 3847 multiplied by 2918?",
        "If I have 1500 dollars and spend 847.50, how much do I have left?",
        "What is 144 divided by 12?",
    ]

    print("Running demo with test questions...\n")

    for question in test_questions:
        try:
            run_agent(question)
            print("\n" + "-"*60 + "\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running: ollama serve")
            break

    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Now you can ask your own questions!")
    print("Try asking math questions to see the calculator tool in action.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not user_input:
                continue
            run_agent(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
