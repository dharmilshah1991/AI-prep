import os

import anthropic
from dotenv import load_dotenv

MODEL="claude-haiku-5-5"
INPUT_PRICE_PER_MTOK = None
OUTPUT_PRICE_PER_MTOK = None
MAX_TOKENS = 1024

HELP = "Commands: /reset (clear history), /cost (show spend), /exit (quit)"

def estimated_cost(usage):
    if INPUT_PRICE_PER_MTOK is None or OUTPUT_PRICE_PER_MTOK is None:
        return None
    return (
        usage["input_tokens"] / 1_000_000 * INPUT_PRICE_PER_MTOK
        + usage["output_tokens"] / 1_000_000 * OUTPUT_PRICE_PER_MTOK
    )

def print_spend(usage):
    cost = estimated_cost(usage)
    cost_text = f"${cost:.6f}" if cost is not None else "unknown (prices not set)"

    print(
        f"Requests: {usage['requests']}  "
        f"Input tokens: {usage['input_tokens']}  "
        f"Output tokens: {usage['output_tokens']}  "
        f"Estimated cost: {cost_text}"
    )

def main():
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise SystemExit("Set ANTHROPIC_API_KEY in the .env file first.")

    client = anthropic.Anthropic(api_key=api_key)

    history = []

    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    print(f"Chatting with {MODEL}. {HELP}")

    while True:
        try:
            user_text = input("\nyou>").strip()
        except(EOFError, KeyboardInterrupt):
            print()
            print_spend(usage)
            break

        if not user_text:
            continue

        if user_text == "/exit":
            print_spend(usage)
            break

        if user_text == "/cost":
            print_spend(usage)
            continue

        if user_text == "/reset":
            history.clear()
            print("conversation history cleared. Spend totals are kept.")
            continue

        history.append({"role": "user", "content": user_text})

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=history,
            )
        except anthropic.APIError as error:
            history.pop()
            print("API Error: {error}")
            continue

        reply = "".join(
            block.text for block in response.content if block.type == "text"
        )
        history.append({"role": "assistant", "content": reply})
        
        usage["requests"] += 1
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
 
        print(f"\nclaude> {reply}")

if __name__ == "__main__":
    main()