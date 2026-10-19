import os

import anthropic
from dotenv import load_dotenv

MODEL="claude-haiku-5-5"
PROMPT="Explain in two sentences why the sky is blue."
TEMPERATURES = [0,1]

def main():
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("API key is not present. please set it up.")

    client = anthropic.Anthropic(api_key=api_key)

    for temperature in TEMPERATURES:
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content" : PROMPT}],
        )

        reply = "".join(
            block.text for block in response.content if block.type == "text"
        )

        print(f"-- temperature {temperature} --")

        print (reply)
        print()

if __name__ == "__main__":
    main()