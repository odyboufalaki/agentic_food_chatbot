import sys
from typing import TextIO

from agent import FoodOrderAgent


def main(
    *, agent: FoodOrderAgent | None = None,
    input_stream: TextIO | None = None, output_stream: TextIO | None = None,
) -> None:
    agent = agent if agent is not None else FoodOrderAgent()
    source = input_stream if input_stream is not None else sys.stdin
    destination = output_stream if output_stream is not None else sys.stdout
    print("Build a food order. Ask for the menu, add or edit selections, remove servings, or clear your draft. Say Submit to review for confirmation.\nType quit or exit to leave. Conversation text is logged locally.", file=destination)
    try:
        while True:
            print("You: ", end="", file=destination, flush=True)
            line = source.readline()
            if not line or line.strip().lower() in {"quit", "exit"}:
                break
            response = agent.send(line.rstrip("\r\n"))
            print(f"Agent: {response['message']}", file=destination)
    except KeyboardInterrupt:
        print("\nGoodbye.", file=destination)


if __name__ == "__main__":
    main()
