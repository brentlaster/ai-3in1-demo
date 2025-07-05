#!/usr/bin/env python3
# weather_mcp_client.py
# ------------------------------------------------------------------
# Robust MCP client showing Thought → Action → Observation → Final
# with retries on MCP tool errors (e.g. timeouts), and an interactive loop.

import asyncio
import json
import re
import textwrap

from fastmcp import Client
from fastmcp.exceptions import ToolError
from langchain_ollama import ChatOllama  # pip install langchain-ollama

# ── 1. System prompt ───────────────────────────────────────────────

# ── 2. Regex for extracting the JSON args ───────────────────────────
ARGS_RE = re.compile(r"Args:\s*(\{.*?\})(?:\s|$)", re.S)

# ── 3. Unwrap helper ─────────────────────────────────────────────────
def unwrap(x):
    if isinstance(x, list) and len(x) == 1:
        return unwrap(x[0])
    if hasattr(x, "text"):
        try:
            return json.loads(x.text)
        except Exception:
            return x.text
    if hasattr(x, "value"):
        return x.value
    return x

# ── 4. Single-run TAO helper with error handling ──────────────────────
async def run(question: str):


        print("\n--- Thought → Action → Observation → Final ---\n")

        # 1. Planning turn → get_weather
        reply1 = llm.invoke(messages)
        plan1  = reply1.content.strip()
        print(plan1 + "\n")

        args1 = json.loads(ARGS_RE.search(plan1).group(1))

        try:
        except ToolError as e:
            print(f"⚠️  Error calling get_weather: {e}\n")
            return

        result1 = unwrap(raw1)
        temp_c     = result1.get("temperature")
        conditions = result1.get("conditions", "Unknown")
        print(f"Observation: {{'temperature': {temp_c}, 'conditions': '{conditions}'}}\n")

        # feed back only the numeric temperature
        messages += [
            {"role": "assistant", "content": plan1},
            {"role": "user",      "content": f"Observation: {temp_c}"},
        ]

        # 2. Planning turn → convert_c_to_f
        reply2 = llm.invoke(messages)
        plan2  = reply2.content.strip()
        print(plan2 + "\n")

        try:
        except ToolError as e:
            print(f"⚠️  Error calling convert_c_to_f: {e}\n")
            return

        result2 = unwrap(raw2)
        try:
            temp_f = float(result2)
        except Exception:
            print(f"⚠️  Unexpected result from convert_c_to_f: {result2}\n")
            return

        print(f"Observation: {{'temperature_f': {temp_f}}}\n")

        # Final answer
        print(f"Final: {conditions} ({temp_f:.1f} °F)\n")

# ── 5. Interactive loop ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Weather agent (type 'exit' to quit)\n")
    while True:
        loc = input("Location (or 'exit'): ").strip()
        if loc.lower() == "exit":
            print("Goodbye!")
            break

        question = f"What is the current weather in {loc}?"
        asyncio.run(run(question))
