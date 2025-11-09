#!/usr/bin/env python3
"""
Autonomous Agent Setup
One-command setup: downloads docs and builds index
"""
import sys
from agent import Agent

def main():
    print("LangGraph Agent")
    print("="*70)
    print()
    
    # Create agent instance
    print("Initializing autonomous agent...")
    agent = Agent(mode="offline")
    print("✓ Agent ready\n")
    
    # Step 1: Pull documentation
    print("Step 1: Loading documentation...")
    docs_list = agent.pull_docs(verbose=True)
    
    print()
    
    # Step 2: Build index
    print("Step 2: Building vector store...")
    success = agent.build_index(docs_list=docs_list, verbose=True)
    if not success:
        print("\n Failed to build index")
        sys.exit(1)
    
    print()
    print("="*70)
    print(" Setup Complete! The agent is fully autonomous and ready.")
    print()
    print("Usage:")
    print("  python3 main.py --offline \"Your question here\"")
    print("  python3 main.py --online \"Your question here\"")
    print("="*70)

if __name__ == "__main__":
    main()

