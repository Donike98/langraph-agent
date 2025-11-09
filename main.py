#!/usr/bin/env python3
"""LangGraph Agentic RAG - CLI Interface"""
import os
import sys

# Suppress all warnings at the earliest point
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ.setdefault('USER_AGENT', 'langraph-agent')

import argparse
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

from agent import Agent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LangGraph Agentic RAG System',
        epilog='Examples:\n'
               '  %(prog)s --offline "What is StateGraph?"\n'
               '  %(prog)s --online "What is AI?" --verbose',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('query', help='Your question')
    parser.add_argument('--online', action='store_true', help='Web search mode (Tavily+Ollama)')
    parser.add_argument('--offline', action='store_true', help='Vector store mode (ChromaDB+Ollama)')
    parser.add_argument('--verbose', action='store_true', help='Show debug output')
    parser.add_argument('--refresh', action='store_true', help='Refresh docs')
    return parser.parse_args()


def get_mode(args):
    """Determine operational mode from arguments or environment."""
    if args.online and args.offline:
        sys.exit("Error: Cannot specify both --online and --offline")
    
    if args.online:
        return 'online'
    if args.offline:
        return 'offline'
    
    # Fallback to environment variable
    mode = os.getenv('AGENT_MODE', '').lower()
    if mode in ('online', 'offline'):
        return mode
    
    sys.exit("Error: Specify mode with --online or --offline")


def validate_api_keys(mode):
    """Validate required API keys for the selected mode."""
    if mode == 'online':
        key = os.getenv('TAVILY_API_KEY')
        if not key:
            sys.exit("Error: TAVILY_API_KEY required for online mode (add to .env)")
        return key
    return None


def main():
    """Main entry point."""
    load_dotenv()
    args = parse_args()
    mode = get_mode(args)
    tavily_key = validate_api_keys(mode)
    
    try:
        agent = Agent(mode=mode, tavily_api_key=tavily_key, verbose=args.verbose)
        
        # Refresh index if requested (offline mode only)
        if args.refresh and mode == "offline":
            agent.refresh_index()
        
        result = agent.run(args.query)
        print(f"\n{result['response']}\n")
    except KeyboardInterrupt:
        sys.exit("\n\n Interrupted")
    except Exception as e:
        sys.exit(f" Error: {e}")


if __name__ == "__main__":
    main()
