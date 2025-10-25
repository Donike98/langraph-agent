#!/usr/bin/env python3
"""LangGraph Documentation Agent - Main CLI"""
import os
import sys
import argparse
from dotenv import load_dotenv
from agent import Agent 

def main(): 
    # Load .env configuration
    load_dotenv()
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='LangGraph Documentation Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python main.py --offline "How do I use checkpointers?"
                python main.py --online "Latest features?"
                
                Or set environment variable:
                export AGENT_MODE=offline
                python main.py "How do I use checkpointers?"
        """
    )
    
    parser.add_argument('query', help='Your question')
    parser.add_argument('--online', action='store_true', help='Use online mode (Tavily+Gemini)')
    parser.add_argument('--offline', action='store_true', help='Use offline mode (BM25+Ollama)')
    
    args = parser.parse_args()
    
    # Check for conflicting flags
    if args.online and args.offline:
        print("Error: Cannot use both --online and --offline flags")
        sys.exit(1)
    
    # Check for API keys
    google_api_key = os.getenv('GOOGLE_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    
    # Determine mode: flags take priority over env var
    if args.online:
        mode = 'online'
        if not google_api_key or not tavily_api_key:
            print(" Online mode requires GOOGLE_API_KEY and TAVILY_API_KEY in .env")
            sys.exit(1)
    elif args.offline:
        mode = 'offline'
    else:
        # Check AGENT_MODE env var
        agent_mode_env = os.getenv('AGENT_MODE', '').lower()
        if agent_mode_env == 'online':
            mode = 'online'
            if not google_api_key or not tavily_api_key:
                print(" Online mode requires GOOGLE_API_KEY and TAVILY_API_KEY in .env")
                sys.exit(1)
        elif agent_mode_env == 'offline':
            mode = 'offline'
        else:
            # No mode specified - require explicit choice
            print("Error: Please specify a mode:")
            print("  --online    Use online mode (Tavily+Gemini)")
            print("  --offline   Use offline mode (BM25+Ollama)")
            print("  Or set: export AGENT_MODE=online|offline")
            sys.exit(1)
    
    # Initialize agent
    try:
        agent = Agent(
            google_api_key=google_api_key,
            mode=mode,
            tavily_api_key=tavily_api_key
        )
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Run query
    try:
        result = agent.run(args.query)
        print(f"\n{result['response']}\n")
    except KeyboardInterrupt:
        print("\n\n Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

