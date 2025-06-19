#!/usr/bin/env python3
"""Simple DVC environment setup - loads .secrets and exports variables"""

import os
import sys
from dotenv import load_dotenv

# Required DVC variables
REQUIRED_VARS = [
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY', 
    'AWS_DEFAULT_REGION',
    'HF_TOKEN'
]

def main():
    # Load .secrets file
    if not os.path.exists('.secrets'):
        print("❌ Error: .secrets file not found!")
        sys.exit(1)
    
    load_dotenv('.secrets')
    
    # Check all required variables are present
    missing = []
    for var in REQUIRED_VARS:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing required variables: {', '.join(missing)}")
        sys.exit(1)
    
    # Export for bash/zsh
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        print(f"export {var}='{value}'")

if __name__ == "__main__":
    main() 