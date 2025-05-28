"""
test for code output from agentic system
"""

import re
import sys

def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome, ignoring case and non-alphanumeric characters.

    Args:
        s (str): The input string to check.

    Returns:
        bool: True if the string is a palindrome, False otherwise.
    """
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    return cleaned_string == cleaned_string[::-1]

def main():
    """Main function to execute the palindrome check from command-line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python script.py <string>", file=sys.stderr)
        sys.exit(1)

    input_string = sys.argv[1]
    result = is_palindrome(input_string)
    print(result)

if __name__ == "__main__":
    main()