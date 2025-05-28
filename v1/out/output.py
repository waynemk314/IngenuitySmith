import sys
import re

def is_palindrome(s: str) -> bool:
    """
    Check if a given string is a palindrome, ignoring case and non-alphanumeric characters.

    :param s: The input string to check.
    :return: True if the string is a palindrome, False otherwise.
    """
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    # Check if the cleaned string is equal to its reverse
    return cleaned_string == cleaned_string[::-1]

def main():
    """
    Main function to execute the palindrome check from command-line arguments.
    """
    try:
        # Ensure only one argument is provided
        if len(sys.argv) != 2:
            raise ValueError("Exactly one string argument is required.")
        
        input_string = sys.argv[1]
        result = is_palindrome(input_string)
        print(result)
    
    except IndexError:
        print("Usage: python script.py <string>")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()