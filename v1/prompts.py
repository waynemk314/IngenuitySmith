coding_task_prompts = [
    """
    Write a Python script that takes a single string as a command-line argument.
    The script should determine if the input string is a palindrome (reads the same forwards and backward),
    ignoring case and non-alphanumeric characters. Print 'True' if it's a palindrome, and 'False' otherwise.
    For example:
    - Input: "A man, a plan, a canal: Panama" -> Output: True
    - Input: "race a car" -> Output: False
    - Input: "Was it a car or a cat I saw?" -> Output: True
    - Input: "hello" -> Output: False
    """,
    """
    Create a Python script that implements a Caesar cipher.
    The script should accept three command-line arguments:
    1.  The mode: 'encrypt' or 'decrypt'.
    2.  The text string to process (enclose in quotes if it contains spaces).
    3.  The shift key (an integer).

    The script should only shift alphabetic characters (a-z, A-Z), preserving their case.
    Non-alphabetic characters should remain unchanged. The shift should wrap around the alphabet
    (e.g., shifting 'z' by 1 results in 'a').
    Print the resulting processed string.

    Example:
    - Input: `encrypt "Hello, World!" 3` -> Output: `Khoor, Zruog!`
    - Input: `decrypt "Khoor, Zruog!" 3` -> Output: `Hello, World!`
    - Input: `encrypt "Cipher" 26` -> Output: `Cipher`
    - Input: `encrypt "Veni, Vidi, Vici!" 4` -> Output: `Ziph, Zmhm, Zmgm!`
    """,
    """
    Write a Python script that reads a text file specified by a command-line argument.
    The script should count the frequency of each word in the file. Words should be
    considered case-insensitive, and punctuation attached to words (e.g., 'example.', 'word,')
    should be stripped so that 'example' and 'example.' count as the same word.
    The script should then print the 10 most frequent words and their counts, sorted in
    descending order of frequency. If two words have the same frequency, their relative order
    doesn't matter. If there are fewer than 10 unique words, print all of them.

    Format for each printed line: `word: count`

    Example:
    If `input.txt` contains:
    `The quick brown fox jumps over the lazy dog. The dog barks.`

    Command: `python your_script.py input.txt`
    Expected Output (order of same-frequency words might vary):
    ```
    the: 3
    dog: 2
    quick: 1
    brown: 1
    fox: 1
    jumps: 1
    over: 1
    lazy: 1
    barks: 1
    ```
    """,
    """
    Write a Python script. It is given a predefined list of task descriptions, for example: `initial_tasks=['Buy groceries', 'Read a book', 'Pay bills']`.
    It also receives a command as a string argument which can be one of:
    1.  `list`: Print all tasks from the (potentially modified) initial list with an ID (1-indexed). E.g., `1. Buy groceries`.
    2.  `done <ID>` (e.g., `done 2`): "Remove" the task with the given ID from the list and print the remaining tasks, 1-indexed.
    3.  `add "<new_task>"` (e.g. `add "Walk the dog"`): Add the new task to the end of the list and print the full list, 1-indexed.

    The script should process this one command against the initial list and print the resulting list or confirmation.
    Assume the `initial_tasks` list will be provided to the script (e.g., hardcoded or passed in a way your framework supports, separate from the command string).
    The command string itself will be the primary input for deciding the action.

    Example with `initial_tasks=['Buy groceries', 'Read a book', 'Pay bills']`:
    - Input command: `add "Walk the dog"`
      Expected output:
      ```
      1. Buy groceries
      2. Read a book
      3. Pay bills
      4. Walk the dog
      ```
    - Input command: `done 1` (after the above add, or on the original list)
      Expected output (if on original list):
      ```
      1. Read a book
      2. Pay bills
      ```
    - Input command: `list`
      Expected output (if on original list):
      ```
      1. Buy groceries
      2. Read a book
      3. Pay bills
      ```
    """,
    """
    Create a Python program to organize files in a specified source directory into a target directory.
    The program should consist of two Python files:
    1.  `file_utils.py`: This module will contain utility functions:
        *   `get_file_extension(filename)`: Returns the lowercase file extension (e.g., 'txt', 'jpg') or an empty string if no extension.
        *   `create_directory_if_not_exists(dir_path)`: Creates a directory if it doesn't already exist.
    2.  `organizer.py`: This main script will use `file_utils.py`. It should take two command-line arguments: `source_directory` and `target_directory`.

    The `organizer.py` script should:
    *   Iterate through all files (not directories) in the `source_directory` (non-recursively).
    *   For each file, get its extension using `file_utils.get_file_extension`.
    *   If the extension is empty, categorize it as 'misc'.
    *   In the `target_directory`, create a subdirectory named after the extension (e.g., `target_directory/txt`, `target_directory/jpg`, `target_directory/misc`) using `file_utils.create_directory_if_not_exists`.
    *   Move the file from `source_directory` to the corresponding subdirectory in `target_directory`.
    *   Print a log message for each file moved, e.g., `Moved 'file.txt' to 'target_directory/txt/'`.
    *   Handle potential errors gracefully (e.g., source directory not found, permission issues) by printing an error message.

    For testing in Docker, you'll need to set up a dummy `source_directory` with a few files of different types (e.g., `doc.txt`, `image.jpg`, `archive.zip`, `no_extension_file`) and an empty `target_directory`. The test should verify that the `target_directory` and its subdirectories are created correctly and contain the moved files.
    """,
    """
    Develop a Python program that processes a CSV file containing sales data and generates a summary report.
    The program should ideally be structured into two files:
    1.  `data_processor.py`: Contains functions/classes to handle CSV loading and data processing.
        *   A function `load_sales_data(csv_filepath)` that reads the CSV and returns a list of dictionaries or custom objects representing sales records. Each record must contain at least 'ProductName', 'Price', and 'QuantitySold'.
        *   A function `calculate_total_sales_per_product(sales_data)` that returns a dictionary where keys are product names and values are their total sales amount (Price * QuantitySold).
        *   A function `find_top_selling_product(sales_data_per_product)` that returns a tuple: (product_name, total_sales_amount) for the product with the highest total sales. If multiple products have the same max sales, any one of them can be returned.
        *   A function `calculate_total_revenue(sales_data_per_product)` that returns the sum of all product sales (grand total revenue).
    2.  `report_generator.py`: The main script that uses `data_processor.py`. It takes the CSV file path as a command-line argument.

    The input CSV file (e.g., `sales.csv`) will have at least the columns: `ProductID,ProductName,Category,Price,QuantitySold`.
    Example `sales.csv` content:
    ```csv
    ProductID,ProductName,Category,Price,QuantitySold
    P001,Laptop,Electronics,1200,10
    P002,Mouse,Electronics,25,50
    P003,Keyboard,Electronics,75,30
    P004,Desk Lamp,Home Goods,40,20
    P001,Laptop,Electronics,1200,5
    P002,Mouse,Electronics,25,100
    ```

    The `report_generator.py` script should:
    *   Use `data_processor.load_sales_data` to load the data.
    *   Use `data_processor.calculate_total_sales_per_product`.
    *   Use `data_processor.find_top_selling_product`.
    *   Use `data_processor.calculate_total_revenue`.
    *   Print a summary report to the console. Prices and totals should be formatted to two decimal places, prefixed with a dollar sign.

        Example Output Format:
        ```
        Sales Report
        --------------------
        Total Revenue: $XX,XXX.XX
        Top Selling Product: ProductName (Total Sales: $Y,YYY.YY)

        Sales by Product:
        - ProductA: $AAA.AA
        - ProductB: $BBB.BB
        ...
        ```
    Handle cases like an empty CSV file or a file not found by printing an appropriate error message. If the CSV is empty or malformed leading to no processable data, the report should indicate this (e.g., Total Revenue: $0.00, Top Selling Product: None).
    """
]

# You can then save this list to a file or use it directly.
# For example, to write to a file:
# with open("prompts.py", "w") as f:
#     f.write("coding_task_prompts = [\n")
#     for prompt in coding_task_prompts:
#         f.write(f"    \"\"\"{prompt.strip()}\"\"\",\n") # .strip() to remove leading/trailing whitespace from the heredoc
#     f.write("]\n")