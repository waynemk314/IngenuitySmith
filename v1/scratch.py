"""
test for code output from agentic system
"""

import random

def calculate_pi_monte_carlo(n: int) -> float:
    """Calculate pi using the Monte Carlo method.

    Args:
        n (int): Number of random samples.

    Returns:
        float: Approximation of pi.

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError("Number of samples must be greater than zero.")

    pi_approximation = (sum(1 for _ in range(n) if random.random()**2 + random.random()**2 <= 1) / n) * 4
    return round(pi_approximation, 10)

if __name__ == "__main__":
    print(calculate_pi_monte_carlo(100000000))