import numpy as np

# Transition matrix as a dictionary of dictionaries
P = {
    'B': {'B': 0.1, 'K': 0.325, 'O': 0.25, '-': 0.325},
    'K': {'B': 0.4, 'K': 0.0,   'O': 0.4,  '-': 0.2},
    'O': {'B': 0.2, 'K': 0.2,   'O': 0.2,  '-': 0.4},
    '-': {'B': 1, 'K': 0,   'O': 0,  '-': 0}
}

letters = ['B', 'K', 'O']

from math import log

def most_probable_word(length):
    dp = [{} for _ in range(length + 1)]  # dp[t][letter] = (log_prob, path)
    dp[0]['B'] = (0.0, ['B'])  # log(1) = 0

    for t in range(1, length):
        for curr in letters:
            best_log_prob = float('-inf')
            best_path = []
            for prev in dp[t - 1]:
                if curr in P[prev] and P[prev][curr] > 0:
                    log_prob = dp[t - 1][prev][0] + log(P[prev][curr])
                    if log_prob > best_log_prob:
                        best_log_prob = log_prob
                        best_path = dp[t - 1][prev][1] + [curr]
            if best_path:
                dp[t][curr] = (best_log_prob, best_path)

    # Last step: transition to '-'
    best_final_prob = float('-inf')
    best_word = []
    for prev in dp[length - 1]:
        if '-' in P[prev] and P[prev]['-'] > 0:
            log_prob = dp[length - 1][prev][0] + log(P[prev]['-'])
            if log_prob > best_final_prob:
                best_final_prob = log_prob
                best_word = dp[length - 1][prev][1] + ['-']

    return best_word, np.exp(best_final_prob)  # Return word and actual probability

# Run the function for a word of length 5
word, prob = most_probable_word(5)
print("Most probable word of length 5:", ''.join(word[:-1]))  # drop '-' for display
print("Probability:", prob)
