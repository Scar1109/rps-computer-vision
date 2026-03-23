"""
game_logic.py — Game Logic for Rock Paper Scissors AI

Handles AI move generation, winner determination, and score tracking.
Kept separate from gesture detection and UI for clean modularity.
"""

import random

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MOVES = ["Rock", "Paper", "Scissors"]

# Emoji mapping for display
MOVE_EMOJIS = {
    "Rock": "🪨",
    "Paper": "📄",
    "Scissors": "✂️",
}


def get_ai_move() -> str:
    """
    Generate a random AI move.

    Returns:
        One of "Rock", "Paper", or "Scissors".
    """
    return random.choice(MOVES)


def determine_winner(player_move: str, ai_move: str) -> str:
    """
    Determine the round winner.

    Args:
        player_move: The player's gesture ("Rock", "Paper", or "Scissors").
        ai_move: The AI's move ("Rock", "Paper", or "Scissors").

    Returns:
        "Win"  → Player wins
        "Lose" → AI wins
        "Draw" → It's a tie
    """
    if player_move == ai_move:
        return "Draw"

    # Winning combinations for the player
    wins = {
        ("Rock", "Scissors"),      # Rock crushes Scissors
        ("Paper", "Rock"),         # Paper covers Rock
        ("Scissors", "Paper"),     # Scissors cuts Paper
    }

    if (player_move, ai_move) in wins:
        return "Win"
    else:
        return "Lose"


# ──────────────────────────────────────────────────────────────
# Round history entry
# ──────────────────────────────────────────────────────────────
def format_round_entry(
    round_num: int,
    player_move: str,
    ai_move: str,
    result: str,
) -> dict:
    """
    Create a formatted round history entry.

    Returns:
        A dictionary with round details for display in the UI.
    """
    result_emoji = {"Win": "🏆", "Lose": "😞", "Draw": "🤝"}.get(result, "")

    return {
        "Round": round_num,
        "You": f"{MOVE_EMOJIS.get(player_move, '')} {player_move}",
        "AI": f"{MOVE_EMOJIS.get(ai_move, '')} {ai_move}",
        "Result": f"{result_emoji} {result}",
    }
