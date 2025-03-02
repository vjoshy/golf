import numpy as np
import random

# Function to generate a deck of cards
def generate_deck():
    values = list(range(1, 11)) + [10, 10, 0]  # 1-10, J=10, Q=10, K=0
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

    deck = []
    for value in values:
        for suit in suits:
            deck.append((value, suit))
    return deck

    
# Function to deal cards to players
def deal_cards(deck, num_players):
    random.shuffle(deck)
    hands = []
    for i in range(num_players):
        hand = []
        for j in range(4): # Deal 4 cards per player
            hand.append(deck.pop())
        hands.append(hand)
    return hands, deck

# Function to calculate the score of a hand
def calculate_score(hand):
    total = 0
    for card in hand:
        total += card[0]  # Use the card value for scoring
    return total

# Function to print the current state of the player's hand
def print_hand(hand, revealed):
    for i, card in enumerate(hand):
        if revealed[i]:
            print(f"[{i + 1}] {card[0]} of {card[1]}")
        else:
            print(f"[{i + 1}] [Hidden]")

# Game setup
num_players = 2
print("\nWelcome to Golf Card Game!")
deck = generate_deck()
#print(deck)

hands, deck = deal_cards(deck, num_players)

#print(deck.pop())

print(hands[0])
print(hands[1])

revealed = []
for n in range(num_players):
    revealed.append([False, False, False, False])
discard_pile = [deck.pop()]

print(discard_pile)

# Main game loop
game_over = False
while not game_over:
    for player in range(num_players):
        print(f"\n--- Player {player + 1}'s Turn ---")
        print("Your hand:")
        print_hand(hands[player], revealed[player])
        print(f"Discard pile: {discard_pile[-1][0]} of {discard_pile[-1][1]}")

        # Player action
        action = input("Enter '1' to draw a card: ")
        if action == '1':
            draw_action = input("Choose an action - (1) Draw from unknown deck or  (2) Draw from top of discard pile: ")

            if draw_action == '1':
                drawn_card = deck.pop()
                print(f"You drew: {drawn_card[0]} of {drawn_card[1]}")

                keep_or_discard_action = input("Choose an action - (1) Replace a card from your hand or (2) Discard drawn card: ")

                if keep_or_discard_action == '1':
                    idx = int(input("Choose a card from your hand to replace with drawn card (1-4): ")) - 1

                    if not revealed[player][idx]:
                        revealed[player][idx] = True
                    print(hands[player][idx])
                    discard_pile.append(hands[player][idx])
                    hands[player][idx] = drawn_card

                elif keep_or_discard_action == '2':
                    discard_pile.append(drawn_card)

            elif draw_action == '2':
                drawn_discard = discard_pile.pop()
                idx = int(input("Choose a card from your hand to replace with card drawn from top of discard (1-4): ")) - 1
                if not revealed[player][idx]:
                    revealed[player][idx] = True
                discard_pile.append(hands[player][idx])
                hands[player][idx] = drawn_discard

        print(f"\n--- Player {player + 1}'s hand ---")
        print_hand(hands[player], revealed[player])
        print(f"Discard pile: {discard_pile[-1][0]} of {discard_pile[-1][1]}")  

        # Check if all cards are revealed
        if all(revealed[player]):
            print(f"\nPlayer {player + 1} has revealed all cards!")
            game_over = True
            break

# Scoring
print("\n--- Final Scores ---")
for player in range(num_players):
    print(f"Player {player + 1}'s hand:")
    print_hand(hands[player], [True] * 4)  # Reveal all cards
    score = calculate_score(hands[player])
    print(f"Score: {score}\n")

print("Thanks for playing!")
