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
def print_hand(hand, revealed, player, discard_pile):

    agent = 'Computer' if player == 1 else 'Player'
    print(f"\n--- {agent}'s hand ---")
    for i, card in enumerate(hand):
        if revealed[i]:
            print(f"[{i + 1}] {card[0]} of {card[1]}")
        else:
            print(f"[{i + 1}] [Hidden]")
    print(f"Discard pile: {discard_pile[-1][0]} of {discard_pile[-1][1]}")  


def update_deck(card, discard_pile, revealed, idx, player, hands):
    if not revealed[player][idx]:
        revealed[player][idx] = True
    discard_pile.append(hands[player][idx])
    hands[player][idx] = card

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

action = None
draw_action = None
keep_or_discard_action = None


# Main game loop
game_over = False
while not game_over:
    action = None
    draw_action = None
    keep_or_discard_action = None
    for player in range(num_players):

        if player == 1:  # Automated player
            print_hand(hands[player], revealed[player], player, discard_pile)
            
            # Rule 1: Check discard pile top card
            discard_value = discard_pile[-1][0]
            
            if discard_value < 5:
                # Draw from discard pile
                card = discard_pile.pop()
                print(f"Computer drew: {card[0]} of {card[1]} from discard pile.")

                # Find highest revealed card or pick hidden card
                highest_revealed_idx = -1
                highest_revealed_value = -1
                hidden_indices = []
                
                for i in range(4):
                    if revealed[player][i]:
                        if hands[player][i][0] > highest_revealed_value:
                            highest_revealed_value = hands[player][i][0]
                            highest_revealed_idx = i
                    else:
                        hidden_indices.append(i)
                
                # Rule 4: If card is less than 5 and not less than revealed cards
                if highest_revealed_value == -1 or card[0] >= highest_revealed_value:
                    # Replace a hidden card if available
                    if hidden_indices:
                        idx = random.choice(hidden_indices)
                        update_deck(card, discard_pile, revealed, idx, player, hands)
                    else:
                        # No hidden cards, replace highest revealed card
                        update_deck(card, discard_pile, revealed, highest_revealed_idx, player, hands)
                else:
                    # Replace highest revealed card
                    update_deck(card, discard_pile, revealed, highest_revealed_idx, player, hands)
            
            else:
                # Rule 2: Draw from deck if discard >= 5
                if not deck:
                    game_over = True
                    break

                card = deck.pop()
                print(f"Computer drew: {card[0]} of {card[1]}")

                # Rule 3: Handle card drawn from deck
                if card[0] > 5:
                    # Find if there's a higher revealed card to replace
                    found_higher = False
                    replace_idx = -1
                    for i in range(4):
                        if revealed[player][i] and hands[player][i][0] > card[0]:
                            found_higher = True
                            replace_idx = i
                            break
                    
                    if found_higher:
                        update_deck(card, discard_pile, revealed, replace_idx, player, hands)
                    else:
                        # Discard if no higher card found
                        discard_pile.append(card)
                else:
                    # Card is <= 5, try to replace highest revealed card or hidden card
                    highest_revealed_idx = -1
                    highest_revealed_value = -1
                    hidden_indices = []
                    
                    for i in range(4):
                        if revealed[player][i]:
                            if hands[player][i][0] > highest_revealed_value:
                                highest_revealed_value = hands[player][i][0]
                                highest_revealed_idx = i
                        else:
                            hidden_indices.append(i)
                    
                    if highest_revealed_value > card[0]:
                        # Replace highest revealed card
                        update_deck(card, discard_pile, revealed, highest_revealed_idx, player, hands)
                    elif hidden_indices:
                        # Replace a random hidden card
                        idx = random.choice(hidden_indices)
                        update_deck(card, discard_pile, revealed, idx, player, hands)
                    else:
                        # No good replacement options, discard
                        discard_pile.append(card)

        else:
            
            print_hand(hands[player], revealed[player], player, discard_pile)

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

                        print(hands[player][idx])
                        update_deck(drawn_card, discard_pile, revealed, idx, player, hands)

                    elif keep_or_discard_action == '2':
                        discard_pile.append(drawn_card)

                elif draw_action == '2':
                    drawn_discard = discard_pile.pop()
                    idx = int(input("Choose a card from your hand to replace with card drawn from top of discard (1-4): ")) - 1
                    update_deck(drawn_discard, discard_pile, revealed, idx, player, hands)

        print_hand(hands[player], revealed[player], player, discard_pile)

        # Check if all cards are revealed
        if all(revealed[player]):
            print(f"\nPlayer {player + 1} has revealed all cards!")
            game_over = True
            break

# Scoring
print("\n--- Final Scores ---")
for player in range(num_players):
    print(f"Player {player + 1}'s hand:")
    print_hand(hands[player], [True] * 4, player, discard_pile)  # Reveal all cards
    score = calculate_score(hands[player])
    print(f"Score: {score}\n")

print("Thanks for playing!")
