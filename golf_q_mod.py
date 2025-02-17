import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

# Q-table with default value 0
Q = defaultdict(float)
alpha = 1.0  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.5  # exploration rate

def get_state(player, hands, revealed, discard_pile):
    agent_hand = []
    for i in range(4):
        if revealed[player][i]:
            agent_hand.append(hands[player][i][0])
        else:
            agent_hand.append(-1)
    if discard_pile:
        discard_top = discard_pile[-1][0]
    else:
        discard_top = -1
    return (tuple(agent_hand), discard_top)

# epsilon - greedy actions
def choose_action(state, epsilon):

    if random.random() < epsilon:
        return random.randint(0, 8)
    else:
        max_q = -float('inf')
        best_actions = []

        # epsilon-greedy policy
        for action in range(9):
            q = Q[(state, action)]

            if q > max_q:
                max_q = q
                best_actions = [action]
            elif q == max_q:
                best_actions.append(action)

        if best_actions:
            return random.choice(best_actions)
        else:
            return random.randint(0, 8)
        

def generate_deck():
    values = list(range(1, 11)) + [10, 10, 0]  # 1-10, J=10, Q=10, K=0
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

    deck = []
    for value in values:
        for suit in suits:
            deck.append((value, suit))
    return deck

def deal_cards(deck, num_players):
    random.shuffle(deck)
    hands = []
    for i in range(num_players):
        hand = []
        for j in range(4): # Deal 4 cards per player
            hand.append(deck.pop())
        hands.append(hand)  
    return hands, deck

# updates hands and discards
def update_deck(card, discard_pile, revealed, idx, player, hands):
    if not revealed[player][idx]:
        revealed[player][idx] = True
    discard_pile.append(hands[player][idx])
    hands[player][idx] = card

# calculate immediate reward based on current known cards
def calculate_reward(hands, revealed, player):
    
    current_sum = sum(hands[player][i][0] for i in range(4) if revealed[player][i])
    
    # negative reward for minimizing sum
    return -current_sum

def train_agent(episodes=10000):
    global Q, alpha
    final_reward = 0

    for episode in (range(episodes)):

        # alpha decay
        current_alpha = max(0.0001, alpha * np.exp(-0.00005 * episode))
        current_epsilon = max(0.01, epsilon * np.exp(-0.00005 * episode))
        deck = generate_deck()
        hands, deck = deal_cards(deck, 2)

        revealed = [[False, False, False, False], [False, False, False, False] ]  
        discard_pile = [deck.pop()] if deck else []

        #print(discard_pile)

        game_over = False
        while not game_over:
            for player in range(2):
                if game_over:
                    break
                
                # computer opponent
                if player == 0:  # Automated player
                    #print_hand(hands[player], revealed[player], player, discard_pile)
                    
                    # Rule 1: Check discard pile top card
                    discard_value = discard_pile[-1][0]
                    
                    if discard_value < 10:
                        # Draw from discard pile
                        card = discard_pile.pop()
                        #print(f"Computer drew: {card[0]} of {card[1]} from discard pile.")

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
                        #print(f"Computer drew: {card[0]} of {card[1]}")

                        # Rule 3: Handle card drawn from deck
                        if card[0] > 10:
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
                    
                    if all(revealed[0]):
                        game_over = True
                    continue

                # RL Agent's turn (player 1)
                current_state = get_state(1, hands, revealed, discard_pile)
                action = choose_action(current_state, current_epsilon)

                # draw from deck
                if action < 5:  
                    if not deck:
                        game_over = True
                        break
                    card = deck.pop()

                    # replace card
                    if action < 4:  
                        update_deck(card, discard_pile, revealed, action, 1, hands)

                    else:  # discard
                        discard_pile.append(card)

                else: 
                    # Draw from discard
                    if not discard_pile:
                        continue
                    card = discard_pile.pop()
                    idx = action - 5
                    update_deck(card, discard_pile, revealed, idx, 1, hands)

                reward = calculate_reward(hands, revealed, 1)
                next_state = get_state(1, hands, revealed, discard_pile)

                if all(revealed[1]):
                    game_over = True

            # Terminal state handling (moved outside player loop)
            if game_over:
                # scores are total sum of cards in hand
                agent_score = sum(card[0] for card in hands[1])
                opponent_score = sum(card[0] for card in hands[0])
                
                # If opponent has finished
                if agent_score < opponent_score:
                    final_reward = 100  # WIN
                else:
                    final_reward = -100 # LOSE
                
                # update Q-value with final reward 
                Q[(current_state, action)] += current_alpha * (final_reward - Q[(current_state, action)])
            else:

                # standard Q-learning update for non-terminal states
                max_next_q = max(Q[(next_state, a)] for a in range(9))
                Q[(current_state, action)] += current_alpha * (reward + gamma * max_next_q - Q[(current_state, action)])

        if episode % 5000 == 0:
            print(f"Episode {episode}, alpha: {current_alpha:.4f}, epsilon: {current_epsilon:.4f}")



def test_agent(num_games=100):
    total_score = 0
    wins = 0  # Track number of wins

    for game in range(num_games):
        deck = generate_deck()
        hands, deck = deal_cards(deck, 2)
        revealed = [[False, False, False, False], [False, False, False, False] ] 
        discard_pile = [deck.pop()] if deck else []
        game_over = False

        while not game_over:
            for player in range(2):
                if game_over:
                    break
                
                # computer opponent ~ evil
                # computer opponent
                if player == 0:  # Automated player
                    #print_hand(hands[player], revealed[player], player, discard_pile)
                    
                    # Rule 1: Check discard pile top card
                    discard_value = discard_pile[-1][0]
                    
                    if discard_value < 5:
                        # Draw from discard pile
                        card = discard_pile.pop()
                        #print(f"Computer drew: {card[0]} of {card[1]} from discard pile.")

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
                        #print(f"Computer drew: {card[0]} of {card[1]}")

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
                    
                    if all(revealed[0]):
                        game_over = True
                    continue

                # RL Agent's turn ~ our hero
                state = get_state(1, hands, revealed, discard_pile)
                action = choose_action(state, 0)  # Greedy policy
                
                if action < 5:
                    if deck:
                        card = deck.pop()
                        if action < 4:
                            update_deck(card, discard_pile, revealed, action, 1, hands)
                        else:
                            discard_pile.append(card)
                elif discard_pile:
                    card = discard_pile.pop()
                    update_deck(card, discard_pile, revealed, action-5, 1, hands)

                if all(revealed[1]):
                    game_over = True

        # Calculate final scores
        agent_score = sum(card[0] for card in hands[1])
        opponent_score = sum(card[0] for card in hands[0])
        total_score += agent_score

        # Check if the agent won
        if agent_score < opponent_score:
            wins += 1

    print(f"Average score over {num_games} games: {total_score/num_games}")
    print(f"Number of wins: {wins}/{num_games} ({wins/num_games*100}%)")



# Train and test the agent
train_agent(episodes=1000001)
test_agent(num_games=500)