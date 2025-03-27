# Testing pre-trained model against a rule based player

import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# generates deck
def generate_deck():
    values = list(range(1, 11)) + [10, 10, 0]  # 1-10, J=10, Q=10, K=0
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    deck = [(value, suit) for value in values for suit in suits]
    return deck

# deals cards to players
def deal_cards(deck, num_players):
    random.shuffle(deck)
    hands = []
    for i in range(num_players):
        hand = []
        for j in range(4): # Deal 4 cards per player
            hand.append(deck.pop())
        hands.append(hand)  
    return hands, deck

# updates decks (unknown + discrds) and hands
def update_deck(card, discard_pile, revealed, idx, player, hands):
    if not revealed[player][idx]:
        revealed[player][idx] = True
    discard_pile.append(hands[player][idx])
    hands[player][idx] = card


# get state information ~ agent hand (revealed or not) and top of discard
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

# select agent action based on max q value
def select_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 8)
    else:
        with torch.no_grad():
            state_tensor = process_state(state) # one hot encode state
            return policy_net(state_tensor).argmax().item() # max q value


# Convert game state to tensor for neural network
# Converts 5x12 tensor into a single vector of length 60
def process_state(state):
    
    hand, discard = state
    # Create a state vector: 4 cards + 1 discard card, each has 12 possibilities (0-10 + unknown)
    state_tensor = torch.zeros(5, 12)
    
    # Encode hand cards
    for i, card_val in enumerate(hand):
        if card_val == -1:  # Unknown card
            state_tensor[i, 11] = 1
        else:  # Known card value 0-10
            state_tensor[i, card_val] = 1
    
    # Encode discard card
    if discard == -1:  # No discard
        state_tensor[4, 11] = 1
    else:
        state_tensor[4, discard] = 1
    
    return state_tensor.flatten()

# intialize neural network 
class DQN(nn.Module):

    # linear layer
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    # forward pass function - Activation Function used is ReLu 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Load saved DQN model
def load_model(filename="golf_dqn_model.pth"):

    # Create a new instance of DQN with the same architecture
    # has to be the same size as trained model
    input_size = 60  
    hidden_size = 128
    output_size = 9  
    
    model = DQN(input_size, hidden_size, output_size) 

    # Load the saved state dictionary
    model.load_state_dict(torch.load(filename, weights_only=True))
    
    # Set the model to evaluation mode
    model.eval()
    
    print(f"Model loaded from {filename}")
    return model



def test_dqn(policy_net, num_games=100):
    wins = 0
    total_score = 0
    total_opponent_score = 0

    opp_end = 0
    agent_end = 0
    
    for game in range(num_games):
        deck = generate_deck()
        hands, deck = deal_cards(deck, 2)
        revealed = [[False] * 4 for i in range(2)]
        discard_pile = [deck.pop()] if deck else []
        game_over = False

        players = [0, 1]
        random.shuffle(players)
        
        while not game_over:
            for player in players:
                if game_over:
                    break
                
                if player == 0:  # Computer opponent ~ smart "evil" player
                    
                    # Rule 1: Check discard pile top card
                    discard_value = discard_pile[-1][0]
                    
                    if discard_value < 10:
                        # Draw from discard pile
                        card = discard_pile.pop()
                        
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
                                update_deck(card, discard_pile, revealed,  highest_revealed_idx, player, hands)
                        else:
                            # Replace highest revealed card
                            update_deck(card, discard_pile, revealed, highest_revealed_idx, player, hands)
                    
                    else:
                        # Rule 2: Draw from deck if discard >= 5
                        if not deck:
                            game_over = True
                            break

                        card = deck.pop()
                        
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
                        opp_end += 1 
                        break
                
                # DQN Agent's turn
                state = get_state(1, hands, revealed, discard_pile)
                action = select_action(state, policy_net, 0)  # No exploration during testing
                
                if action < 5 and deck:
                    card = deck.pop()
                    if action < 4:
                        update_deck(card, discard_pile, revealed, action, 1, hands)
                    else:
                        discard_pile.append(card)
                elif action >= 5 and discard_pile:
                    card = discard_pile.pop()
                    update_deck(card, discard_pile, revealed, action-5, 1, hands)
                
                if all(revealed[1]):
                    game_over = True
                    agent_end += 1
        
        agent_score = sum(card[0] for card in hands[1])
        opponent_score = sum(card[0] for card in hands[0])
        total_score += agent_score
        total_opponent_score += opponent_score
        
        if agent_score < opponent_score:
            wins += 1
    
    print(f"Average score over {num_games} games: {total_score/num_games:.2f}")
    print(f"Averag Opps score: {total_opponent_score/num_games:.2f}")
    print(f"Win rate: {wins}/{num_games} ({(wins/num_games)*100:.2f}%)")
    print(f"Opp revealed: {opp_end}, agent revealed: {agent_end}")


policy_net = load_model("models/dqn_golf_1M.pth")

test_dqn(policy_net, num_games=10000)