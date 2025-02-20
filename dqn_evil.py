import numpy as np
import random
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
MEMORY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.0
EPSILON_DECAY = 0.9
LEARNING_RATE = 0.001
TARGET_UPDATE = 25  # How often to update target network

# Convert game state to tensor for neural network
# Converts 5x12 tensor into a single vector of length 60
def process_state(state):
    hand, recent_discards, opponent_revealed, position_weights = state
    
    # Calculate total size needed: 
    # 4 cards (hand) + 3 cards (discard history) + 4 cards (opponent) + 4 positions = 15 positions
    # Each card position has 12 possibilities (0-10 + unknown)
    state_tensor = torch.zeros(15, 12)
    
    # Encode hand cards (0-3)
    for i, card_val in enumerate(hand):
        if card_val == -1:  # Unknown card
            state_tensor[i, 11] = 1
        else:  # Known card value 0-10
            state_tensor[i, card_val] = 1
    
    # Encode recent discards (4-6)
    for i, card_val in enumerate(recent_discards):
        if card_val == -1:  # No card
            state_tensor[i+4, 11] = 1
        else:
            state_tensor[i+4, card_val] = 1
    
    # Encode opponent revealed cards (7-10)
    for i, card_val in enumerate(opponent_revealed):
        if card_val == -1:  # Unknown card
            state_tensor[i+7, 11] = 1
        else:
            state_tensor[i+7, card_val] = 1
            
    # Encode position weights (11-14)
    # Normalize weights to 0-10 range and add to tensor
    for i, weight in enumerate(position_weights):
        weight_idx = min(int(weight * 10), 10)
        state_tensor[i+11, weight_idx] = 1
    
    return state_tensor.flatten()

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
class ReplayMemory:
    # constructor initializes a double-ended queue with fixed maximum length
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    # Adds a new experience to memory
    # takes 5 parameters that represent one complete interaction
    # stores them as a tuple in memory
    """ 
    state: Current state (hand and discard card)
    action: What the agent did (0-8)
    reward: reward for that action
    next_state: next state
    done: if the game ended 
    """

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    # Randomly selects experiences for learning for some minimum batch size 
    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    # Returns current number of stored experiences, 
    # used to check if we have enough experiences to start learning
    def __len__(self):
        return len(self.memory)
    
def initial_peek(peeked_cards, player):
    """
    Allow player to peek at their two adjacent cards (last two cards in hand)
    This happens only once at the start of the game
    """
    # Record which cards were peeked at (positions 2 and 3)
    peeked_cards[player][2] = True
    peeked_cards[player][3] = True

# get state information ~ agent hand (revealed or not) and top of discard
def get_state(player, hands, revealed, peeked_cards, discard_pile):
    # Basic hand state
    agent_hand = []
    for i in range(4):
        if revealed[player][i]:
            agent_hand.append(hands[player][i][0])
        elif peeked_cards[player][i]:
            agent_hand.append(hands[player][i][0])
        else:
            agent_hand.append(-1)
            
    # Recent discard history (last 3 cards)
    recent_discards = []
    for i in range(min(3, len(discard_pile))):
        recent_discards.append(discard_pile[-(i+1)][0])
    while len(recent_discards) < 3:
        recent_discards.append(-1)
        
    # Opponent's revealed cards
    opponent_revealed = []
    for i in range(4):
        if revealed[0][i]:  # opponent is player 0
            opponent_revealed.append(hands[0][i][0])
        else:
            opponent_revealed.append(-1)
            
    # Position success weights
    total_success = sum(position_success) or 1
    position_weights = [p/total_success for p in position_success]
    
    return (tuple(agent_hand), tuple(recent_discards), 
            tuple(opponent_revealed), tuple(position_weights))

# select agent action based on max q value
def select_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 8)
    else:
        with torch.no_grad():
            state_tensor = process_state(state) # one hot encode state
            return policy_net(state_tensor).argmax().item()

# target net is the older copy of network for stable training

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    
    # random batch of experiences from  replay memory
    transitions = memory.sample(BATCH_SIZE)
    
    # transpose the batch and store as list
    # [(s1,a1,r1,s1',d1), (s2,a2,r2,s2',d2)...] to  [(s1,s2...), (a1,a2...), (r1,r2...)...]
    batch = list(zip(*transitions))
    
    # convert to tensors for neural network
    state_batch = torch.stack([process_state(s) for s in batch[0]])
    action_batch = torch.tensor(batch[1])
    reward_batch = torch.tensor(batch[2], dtype=torch.float32)
    next_state_batch = torch.stack([process_state(s) for s in batch[3]])
    done_batch = torch.tensor(batch[4], dtype=torch.float32)
    
    # compute current Q values
    # calls `forward` implicitly
    current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
    # compute next Q values using target network
    # calls `forward` implicitly
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]
    
    # Compute expected Q values
    expected_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)
    
    # compute loss
    loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
    
    # optimize model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

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
def update_deck(card, discard_pile, revealed, peeked_cards, idx, player, hands):
    """When a card is replaced, it becomes revealed and loses peeked status"""
    # Card being replaced becomes revealed
    revealed[player][idx] = True
    # Remove peeked status since position is now revealed
    peeked_cards[player][idx] = False
    
    discard_pile.append(hands[player][idx])
    hands[player][idx] = card

# Global position tracking
position_success = [0, 0, 0, 0]  # Success count for each position

def update_position_success(old_value, new_value, position):
    """Track successful card replacements"""
    if new_value < old_value:  # Better card was placed
        position_success[position] += 1
    elif new_value > old_value:  # Worse card was placed
        position_success[position] = max(0, position_success[position] - 1)

# rewards
def calculate_reward(hands, revealed, player, opponent_revealed, position):
    # Base reward from card values
    current_sum = sum(hands[player][i][0] for i in range(4) if revealed[player][i])
    
    # Opponent awareness reward
    opponent_sum = sum(val for val in opponent_revealed if val != -1)
    relative_performance = 0.2 * (opponent_sum - current_sum)
    
    # Position-based reward
    position_bonus = 0.1 * (position_success[position] / (sum(position_success) + 1))
    
    # King (0) bonus and Face card (10) penalty
    king_bonus = sum(2 for i, card in enumerate(hands[player]) if card[0] == 0 and revealed[player][i])
    face_penalty = sum(-1 for i, card in enumerate(hands[player]) if card[0] == 10 and revealed[player][i])
    
    return -(current_sum + relative_performance) + position_bonus + king_bonus + face_penalty

# training loop
def train_dqn(episodes=10000):
    # Initialize networks and optimizer
    input_size = 15 * 12  # 15 positions * 12 possibilities
    hidden_size = 256 
    output_size = 9  # Number of possible actions
    
    # Policy network: Active student constantly learning and changing
    # Target network: Teacher who updates their knowledge periodically but maintains stability
    policy_net = DQN(input_size, hidden_size, output_size) # main NN
    target_net = DQN(input_size, hidden_size, output_size) # target NN

    # copies all the weights and parameters from the policy network to the target network.
    target_net.load_state_dict(policy_net.state_dict())
    
    global position_success
    position_success = [0, 0, 0, 0]
    
    # adam optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    
    epsilon = EPSILON_START

    # Store known card values for each player
    #known_cards = {0: [], 1: []}  # Player -> [card1_value, card2_value]    
    
    for episode in tqdm(range(episodes)):
        deck = generate_deck()
        hands, deck = deal_cards(deck, 2)
        revealed = [[False] * 4 for i in range(2)]

        # Separate tracking for peeked cards
        peeked_cards = [[False, False, False, False], [False, False, False, False]]

        # Do initial peek for both players
        initial_peek(revealed, 0)
        initial_peek(revealed, 1)

        discard_pile = [deck.pop()] if deck else []
        
        game_over = False
        episode_loss = 0
        
        while not game_over:
            for player in range(2):
                if game_over:
                    break
                
                # Computer opponent's turn
                if player == 0:
                    
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
                                update_deck(card, discard_pile, revealed, peeked_cards, idx, player, hands)
                            else:
                                # No hidden cards, replace highest revealed card
                                update_deck(card, discard_pile, revealed, peeked_cards, highest_revealed_idx, player, hands)
                        else:
                            # Replace highest revealed card
                            update_deck(card, discard_pile, revealed, peeked_cards, highest_revealed_idx, player, hands)
                    
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
                                update_deck(card, discard_pile, revealed, peeked_cards, replace_idx, player, hands)
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
                                update_deck(card, discard_pile, revealed, peeked_cards, highest_revealed_idx, player, hands)
                            elif hidden_indices:
                                # Replace a random hidden card
                                idx = random.choice(hidden_indices)
                                update_deck(card, discard_pile, revealed, peeked_cards, idx, player, hands)
                            else:
                                # No good replacement options, discard
                                discard_pile.append(card)
                    
                    if all(revealed[0]):
                        game_over = True
                    continue
                
                # DQN Agent's turn
                current_state = get_state(1, hands, revealed, peeked_cards, discard_pile)
                action = select_action(current_state, policy_net, epsilon)
                
                # Execute action
                if action < 5 and deck:  # Draw from deck
                    card = deck.pop()
                    if action < 4:
                        # Store old value before updating
                        old_value = hands[1][action][0] if revealed[1][action] else -1
                        update_deck(card, discard_pile, revealed, peeked_cards, action, 1, hands)
                        # Update position success
                        update_position_success(old_value, card[0], action)
                    else:
                        discard_pile.append(card)
                elif action >= 5 and discard_pile:  # Draw from discard
                    card = discard_pile.pop()
                    position = action-5
                    old_value = hands[1][position][0] if revealed[1][position] else -1
                    update_deck(card, discard_pile, revealed, peeked_cards, position, 1, hands)
                    update_position_success(old_value, card[0], position)
                
                # Get opponent revealed cards for reward calculation
                opponent_revealed = [hands[0][i][0] if revealed[0][i] else -1 for i in range(4)]
                
                # Get reward and next state
                reward = calculate_reward(hands, revealed, 1, opponent_revealed, action if action < 4 else (action-5 if action >= 5 else -1))
                next_state = get_state(1, hands, revealed, peeked_cards, discard_pile)
                
                # Check if game is over
                done = all(revealed[1]) or not deck
                if done:
                    game_over = True
                    agent_score = sum(card[0] for card in hands[1])
                    opponent_score = sum(card[0] for card in hands[0])
                    reward += 5.0 if agent_score < opponent_score else -5.0
                
                # Store transition in memory
                memory.push(current_state, action, reward, next_state, done)
                
                # Optimize model
                loss = optimize_model(policy_net, target_net, memory, optimizer)
                if loss is not None:
                    episode_loss += (loss - episode_loss)/episode
        
        # Update target network
        #  we copy the parameters every 10 episodes,
        #  meaning the target network provides a more stable set of predictions for our learning updates

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        if episode % 500 == 0:
            print(f"Episode {episode}, Loss: {episode_loss:.4f}, Epsilon: {epsilon:.4f}")
    
    return policy_net

def test_dqn(policy_net, num_games=100):
    wins = 0
    total_score = 0
    total_opponent_score = 0

    # Store known card values for each player
    #known_cards = {0: [], 1: []}  # Player -> [card1_value, card2_value]  
    
    for game in range(num_games):
        deck = generate_deck()
        hands, deck = deal_cards(deck, 2)
        revealed = [[False] * 4 for i in range(2)]

        peeked_cards = [[False] * 4 for i in range(2)]


        # Do initial peek for both players
        initial_peek( revealed, 0)
        initial_peek( revealed, 1)

        discard_pile = [deck.pop()] if deck else []
        game_over = False
        
        while not game_over:
            for player in range(2):
                if game_over:
                    break
                
                if player == 0:  # Computer opponent
                    
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
                                update_deck(card, discard_pile, revealed, peeked_cards, idx, player, hands)
                            else:
                                # No hidden cards, replace highest revealed card
                                update_deck(card, discard_pile, revealed, peeked_cards, highest_revealed_idx, player, hands)
                        else:
                            # Replace highest revealed card
                            update_deck(card, discard_pile, revealed, peeked_cards, highest_revealed_idx, player, hands)
                    
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
                                update_deck(card, discard_pile, revealed, peeked_cards, replace_idx, player, hands)
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
                                update_deck(card, discard_pile, revealed, peeked_cards, highest_revealed_idx, player, hands)
                            elif hidden_indices:
                                # Replace a random hidden card
                                idx = random.choice(hidden_indices)
                                update_deck(card, discard_pile, revealed, peeked_cards, idx, player, hands)
                            else:
                                # No good replacement options, discard
                                discard_pile.append(card)
                    
                    if all(revealed[0]):
                        game_over = True
                    continue
                
                # DQN Agent's turn
                state = get_state(1, hands, revealed, peeked_cards, discard_pile)
                action = select_action(state, policy_net, 0)  # No exploration during testing
                
                if action < 5 and deck:
                    card = deck.pop()
                    if action < 4:
                        update_deck(card, discard_pile, revealed, peeked_cards, action, 1, hands)
                    else:
                        discard_pile.append(card)
                elif action >= 5 and discard_pile:
                    card = discard_pile.pop()
                    update_deck(card, discard_pile, revealed,peeked_cards,  action-5, 1, hands)
                
                if all(revealed[1]):
                    game_over = True
        
        agent_score = sum(card[0] for card in hands[1])
        opponent_score = sum(card[0] for card in hands[0])
        total_score += agent_score
        total_opponent_score += opponent_score
        
        if agent_score < opponent_score:
            wins += 1
    
    print(f"Average score over {num_games} games: {total_score/num_games:.2f}")
    print(f"Averag Opps score: {total_opponent_score/num_games:.2f}")
    print(f"Win rate: {wins}/{num_games} ({(wins/num_games)*100:.2f}%)")

# Train and test the DQN agent
policy_net = train_dqn(episodes=1001)
test_dqn(policy_net, num_games=1000)