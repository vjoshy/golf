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
INITIAL_LR = 0.01
LR_DECAY = 0.99
MIN_LR = 0.0001
TARGET_UPDATE = 10  # How often to update target network

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

class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
def update_deck(card, discard_pile, revealed, idx, player, hands):
    if not revealed[player][idx]:
        revealed[player][idx] = True
    discard_pile.append(hands[player][idx])
    hands[player][idx] = card

# rewards
def calculate_reward(hands, revealed, player):
    current_sum = sum(hands[player][i][0] for i in range(4) if revealed[player][i])
    # Add intermediate rewards based on card values
    card_quality = sum(1 for card in hands[player] if card[0] == 0)  # Bonus for Kings
    penalty = sum(1 for card in hands[player] if card[0] == 10)  # Penalty for Q/J
    return (-current_sum/40) + (card_quality * 0.5) - (penalty * 0.3)

# training loop
def train_dqn(episodes=10000):
    # Initialize networks and optimizer
    input_size = 60  # 5 positions * 12 possibilities
    hidden_size = 128 
    output_size = 9  # Number of possible actions
    
    # Policy network: Active student constantly learning and changing
    # Target network: Teacher who updates their knowledge periodically but maintains stability
    policy_net = DQN(input_size, hidden_size, output_size) # main NN
    target_net = DQN(input_size, hidden_size, output_size) # target NN

    # copies all the weights and parameters from the policy network to the target network.
    target_net.load_state_dict(policy_net.state_dict())
    
    # adam optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    memory = ReplayMemory(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    
    for episode in tqdm(range(episodes)):
        deck = generate_deck()
        hands, deck = deal_cards(deck, 2)
        revealed = [[False] * 4 for i in range(2)]
        discard_pile = [deck.pop()] if deck else []
        
        game_over = False
        episode_loss = 0
        
        while not game_over:
            for player in range(2):
                if game_over:
                    break
                
                # Computer opponent's turn
                if player == 0:
                    if not deck:
                        game_over = True
                        break
                        
                    if random.choice(['deck', 'discard']) == 'deck' and deck:
                        card = deck.pop()
                        if random.random() < 0.5 and discard_pile:
                            update_deck(card, discard_pile, revealed, random.randint(0, 3), 0, hands)
                        else:
                            discard_pile.append(card)
                    elif discard_pile:
                        card = discard_pile.pop()
                        update_deck(card, discard_pile, revealed, random.randint(0, 3), 0, hands)
                        
                    if all(revealed[0]):
                        game_over = True
                    continue
                
                # DQN Agent's turn
                current_state = get_state(1, hands, revealed, discard_pile)
                action = select_action(current_state, policy_net, epsilon)
                
                # Execute action
                if action < 5 and deck:  # Draw from deck
                    card = deck.pop()
                    if action < 4:
                        update_deck(card, discard_pile, revealed, action, 1, hands)
                    else:
                        discard_pile.append(card)
                elif action >= 5 and discard_pile:  # Draw from discard
                    card = discard_pile.pop()
                    update_deck(card, discard_pile, revealed, action-5, 1, hands)
                
                # Get reward and next state
                reward = calculate_reward(hands, revealed, 1)
                next_state = get_state(1, hands, revealed, discard_pile)
                
                # Check if game is over
                done = all(revealed[1]) or not deck
                if done:
                    game_over = True
                    agent_score = sum(card[0] for card in hands[1])
                    opponent_score = sum(card[0] for card in hands[0])
                    reward = 1.0 if agent_score < opponent_score else -1.0
                
                # Store transition in memory
                memory.push(current_state, action, reward, next_state, done)
                
                # Optimize model
                loss = optimize_model(policy_net, target_net, memory, optimizer)
                if loss is not None:
                    episode_loss += (loss - episode_loss)/episode

                    if optimizer.param_groups[0]['lr'] > MIN_LR:
                        scheduler.step()
        
        # Update target network
        #  we copy the parameters every 10 episodes,
        #  meaning the target network provides a more stable set of predictions for our learning updates

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        if episode % 500 == 0:
            print(f"Episode {episode}, Loss: {episode_loss:.4f}, Epsilon: {epsilon:.4f}, LR: {optimizer.param_groups[0]['lr']:.4f}")
    
    return policy_net

def test_dqn(policy_net, num_games=100):
    wins = 0
    total_score = 0
    total_opponent_score = 0
    
    for game in range(num_games):
        deck = generate_deck()
        hands, deck = deal_cards(deck, 2)
        revealed = [[False] * 4 for i in range(2)]
        discard_pile = [deck.pop()] if deck else []
        game_over = False
        
        while not game_over:
            for player in range(2):
                if game_over:
                    break
                
                if player == 0:  # Computer opponent
                    if random.choice([0, 1]) == 0 and deck:
                        card = deck.pop()
                        if random.random() < 0.5:
                            update_deck(card, discard_pile, revealed, random.randint(0, 3), 0, hands)
                        else:
                            discard_pile.append(card)
                    elif discard_pile:
                        card = discard_pile.pop()
                        update_deck(card, discard_pile, revealed, random.randint(0, 3), 0, hands)
                    
                    if all(revealed[0]):
                        game_over = True
                    continue
                
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