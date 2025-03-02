import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

# Q-table with default value 0
Q = defaultdict(float)
alpha = 0.1  # learning rate
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
def choose_action(state, epsilon, net):

    if random.random() < epsilon:
        return random.randint(0, 8)
    else:

        state_tensor = process_state(state)

        with torch.no_grad():
            q_values = net(state_tensor)

        action = torch.argmax(q_values).item()

        return action
        

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
    return -current_sum/40


def create_network(input_dim, hidden_dim, output_dim):
    
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)   
    )


def process_state(state):
    hand, discard = state
    # 12 possibilities for each position (0-10 plus unknown)
    state_tensor = torch.zeros(5, 12)  # 5 positions (4 hand + 1 discard), 12 values each
    
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
    
    return state_tensor.flatten()  # Return flat tensor of size 60

def store_transition(buffer, state, action, reward, next_state, max_size=10000):
    buffer.append((state, action, reward, next_state))
    if len(buffer) > max_size:
        buffer.pop(0)
    return buffer

def sample_batch(buffer, batch_size=32):
    batch = random.sample(buffer, min(batch_size, len(buffer)))
    return zip(*batch)  # unzip into separate lists

def train_agent(net, episodes=10000):
    global Q, alpha
    final_reward = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=10000)

    episode_losses = []
    replay_buffer = []

    for episode in tqdm(range(episodes)):

        # alpha decay
        current_alpha = max(0.0001, alpha * np.exp(-0.01 * episode))
        current_epsilon = max(0.01, epsilon * np.exp(-0.001 * episode))
        deck = generate_deck()
        hands, deck = deal_cards(deck, 2)

        revealed = [[False, False, False, False], [False, False, False, False] ]  
        discard_pile = [deck.pop()] if deck else []

        game_over = False
        while not game_over:
            for player in range(2):
                if game_over:
                    break
                
                # computer opponent
                if player == 0: 
                    if not deck:
                        game_over = True
                        break

                    draw_action = random.choice(['deck', 'discard'])
                    if draw_action == 'deck':
                        if not deck:
                            game_over = True
                            break
                        card = deck.pop()

                        # randomly draw or discard with 0.5 prob
                        if random.random() < 0.5 and discard_pile:
                            idx = random.randint(0, 3)
                            update_deck(card, discard_pile, revealed, idx, 0, hands)
                        else:
                            discard_pile.append(card)
                    else:

                        if not discard_pile:
                            continue
                        card = discard_pile.pop()
                        idx = random.randint(0, 3)
                        update_deck(card, discard_pile, revealed, idx, 0, hands)
                    
                    if all(revealed[0]):
                        game_over = True
                    continue

                # RL Agent's turn (player 1)
                current_state = get_state(1, hands, revealed, discard_pile)
                action = choose_action(current_state, current_epsilon, net)

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

                    # scores are total sum of cards in hand
                    agent_score = sum(card[0] for card in hands[1])
                    opponent_score = sum(card[0] for card in hands[0])
                    
                    # If opponent has finished
                    if agent_score < opponent_score:
                        final_reward = 100  # WIN 
                    else:
                        final_reward = -100 # LOSE

            # Store transition in replay buffer
            replay_buffer.append((current_state, action, reward, next_state))
            if len(replay_buffer) > 10000:  # Keep buffer size fixed
                replay_buffer.pop(0)

            if len(replay_buffer) > 64:  # Only train when we have enough samples
                # Sample random batch
                batch = random.sample(replay_buffer, 64)
                states, actions, rewards, next_states = zip(*batch)
                
                # Convert to tensors
                state_batch = torch.stack([process_state(s) for s in states])
                next_state_batch = torch.stack([process_state(s) for s in next_states])
                action_batch = torch.LongTensor(actions)
                reward_batch = torch.FloatTensor(rewards)

                # If terminal state, update with final reward
                if game_over:
                    reward_batch = torch.where(
                        torch.arange(len(rewards)) == len(rewards)-1,
                        torch.tensor(final_reward),
                        reward_batch
                    )

                # Get Q-values
                current_q_batch = net(state_batch)
                next_q_batch = net(next_state_batch)

                # Compute target Q values
                target_q_batch = current_q_batch.clone()
                
                # Update only the actions that were taken
                target_q_batch[range(len(actions)), action_batch] = reward_batch + gamma * torch.max(next_q_batch, dim=1)[0]

                # Update network
                optimizer.zero_grad()
                loss = criterion(current_q_batch, target_q_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                episode_losses.append(loss.item())

        if episode % 1000 == 0:
            if len(episode_losses) > 0:  # Only calculate if we have losses
                avg_loss = sum(episode_losses[-min(1000, len(episode_losses)):]) / min(1000, len(episode_losses))
                print(f"Episode {episode}, Loss: {avg_loss:.4f}, epsilon: {current_epsilon:.4f}, lr: {optimizer.param_groups[0]['lr']:.4f}")
            else:
                print(f"Episode {episode}, No losses yet, epsilon: {current_epsilon:.4f}")



def test_agent(net, num_games=100):
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
                if player == 0: 
                    if not deck:
                        game_over = True
                        break

                    draw_action = random.choice(['deck', 'discard'])
                    if draw_action == 'deck':
                        if not deck:
                            game_over = True
                            break
                        card = deck.pop()

                        # randomly draw or discard with 0.5 prob
                        if random.random() < 0.5 and discard_pile:
                            idx = random.randint(0, 3)
                            update_deck(card, discard_pile, revealed, idx, 0, hands)
                        else:
                            discard_pile.append(card)
                    else:

                        if not discard_pile:
                            continue
                        card = discard_pile.pop()
                        idx = random.randint(0, 3)
                        update_deck(card, discard_pile, revealed, idx, 0, hands)
                    
                    if all(revealed[0]):
                        game_over = True
                    continue

                # RL Agent's turn ~ our hero
                state = get_state(1, hands, revealed, discard_pile)
                action = choose_action(state, 0, net)  # Greedy policy
                
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
    print(f"Number of wins: {wins}/{num_games}")



input = (4 + 1) * 12
inside_dim = 64
output = 9 # for each action

net = create_network(input, inside_dim, output)

# Train and test the agent
train_agent(net, episodes=5001)
test_agent(net, num_games=100)