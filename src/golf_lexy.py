import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm 

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


def game_setup():
    deck = generate_deck()
    hands, deck = deal_cards(deck, 2)
    revealed = [[False, False, False, False], [False, False, False, False] ] # start with all cards unrevealed 
    discard_pile = [deck.pop()] if deck else [] # pull top card
    game_over = False
    return deck,discard_pile,revealed,hands,game_over


def update_deck(card, discard_pile, revealed, idx, player, hands):
    if not revealed[player][idx]:
        revealed[player][idx] = True
    discard_pile.append(hands[player][idx])
    hands[player][idx] = card


def get_state(player, hands, revealed, discard_pile):   
    agent_hand = []
    for i in range(4): 
        if revealed[player][i]: # obtain revealed card values
            agent_hand.append(hands[player][i][0])
        else:
            agent_hand.append(-1)
    if discard_pile: 
        discard_top = discard_pile[-1][0] # card from top of discard pile
    else:
        discard_top = -1 
    return (tuple(agent_hand), discard_top) # current state


# random player
def opponent_turn(deck,discard_pile,revealed,hands):
    loop=True
    game_over=False
    while loop==True:    
        if not deck: # stop game when deck is empty
            game_over=True
            break
        draw_action = random.choice(['deck', 'discard'])
        if draw_action == 'deck': # choose card from deck
            if not deck:
                game_over=True
                break
            card = deck.pop()

            if random.random() < 0.5 and discard_pile:
                idx = random.randint(0, 3)
                update_deck(card, discard_pile, revealed, idx, 0, hands)
            else:
                discard_pile.append(card)
        else:
            card = discard_pile.pop()
            idx = random.randint(0, 3)
            update_deck(card, discard_pile, revealed, idx, 0, hands)
        
        if all(revealed[0]):
            game_over=True
            break 
        loop=False

    op_hand = []
    for i in range(4): 
        if revealed[0][i]: # obtain revealed card values
            op_hand.append(hands[0][i][0])
        else:
            op_hand.append(-1)

    return op_hand, game_over


def get_action(epsilon, params, state, deck, discard_pile, revealed, hands):
    if random.random() < epsilon:  
        return random.randint(0, 8)
    
    # calculate q-values for all actions
    q_vals = np.zeros(9)
    for action in range(9):
        
        q_vals[action], grad_q = q_function(params, state, action)
    
    # choose best action 
    max_q = np.max(q_vals)
    best_actions = np.where(q_vals == max_q)[0]
    action = np.random.choice(best_actions)
    
    return action

def compute_next_state(action,deck,discard_pile,revealed,hands):
    game_over=False
    loop=True
    while loop==True:
        if action < 5:  
            if not deck:
                game_over=True
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
                game_over=True
                break
            card = discard_pile.pop()
            idx = action - 5
            update_deck(card, discard_pile, revealed, idx, 1, hands) 
        loop=False
    next_state=get_state(1, hands, revealed, discard_pile)
    if all(revealed[0]) or all (revealed[1]):
        game_over = True
    return next_state,game_over


def q_function(params, state, action):
    # create a feature vector that depends on both state and action
    # 4 cards + discard + 9 actions, we'll use a larger feature vector
    # 15 state features * 9 actions
    num_features = 15 * 9  
    features = np.zeros(num_features)
    
    
    # get state information
    vals = np.array(state[0])
    expected_vals = np.where(vals == -1, 6, vals) / 10  # Normalize values and handle hidden cards (this is from chat)
    discard = state[1] / 10 if state[1] != -1 else 0.6  # Normalize discard  (this is from chat)
    
    # Base index for the current action 
    action_offset = action * 15
    
    # filling in state features for the chosen action
    for card_idx in range(4):
        val = expected_vals[card_idx]
        if val <= 0.3:
            features[action_offset + 3*card_idx] = 1  # low
        elif 0.4 <= val <= 0.7:
            features[action_offset + 3*card_idx + 1] = 1  # med
        else:
            features[action_offset + 3*card_idx + 2] = 1  # high
    
    # discard pile features
    if discard <= 0.3:
        features[action_offset + 12] = 1
    elif 0.4 <= discard <= 0.7:
        features[action_offset + 13] = 1
    else:
        features[action_offset + 14] = 1
    
    # bias feature for each action
    features[action_offset + 14] = 1  # Use the last feature as a bias term
    
    q = np.dot(features, params)
    grad_q = features
    
    return q, grad_q

# Chat
def calculate_reward(state, next_state, revealed):
    old_hand = np.array(state[0])
    new_hand = np.array(next_state[0])

    if np.array_equal(old_hand, new_hand):
        return 0  # No change, no reward
    
    changed_card_idx = np.where(new_hand != old_hand)[0]
    if len(changed_card_idx) == 0:
        return 0  # Edge case: No detected change

    # Estimate unseen cards as 6
    old_hand = np.where(old_hand == -1, 6, old_hand)
    new_hand = np.where(new_hand == -1, 6, new_hand)

    # Compute difference in hand sum
    reward = np.sum(old_hand) - np.sum(new_hand)

    return reward   # Normalize?



def calculate_term_reward(hands):
    op_vals=[num for num,suit in hands[0]]
    agent_vals=[num for num,suit in hands[1]]
    op_sum=sum(op_vals)
    agent_sum=sum(agent_vals)
    if agent_sum<op_sum:
        term_reward=10    
    elif agent_sum>op_sum:
        term_reward=-10
    else:
        term_reward=0
    return term_reward



# helper function to get opponent hand
def get_opponent_hand(revealed, hands):
    op_hand = []
    for i in range(4):
        if revealed[0][i]: 
            op_hand.append(hands[0][i][0])
        else:
            op_hand.append(-1)
    return op_hand

# this is the "smart/evil" player 
# used chat to refactor my code to fit lexy's
def smart_opponent_turn(deck, discard_pile, revealed, hands):
    game_over = False
    
    # Check if deck is empty
    if not deck:
        game_over = True
        return get_opponent_hand(revealed, hands), game_over
    
    # Rule 1: Check discard pile top card
    if discard_pile:
        discard_value = discard_pile[-1][0]
        
        if discard_value < 5:  # Low value card in discard - take it
            # Draw from discard pile
            card = discard_pile.pop()
            
            # Find highest revealed card or pick hidden card
            highest_revealed_idx = -1
            highest_revealed_value = -1
            hidden_indices = []
            
            for i in range(4):
                if revealed[0][i]:
                    if hands[0][i][0] > highest_revealed_value:
                        highest_revealed_value = hands[0][i][0]
                        highest_revealed_idx = i
                else:
                    hidden_indices.append(i)
            
            # Rule 4: If card is less than 5 and not less than revealed cards
            if highest_revealed_value == -1 or card[0] >= highest_revealed_value:
                # Replace a hidden card if available
                if hidden_indices:
                    idx = random.choice(hidden_indices)
                    update_deck(card, discard_pile, revealed, idx, 0, hands)
                else:
                    # No hidden cards, replace highest revealed card
                    update_deck(card, discard_pile, revealed, highest_revealed_idx, 0, hands)
            else:
                # Replace highest revealed card
                update_deck(card, discard_pile, revealed, highest_revealed_idx, 0, hands)
        else:
            # Rule 2: Draw from deck if discard >= 5
            if not deck:
                game_over = True
                return get_opponent_hand(revealed, hands), game_over
                
            card = deck.pop()
            
            # Rule 3: Handle card drawn from deck
            if card[0] > 5:  # High card
                # Find if there's a higher revealed card to replace
                found_higher = False
                replace_idx = -1
                for i in range(4):
                    if revealed[0][i] and hands[0][i][0] > card[0]:
                        found_higher = True
                        replace_idx = i
                        break
                
                if found_higher:
                    update_deck(card, discard_pile, revealed, replace_idx, 0, hands)
                else:
                    # Discard if no higher card found
                    discard_pile.append(card)
            else:  # Low card (<=5)
                # Card is <= 5, try to replace highest revealed card or hidden card
                highest_revealed_idx = -1
                highest_revealed_value = -1
                hidden_indices = []
                
                for i in range(4):
                    if revealed[0][i]:
                        if hands[0][i][0] > highest_revealed_value:
                            highest_revealed_value = hands[0][i][0]
                            highest_revealed_idx = i
                    else:
                        hidden_indices.append(i)
                
                if highest_revealed_value > card[0]:
                    # Replace highest revealed card
                    update_deck(card, discard_pile, revealed, highest_revealed_idx, 0, hands)
                elif hidden_indices:
                    # Replace a random hidden card
                    idx = random.choice(hidden_indices)
                    update_deck(card, discard_pile, revealed, idx, 0, hands)
                else:
                    # No good replacement options, discard
                    discard_pile.append(card)
    else:
        # No discard pile, draw from deck
        if not deck:
            game_over = True
            return get_opponent_hand(revealed, hands), game_over
            
        card = deck.pop()
        
        # Similar logic as above for using the card
        if card[0] <= 5:  # Low card
            # Try to replace a hidden card
            hidden_indices = [i for i in range(4) if not revealed[0][i]]
            if hidden_indices:
                idx = random.choice(hidden_indices)
                update_deck(card, discard_pile, revealed, idx, 0, hands)
            else:
                # Find highest revealed card
                highest_idx = max(range(4), key=lambda i: hands[0][i][0] if revealed[0][i] else -1)
                if hands[0][highest_idx][0] > card[0]:
                    update_deck(card, discard_pile, revealed, highest_idx, 0, hands)
                else:
                    discard_pile.append(card)
        else:
            # High card - only replace if we find a higher revealed card
            replace_idx = -1
            for i in range(4):
                if revealed[0][i] and hands[0][i][0] > card[0]:
                    replace_idx = i
                    break
            
            if replace_idx >= 0:
                update_deck(card, discard_pile, revealed, replace_idx, 0, hands)
            else:
                discard_pile.append(card)
    
    # Check if game is over
    if all(revealed[0]):
        game_over = True
    
    return get_opponent_hand(revealed, hands), game_over


def generate_episode(epsilon, params):
    deck, discard_pile, revealed, hands, game_over = game_setup()
    current_player = (random.random() < 0.5) + 1

    state_history = []
    action_history = []
    reward_history = []

    while not game_over:
        if current_player == 1:

            state = get_state(current_player, hands, revealed, discard_pile)
            state_history.append(state)

            action = get_action(epsilon, params, state, deck, discard_pile, revealed, hands)
            action_history.append(action)
            

            new_state, game_over = compute_next_state(action, deck, discard_pile, revealed, hands)
            
            reward = calculate_reward(state, new_state, revealed)
            reward_history.append(reward)
        else:
            if random.random() < 0.1:
                op_state, game_over = smart_opponent_turn(deck, discard_pile, revealed, hands)
            else:
                op_state, game_over = opponent_turn(deck, discard_pile, revealed, hands)

        current_player = 3 - current_player
    
    # game over
    state_history.append(new_state)  
    terminal_reward = calculate_term_reward(hands)
    reward_history.append(terminal_reward)
    
    return state_history, action_history, reward_history


def train_agent(episodes= 10001):
    alpha=0.1
    epsilon=0.5
    gamma = 0.85

    # num_params=19
    num_params=15 * 9
    # num_params=5

    
    # params=np.zeros(num_params)
    params=np.random.uniform(-0.1, 0.1, size=num_params)
    
    # params=np.ones(num_params)


    loss=np.zeros(episodes)
    total_return=np.zeros(episodes)
    epsilon_vec=np.zeros(episodes)
    avg_returns=np.zeros(episodes)
    avg_apprx_returns=np.zeros(episodes)

    for episode in tqdm(range(episodes)):
        epsilon = max(epsilon * 0.999, 0.01)
        alpha = max(alpha * 0.9999, 0.001) 

        states, actions, rewards=generate_episode(epsilon,params)
        tmax=len(states) - 1

        apprx_returns=np.zeros(tmax+1)
        epsilon_vec[episode]=epsilon
        total_return[episode]=sum(rewards)
        returns_per_state=np.zeros(tmax+1)

        for t in range(tmax):
            state=states[t]
            action = actions[t]

            # ERROR HERE!
            G = 0
            for k in range(t, tmax):  
                G += (gamma ** (k - t)) * rewards[k]

            # G=np.sum(rewards[t:tmax]) 

            value,grad_value=q_function(params,state, action)
            params+=alpha*(G-value)*grad_value

            returns_per_state[t]=G
            apprx_returns[t]=value

        avg_returns[episode]=np.mean(returns_per_state)
        avg_apprx_returns[episode]=np.mean(apprx_returns)
        loss[episode] = np.mean((avg_returns-avg_apprx_returns)**2)

        if episode%5000==0:
            # print(f"{episode=}")
            # print(f"{params=}")
            avg_win_rate = np.mean(total_return[max(0, episode-1000):episode])
            print(f"Episode {episode}, Avg Win Rate (last 1000): {avg_win_rate:.2f}")

    return params,total_return,epsilon_vec,loss



def test_agent(params, num_games, smart_opponent = False):
    wins = 0
    total_score = 0
    total_opponent_score = 0

    if smart_opponent:
        print("Testing against smart player...")
    else:
        print("Testing against random player...")
    
    for episode in range(num_games):
        deck, discard_pile, revealed, hands, game_over = game_setup()
        current_player = (random.random() < 0.5) + 1
        
        while not game_over:
            if current_player == 1:  
                state = get_state(1, hands, revealed, discard_pile)
                
                # purely greedy policy
                q_vals = np.zeros(9)
                for a in range(9):
                    q_vals[a], _ = q_function(params, state, a)
                action = np.argmax(q_vals)
                
                next_state, game_over = compute_next_state(action, deck, discard_pile, revealed, hands)
            else:
                if smart_opponent:
                    opponent_state, game_over = smart_opponent_turn(deck, discard_pile, revealed, hands)
                else:
                    opponent_state, game_over = opponent_turn(deck, discard_pile, revealed, hands)

            current_player = 3 - current_player
        
        # Calculate final scores
        agent_score = sum(card[0] for card in hands[1])
        opponent_score = sum(card[0] for card in hands[0])
        
        total_score += agent_score
        total_opponent_score += opponent_score
        
        # Check if agent won
        if agent_score < opponent_score:
            wins += 1
    
    print(f"Agent wins: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"Average agent score: {total_score/num_games:.2f}")
    print(f"Average opponent score: {total_opponent_score/num_games:.2f}")
    
    return wins / num_games


params,total_return,epsilon_vec,loss=train_agent(10001)

print(f"{params=}")

# test
test_agent(params, 1000, smart_opponent = False)

plt.subplot(1,3,1)
plt.plot(np.cumsum(total_return))
plt.title("Total return")
plt.xlabel("Episodes")

plt.subplot(1,3,2)
plt.plot(loss)
plt.title("Loss")
plt.xlabel("Episodes")

plt.subplot(1,3,3)
plt.plot(epsilon_vec)
plt.title("Epsilon")
plt.xlabel("Episodes")
plt.show()