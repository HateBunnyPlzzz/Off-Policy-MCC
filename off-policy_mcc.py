import numpy as np
import gym 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    EPS = 0.05
    GAMMA = 1.0

    agentSumSpace = [i for i in range(4, 22)]
    dealerShowCardSpace = [i+1 for i in range(10)]
    agentAceSpace = [False, True]
    actionSpace = [0, 1] # stick or hit
    stateSpace = []

    """ We have two policies in the Off-policy methods.
        One is the agent's estimate of the future rewards which is the usual Q-function, function of states and actions
        Second one is the sum of the relative weights of the particular trajectories occuring under both the target and behavioural policy.
    """
    Q = {}
    C = {}
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                for action in actionSpace:
                    Q[((total, card, ace), action)] = 0
                    C[((total, card, ace), action)] = 0
                stateSpace.append((total, card, ace))

    # This is the policy we'll be using to calculate the optimal behaviour in out environment
    targetPolicy = {}
    for state in stateSpace:
        values = np.array([Q[(state, a)] for a in actionSpace])
        # built-in numpy argMax function
        # takes the first element of a tied list with equal values.
        best = np.random.choice(np.where(values == values.max())[0]) 
        targetPolicy[state] = actionSpace[best] # assigning the best action value to "that state" in the target-policy
    
    numEpisodes = 1000000
    for i in range(numEpisodes):
        memory=[]
        if i % 100000 == 0:
            print('starting episode', i)
        # Epsilon Soft behavioural Policy
        behaviorPolicy = {}
        for state in stateSpace:
            rand = np.random.random()
            if rand < 1 - EPS: # greedy action policy, pull directly from our target policy
                behaviorPolicy[state] = [targetPolicy[state]]
            else:
                behaviorPolicy[state] = actionSpace
        observation = env.reset()
        done = False
        # Play the Game, select the action from Behavioural Policy
        while not done:
            action = np.random.choice(behaviorPolicy[observation]) # random choice from the list.
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], observation[2], action, reward))
            observation = observation_
        # when the episode is over, append the Terminal State, action and reward to the memory
        memory.append((observation[0], observation[1], observation[2], action, reward))

        """ At the end of every episode, we want to iterate backwards over the agent's memory
        and find the returns that followed the time agent encountered the state
        """
        G = 0
        W = 1
        last = True
        for playerSum, dealerCard, usableAce, action, reward in reversed(memory):
            #state-action pair
            sa = ((playerSum, dealerCard, usableAce), action)
            # skip the terminal state
            if last:
                last = False
            else:
                C[sa] += W # update the relative weighting for the particular state-action for the weight.
                # agents estimates of the discounted future rewards get updated according to the following
                # G -> Return (new sample)
                Q[sa] += (W / C[sa])*(G-Q[sa]) 
                values = np.array([Q[(state, a)] for a in actionSpace])
                best = np.random.choice(np.where(values == values.max())[0])
                targetPolicy[state] = actionSpace[best]
                
                # The weakness of the Off-policy methods for Monte-Carlo Control is that 
                # if you take a sub-optimal action, then it goes ahead and breaks out of the learning loop, it only learns from the greedy actions.
                if action != targetPolicy[state]:
                    break
                if len(behaviorPolicy[state]) == 1:
                    prob = 1 -EPS
                else:
                    prob = EPS / len(behaviorPolicy[state])
                W *= 1/prob
            G = GAMMA*G + reward
        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0
    # Testing our learned policy
    numEpisodes = 1000
    rewards = np.zeros(numEpisodes)
    totalReward = 0
    wins = 0
    losses = 0
    draws = 0
    print('getting ready to test target policy')   
    for i in range(numEpisodes):
        observation = env.reset()
        done = False
        while not done:
            action = targetPolicy[observation]
            observation_, reward, done, info = env.step(action)            
            observation = observation_
        totalReward += reward
        rewards[i] = totalReward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1
    
    wins /= numEpisodes
    losses /= numEpisodes
    draws /= numEpisodes
    print('win rate', wins, 'loss rate', losses, 'draw rate', draws)
    plt.plot(rewards)
    plt.show()
