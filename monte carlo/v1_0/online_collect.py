def compute_returns(rewards,gamma):
    returns=[]
    G=0
    for reward in reversed(rewards):
        G=reward+gamma*G
        returns.append(G)
    returns.reverse()
    return returns
import numpy as np

def online_collect(agent,env,fixed_epi,return_list,gamma):
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],'returns':[]}
    total_epi=0
    episode_return = []
    state = env.reset()
    done = False
    while total_epi<fixed_epi:
        state=state.astype(np.float32)
        state/=255
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        state = next_state
        episode_return.append(reward)
        if reward!=0:
            total_epi+=1
            return_list.append(sum(episode_return))
            returns=compute_returns(episode_return,gamma)
            transition_dict['returns'].extend(returns)
            episode_return = []
        if done:
            state = env.reset()
            done = False
    return transition_dict
    