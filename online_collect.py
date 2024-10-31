import numpy as np
def online_collect(agent,env,fixed_epi,return_list):
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    total_epi=0
    episode_return = 0
    state = env.reset()
    done = False
    while total_epi<fixed_epi:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        state = next_state
        episode_return += reward
        if done:
            total_epi+=1
            return_list.append(episode_return)
            episode_return = 0
            state = env.reset()
            done = False
    return transition_dict
    