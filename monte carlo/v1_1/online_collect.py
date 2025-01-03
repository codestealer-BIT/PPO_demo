import numpy as np
from sklearn.preprocessing import MinMaxScaler
# 分桶函数
def bucketize(value, min_val, max_val, num_buckets):
    # 限制输入值在指定的范围内
    value = np.clip(value, min_val, max_val)
    
    # 计算桶宽度
    bucket_width = (max_val - min_val) / num_buckets
    bucket_index = int((value - min_val) // bucket_width)
    
    # 确保桶索引在合理范围内
    return max(0, min(bucket_index, num_buckets - 1))

# 定义每个特征的最小值、最大值和分桶数
buckets = {
    'x': (-1, 1, 10),
    'y': (-1, 1, 10),
    'x_velocity': (-2, 2, 10),
    'y_velocity': (-2, 2, 10),
    'angle': (-np.pi, np.pi, 10),
    'angular_velocity': (-3, 3, 10),
    'left_leg_contact': (0, 1, 2),  # 二元特征，0 或 1
    'right_leg_contact': (0, 1, 2)  # 二元特征，0 或 1
}

# 分桶函数：f(state) 返回分桶后的状态
def bucket(state):
    binned_state = []
    for i, value in enumerate(state):
        if i<len(buckets):
            # 获取每个特征对应的分桶参数
            feature_name = list(buckets.keys())[i]
            min_val, max_val, num_buckets = buckets[feature_name]
            
            # 根据分桶规则计算该特征所属的桶
            binned_state.append(bucketize(value, min_val, max_val, num_buckets))
        else:
            binned_state.append(state[i])
    return binned_state


def compute_returns(rewards,gamma):
    returns=[]
    G=0
    for reward in reversed(rewards):
        G=reward+gamma*G
        returns.append(G)
    returns.reverse()
    return returns


def online_collect(agent,env,fixed_epi,return_list,gamma):
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],'returns':[],'probs':[]}
    total_epi=0
    episode_return = []
    state = env.reset()
    done = False
    while total_epi<fixed_epi:
        # print(state,end=' ')
        # state = bucket(state)
        state=state.astype(np.float32)
        state/=255
        action,prob = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        # next_state = scaler.fit_transform(next_state.reshape(-1,1)).reshape(-1)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        transition_dict['probs'].append(prob)
        state = next_state
        episode_return.append(reward)
        if done:
            total_epi+=1
            return_list.append(sum(episode_return))
            returns=compute_returns(episode_return,gamma)
            transition_dict['returns'].extend(returns)
            episode_return = []
            state = env.reset()
            done = False
    return transition_dict['states'],transition_dict['actions'],transition_dict['probs'],transition_dict['returns']
    