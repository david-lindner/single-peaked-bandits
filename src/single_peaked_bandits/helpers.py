cumulative_reward_cache = {}

def cumulative_reward(reward_function, t, cache=True):
    if cache and (reward_function, t) in cumulative_reward_cache:
        return cumulative_reward_cache[(reward_function, t)]
    else:
        res = 0
        for i in range(1, t + 1):
            res += reward_function(i)
        if cache:
            cumulative_reward_cache[(reward_function, t)] = res
        return res

def get_reward_for_policy(arms, T, policy, cache=True):
    reward = 0
    for f, n in zip(arms, policy):
        reward += cumulative_reward(f, n, cache=cache)
    return reward
