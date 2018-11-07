import gym
from baselines import deepq

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    if is_solved:
        print('reward exceeds 199, problem is solved')
    return is_solved


def register(id):
    gym.envs.registration.register(
        id=id,
        entry_point='gym_gomoku.envs:GomokuEnv',
        kwargs={
            'player_color': 'black',
            'opponent': 'random',  # beginner opponent policy has defend and strike rules
            'board_size': 19,
        },
        nondeterministic=True,
    )


def main():
    game_id = 'Gomoku19x19-v0'
    register(game_id)
    env = gym.make(game_id)  # default 'beginner' level opponent policy
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        learning_starts=0
    )
    print("Saving model to gomoku_model.pkl")
    act.save("gomoku_model.pkl")


if __name__ == '__main__':
    main()
