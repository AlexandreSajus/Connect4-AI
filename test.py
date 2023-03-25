import torch
import gym

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy, DQNPolicy
from tianshou.utils.net.common import Net

from pettingzoo.classic import connect_four_v3

POLICY_PATH = "log/rps/dqn/best_policy.pth"

if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = connect_four_v3.env(render_mode="human")

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    # model
    net = Net(
        state_shape=observation_space["observation"].shape
        or observation_space["observation"].n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[256, 256, 256, 128],
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    agent_learn = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=320,
    )

    # Load policy
    agent_learn.load_state_dict(torch.load(POLICY_PATH))

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy(), agent_learn], env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=4, render=0.5)
