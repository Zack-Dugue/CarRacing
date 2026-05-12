# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CarRacing-v3"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    save_path: str = "agent_model.pth"

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""



class RecordEpisodeStatistics(gym.Wrapper):
    """
    Episode return/length tracker that works with EnvPool-style vector envs
    and Gymnasium-style APIs.

    It supports either:

        reset() -> obs
        step()  -> obs, reward, done, info

    or:

        reset() -> obs, info
        step()  -> obs, reward, terminated, truncated, info

    It returns the same API style it receives from step():
        - If env.step gives 4 values, this wrapper returns 4 values.
        - If env.step gives 5 values, this wrapper returns 5 values.

    It injects:
        info["r"] = episode returns before reset masking
        info["l"] = episode lengths before reset masking
        info["terminated"] = done/terminated array
        info["truncated"] = truncated array if available, else zeros
        info["done"] = final done array
    """

    def __init__(self, env, deque_size=100):
        super().__init__(env)

        self.num_envs = int(getattr(env, "num_envs", 1))

        # CleanRL expects these names. EnvPool may expose read-only properties,
        # so we define them on the wrapper instead of mutating the raw env.
        self.single_observation_space = getattr(
            env,
            "single_observation_space",
            env.observation_space,
        )
        self.single_action_space = getattr(
            env,
            "single_action_space",
            env.action_space,
        )

        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        # Gymnasium reset returns (obs, info); old Gym/EnvPool-gym often returns obs.
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return_info = True
        else:
            obs = out
            info = {}
            return_info = False

        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        if return_info:
            return obs, info
        return obs

    def step(self, action):
        out = self.env.step(action)

        if not isinstance(out, tuple):
            raise RuntimeError(f"Expected env.step(action) to return a tuple, got {type(out)}")

        # Gymnasium API: obs, reward, terminated, truncated, info
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)
            return_gymnasium_api = True

        # Old Gym / EnvPool gym API: obs, reward, done, info
        elif len(out) == 4:
            obs, reward, done, info = out
            done = np.asarray(done, dtype=bool)

            # EnvPool sometimes includes these in info; otherwise infer them.
            if isinstance(info, dict) and "terminated" in info:
                terminated = np.asarray(info["terminated"], dtype=bool)
            else:
                terminated = done

            if isinstance(info, dict) and "truncated" in info:
                truncated = np.asarray(info["truncated"], dtype=bool)
            else:
                truncated = np.zeros_like(done, dtype=bool)

            return_gymnasium_api = False

        else:
            raise RuntimeError(
                f"Expected env.step(action) to return 4 or 5 values, got {len(out)}"
            )

        reward = np.asarray(reward, dtype=np.float32)

        # EnvPool info should be a dict of arrays. If not, make a mutable dict.
        if info is None:
            info = {}
        elif not isinstance(info, dict):
            info = dict(info)

        self.episode_returns += reward
        self.episode_lengths += 1

        # Store the just-finished episode stats before zeroing done envs.
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        info["r"] = self.returned_episode_returns.copy()
        info["l"] = self.returned_episode_lengths.copy()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["done"] = done

        # Reset counters for envs whose episode ended.
        self.episode_returns[done] = 0.0
        self.episode_lengths[done] = 0

        if return_gymnasium_api:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ConvSimpleAgent(nn.Module):
    """
    PQN-style ConvNet actor-critic for Atari EnvPool observations,
    with input BatchRenorm2d.

    Expected input:
      x shape [batch, 4, 84, 84], uint8-ish pixels in [0, 255]

    Input normalization:
      x.float() / 255.0
      BatchRenorm2d(C)

    For Atari frame stacks:
      C = 4

    So BatchRenorm2d tracks one running distribution per stacked-frame channel,
    across batch and spatial dimensions. This is much more natural for convnets
    than flattening the whole image and using BatchRenorm1d(obs_dim).

    PQN-style encoder:
      Conv2d -> LayerNorm -> ReLU
      Conv2d -> LayerNorm -> ReLU
      Conv2d -> LayerNorm -> ReLU
      Flatten
      Linear -> LayerNorm -> ReLU

    PPO-style separate heads:
      actor_out:  Linear(hidden_dim, action_dim), std=0.01
      critic_out: Linear(hidden_dim, 1), std=1.0

    Muon routing:
      default:
        Muon gets trunk_fc.weight

      use_muon_input=True:
        also sends conv1.weight, conv2.weight, conv3.weight to Muon

      use_muon_output=True:
        also sends actor_out.weight and critic_out.weight to Muon

      Adam gets:
        BatchRenorm params,
        LayerNorm params,
        all biases,
        and anything not explicitly routed to Muon.
    """

    def __init__(
        self,
        envs,
        hidden_dim=512,
        *,
        use_muon_input=False,
        use_muon_output=False,
        continuous_eps = .00001
    ):
        super().__init__()

        self.use_muon_input = use_muon_input
        self.use_muon_output = use_muon_output

        obs_shape = envs.single_observation_space.shape
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.register_buffer( "action_low",torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer( "action_high",torch.tensor(envs.single_action_space.high, dtype=torch.float32))
        if len(obs_shape) != 3:
            raise ValueError(
                f"Expected Atari image observation shape [C,H,W], got {obs_shape}"
            )

        h, w, c = obs_shape

        if c != 4:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] expected 4 stacked frames, got C={c}. "
                "Continuing anyway."
            )

        if h != 84 or w != 84:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] expected 84x84 Atari obs, got H={h}, W={w}. "
                "LayerNorm shapes assume standard Atari preprocessing."
            )

        if hidden_dim != 512:
            print(
                f"[BetterSimpleAgent/PQN-BRN2d warning] PQN usually uses hidden_dim=512. "
                f"You passed hidden_dim={hidden_dim}."
            )

        # Input BatchRenorm over image channels.
        #
        # For x shape [B, C, H, W], BatchRenorm2d(C) normalizes each channel
        # using statistics computed over B,H,W.
        #
        # For Atari stacked grayscale frames, C=4, so each frame index gets its
        # own running statistics.
        # self.input_brn = BatchRenorm2d(
        #     c,
        #     eps=brn_eps,
        #     momentum=brn_momentum,
        #     max_r=brn_max_r,
        #     max_d=brn_max_d,
        #     warmup_steps=brn_warmup_steps,
        #     smooth=brn_smooth,
        # )

        # ----- PQN-style conv encoder -----
        self.conv1 = layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4))
        self.ln1 = nn.LayerNorm([32, 23, 23])

        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.ln2 = nn.LayerNorm([64, 10, 10])

        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.ln3 = nn.LayerNorm([64, 8, 8])

        self.flatten = nn.Flatten()

        # For 84x84 Atari:
        # 84 -> conv1 -> 20
        # 20 -> conv2 -> 9
        # 9  -> conv3 -> 7
        # so final conv output is 64 * 7 * 7 = 3136.
        self.trunk_fc = layer_init(nn.Linear(4096, hidden_dim))
        self.trunk_ln = nn.LayerNorm(hidden_dim)

        self.act = nn.GELU()

        # ----- Separate PPO actor/value heads -----
        self.actor_fc = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.critic_fc = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.actor_ln = nn.LayerNorm(hidden_dim)
        self.critic_ln = nn.LayerNorm(hidden_dim)

        self.continuous_eps = continuous_eps

        self.actor_mean = layer_init(
            nn.Linear(hidden_dim, action_dim),
            std=0.01,
        )

        self.actor_log_std =  layer_init(
            nn.Linear(hidden_dim, action_dim),
            std=0.01,
        )

        self.critic_out = layer_init(
            nn.Linear(hidden_dim, 1),
            std=1.0,
        )

        total_params = sum(p.numel() for p in self.parameters())
        muon_params, adam_params = self.get_split_params()

        print(
            f"[BetterSimpleAgent/PQN-BRN2d] obs_shape={obs_shape}, "
            f"hidden_dim={hidden_dim}, action_dim={action_dim}"
        )
        print("[BetterSimpleAgent/PQN-BRN2d] input normalization: BatchRenorm2d(C)")
        print(f"[BetterSimpleAgent/PQN-BRN2d] total parameters: {total_params:,}")
        print(f"[BetterSimpleAgent/PQN-BRN2d] Muon parameters: {sum(p.numel() for p in muon_params):,}")
        print(f"[BetterSimpleAgent/PQN-BRN2d] Adam parameters: {sum(p.numel() for p in adam_params):,}")
        print(
            f"[BetterSimpleAgent/PQN-BRN2d] "
            f"use_muon_input={use_muon_input}, use_muon_output={use_muon_output}"
        )

    def _normalize_input(self, x):
        """
        Normalize raw Atari image input.

        Input:
          x: [B, C, H, W], usually uint8 in [0, 255]

        Output:
          x: [B, C, H, W], float normalized by /255 and BatchRenorm2d.
        """
        x = x.float() / 255.0
        x = torch.permute(x, (0, 3, 1, 2))
        # x = self.input_brn(x)
        return x

    def _features(self, x):
        x = self._normalize_input(x)

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.ln3(x)
        x = self.act(x)

        x = self.flatten(x)

        x = self.trunk_fc(x)
        x = self.trunk_ln(x)
        x = self.act(x)

        return x

    def get_split_params(self):
        """
        Returns:
            muon_params, adam_params

        Default:
          Muon:
            trunk_fc.weight

          Adam:
            input_brn params
            conv weights unless use_muon_input=True
            output heads unless use_muon_output=True
            all biases
            all LayerNorm params

        Optional:
          use_muon_input=True:
            conv1.weight
            conv2.weight
            conv3.weight

          use_muon_output=True:
            actor_out.weight
            critic_out.weight
        """

        muon_params = [
            self.actor_fc.weight,
            self.critic_fc.weight,
            self.trunk_fc.weight,
        ]

        if self.use_muon_input:
            muon_params.extend([
                # self.conv1.weight,
                self.conv2.weight,
                self.conv3.weight,
            ])

        if self.use_muon_output:
            muon_params.extend([
                self.actor_mean.weight,
                self.actor_log_std.weight,
                self.critic_out.weight,
            ])

        muon_ids = {id(p) for p in muon_params}

        adam_params = [
            p for p in self.parameters()
            if id(p) not in muon_ids
        ]

        return muon_params, adam_params

    def get_value(self, x):
        features = self._features(x)
        return self.critic_out(self.act(self.critic_ln(self.critic_fc(features))))

    def get_action_and_value(self, x, action=None):
        features = self._features(x)

        actor_feactures = self.act(self.actor_ln(self.actor_fc(features)))
        actor_log_std = self.actor_log_std(actor_feactures)
        actor_mean = self.actor_mean(actor_feactures)
        normal = Normal(actor_mean, torch.exp(actor_log_std))

        #The action is assumed to be the environment action so

        if action is None:
            squashed_action = normal.sample()
            squashed_action = torch.sigmoid(squashed_action)
            squashed_action = squashed_action.clamp(self.continuous_eps, 1 - self.continuous_eps)
        else:
            # we need to undo the scaling
            squashed_action = (action - self.action_low)/(self.action_high - self.action_low)
            squashed_action = squashed_action.clamp(self.continuous_eps, 1 - self.continuous_eps)

        raw_action = torch.logit(squashed_action, eps=self.continuous_eps)
        log_prob = normal.log_prob(raw_action) - torch.log(squashed_action * ( 1-squashed_action) + self.continuous_eps)
        log_prob = log_prob.sum(dim=-1)
        action = squashed_action * (self.action_high - self.action_low) + self.action_low

        value = self.critic_out(self.act(self.critic_ln(self.critic_fc(features))))

        return action,log_prob, normal.entropy().sum(dim=-1), value

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        # continuous=False,
        # episodic_life=True,
        # reward_clip=True,
        seed=args.seed,
    )
    # envs.num_envs = args.num_envs
    # envs.single_action_space = envs.action_space
    # envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    # assert isinstance(envs.action_space, gym.spaces.Continuous), "only continuous action space is supported"

    agent = ConvSimpleAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    torch.save(agent.state_dict(), args.save_path)
    envs.close()
    writer.close()