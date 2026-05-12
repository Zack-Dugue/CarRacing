# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqn_atari_envpoolpy
#
# CarRacing-v3 + EnvPool + PQN-style Q(lambda), adapted from:
#   1. your continuous PPO CarRacing script
#   2. the CleanRL-style PQN Atari EnvPool script
#
# Main changes from your PPO script:
#   - continuous sigmoid-squashed Gaussian actor is replaced by a discrete Q-network
#   - a discrete action wrapper maps 11 discrete actions to CarRacing continuous controls
#   - PPO/GAE losses are replaced by PQN Q(lambda) bootstrapped regression targets
#
# Discrete action map:
#   0: no-op
#   1: left
#   2: right
#   3: gas
#   4: brake
#   5: gas + left
#   6: gas + right
#   7: brake + left
#   8: brake + right
#   9: light gas + left
#   10: light gas + right

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
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic=True"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RacingProject"
    """the wandb project name"""
    wandb_entity: str | None = None
    """the wandb entity/team, or None"""
    capture_video: bool = False
    """kept for CLI compatibility; not used by this EnvPool script"""

    # Optional multi-GPU launcher compatibility, copied from your PQN script.
    device: str | None = None
    """optional device string, e.g. cuda:0; if None, infer cuda/cpu"""

    # Environment / training settings.
    env_id: str = "CarRacing-v3"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiment"""
    learning_rate: float = 2e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per rollout"""
    reset_noop_steps: int = 25
    """number of no-op steps after reset to skip CarRacing zoom-in"""
    road_threshold: float = 150.0
    """grayscale threshold for road-mask channel; pixels below this are road"""
    light_gas: float = 0.5
    """gas value for light-gas turn actions"""
    anneal_lr: bool = True
    """toggle learning rate annealing"""
    gamma: float = 0.99
    """discount factor"""
    num_minibatches: int = 4
    """number of mini-batches"""
    update_epochs: int = 2
    """number of epochs to update the Q-network per rollout"""
    max_grad_norm: float = 0.5
    """maximum gradient norm for clipping"""

    # PQN / Q(lambda) settings.
    start_e: float = 1.0
    """starting epsilon for epsilon-greedy exploration"""
    end_e: float = 0.01
    """final epsilon for epsilon-greedy exploration"""
    exploration_fraction: float = 0.10
    """fraction of total_timesteps used to anneal epsilon"""
    q_lambda: float = 0.65
    """lambda for Q(lambda) targets"""

    # Model / optimizer settings.
    hidden_dim: int = 1024
    """hidden dimension after conv encoder"""
    activation: str = "gelu"
    """activation: gelu, relu, or silu"""
    optimizer: str = "AdamW"
    """optimizer: AdamW, Adam, or SGD"""
    momentum: float = 0.9
    """Adam beta1 or SGD momentum"""
    weight_decay: float = 1e-5
    """AdamW/SGD weight decay"""

    # Checkpointing.
    save_path: str = "checkpoints/carracing_pqn_discrete_seed1.agent.pt"
    """where to save the final checkpoint"""

    # Runtime-filled fields.
    batch_size: int = 0
    """computed at runtime"""
    minibatch_size: int = 0
    """computed at runtime"""
    num_iterations: int = 0
    """computed at runtime"""


class DiscreteCarRacingActionWrapper(gym.Wrapper):
    """
    Wrap continuous CarRacing action space as an 11-action discrete space.

    EnvPool/Gymnasium CarRacing expects actions [steering, gas, brake]:
      steering in [-1, 1]
      gas      in [0, 1]
      brake    in [0, 1]

    The Q-network chooses integer actions. This wrapper maps them to continuous
    controls before passing them into the underlying environment.
    """

    def __init__(self, env, light_gas: float = 0.5):
        super().__init__(env)
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.light_gas = float(light_gas)

        self.action_table = np.array(
            [
                [0.0, 0.0, 0.0],                  # 0: no-op
                [-1.0, 0.0, 0.0],                 # 1: left
                [1.0, 0.0, 0.0],                  # 2: right
                [0.0, 1.0, 0.0],                  # 3: gas
                [0.0, 0.0, 1.0],                  # 4: brake
                [-1.0, 1.0, 0.0],                 # 5: gas + left
                [1.0, 1.0, 0.0],                  # 6: gas + right
                [-1.0, 0.0, 1.0],                 # 7: brake + left
                [1.0, 0.0, 1.0],                  # 8: brake + right
                [-1.0, self.light_gas, 0.0],      # 9: light gas + left
                [1.0, self.light_gas, 0.0],       # 10: light gas + right
            ],
            dtype=np.float32,
        )

        base_obs_space = getattr(env, "single_observation_space", env.observation_space)
        self.single_observation_space = base_obs_space
        self.observation_space = base_obs_space

        self.single_action_space = gym.spaces.Discrete(len(self.action_table))
        self.action_space = self.single_action_space

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        continuous_action = self.action_table[action]
        return self.env.step(continuous_action)


class CarRacingFrameStack(gym.Wrapper):
    """
    Frame-stack + preprocessing wrapper for vectorized EnvPool CarRacing.

    Input observation:
        [num_envs, 96, 96, 3]

    Per-frame preprocessing:
      1. crop off the dashboard: obs[:, 0:84, :, :]
      2. compute grayscale
      3. compute a binary road mask: grayscale < road_threshold
      4. concatenate RGB crop + road mask as 4 channels

    Output observation with stack_size=4:
        [num_envs, 96, 96, 4 * stack_size] = [num_envs, 96, 96, 16]

    The RGB image is NOT cropped. The road-mask channel is full-size too, but
    only the top 84 rows are computed from the dashboard-free crop. The bottom
    dashboard rows are zero-padded in the road-mask channel, exactly as requested.

    The road-mask channel is stored as 0/255 so the model can still simply use
    x.float() / 255.0.
    """

    def __init__(self, env, stack_size=4, road_threshold: float = 150.0, reset_noop_steps: int = 25):
        super().__init__(env)

        self.num_envs = int(getattr(env, "num_envs", 1))
        self.stack_size = int(stack_size)
        self.road_threshold = float(road_threshold)
        self.reset_noop_steps = int(reset_noop_steps)

        base_obs_space = getattr(env, "single_observation_space", env.observation_space)
        base_action_space = getattr(env, "single_action_space", env.action_space)

        if len(base_obs_space.shape) != 3:
            raise ValueError(f"Expected image observation shape [H, W, C], got {base_obs_space.shape}")

        h, w, c = base_obs_space.shape
        if c < 3:
            raise ValueError(f"Expected RGB observation with at least 3 channels, got shape {base_obs_space.shape}")

        self.crop_h = min(84, h)
        self.full_h = h
        self.full_w = w
        per_frame_channels = 4  # full RGB + full road mask with dashboard rows zero-padded

        self.single_observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.full_h, self.full_w, per_frame_channels * self.stack_size),
            dtype=np.uint8,
        )
        self.observation_space = self.single_observation_space

        self.single_action_space = base_action_space
        self.action_space = base_action_space

        self.frames = deque(maxlen=self.stack_size)

    def _preprocess_obs(self, obs):
        """
        Convert raw CarRacing RGB observations to full RGB + full-size road-mask.

        obs: [B, 96, 96, 3] or [96, 96, 3]
        returns: [B, 96, 96, 4] or [96, 96, 4]

        The road-mask is computed only on rows 0:84, then zero-padded on
        dashboard rows 84:96. The RGB channels remain uncropped.
        """
        obs = np.asarray(obs)
        single_obs = obs.ndim == 3
        if single_obs:
            obs = obs[None, ...]

        # Keep the full RGB image. Only the road-mask calculation ignores the dashboard.
        obs_u8 = obs[:, :, :, :3].astype(np.uint8, copy=False)

        # Same grayscale weights as the snippet, but only on the non-dashboard crop.
        crop = obs_u8[:, : self.crop_h, :, :]
        gray = (
            0.2989 * crop[..., 0].astype(np.float32)
            + 0.5870 * crop[..., 1].astype(np.float32)
            + 0.1140 * crop[..., 2].astype(np.float32)
        )

        road_mask_crop = (gray < self.road_threshold).astype(np.uint8) * 255

        # Full-size road-mask with zeros outside the crop, so dashboard rows do not
        # contribute false road/not-road information.
        road_mask = np.zeros((obs_u8.shape[0], obs_u8.shape[1], obs_u8.shape[2], 1), dtype=np.uint8)
        road_mask[:, : self.crop_h, :, 0] = road_mask_crop

        processed = np.concatenate([obs_u8, road_mask], axis=-1)
        if single_obs:
            processed = processed[0]
        return processed

    def _skip_reset_zoom(self, obs):
        """
        After reset, CarRacing visually zooms in for a short period. The user
        requested the same idea as stepping no-op for 25 frames before looking
        at the road. Since the discrete wrapper is inside this wrapper, no-op is
        integer action 0.
        """
        if self.reset_noop_steps <= 0:
            return obs

        noops = np.zeros(self.num_envs, dtype=np.int64)
        latest_obs = obs
        for _ in range(self.reset_noop_steps):
            out = self.env.step(noops)
            if len(out) == 5:
                latest_obs, reward, terminated, truncated, info = out
                done = np.logical_or(terminated, truncated)
            elif len(out) == 4:
                latest_obs, reward, done, info = out
            else:
                raise RuntimeError(f"Expected env.step() to return 4 or 5 values, got {len(out)}")

            # Extremely unlikely during the opening zoom, but if any env ends,
            # reset so frame stacking does not start from a terminal frame.
            if np.asarray(done).any():
                reset_out = self.env.reset()
                latest_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        return latest_obs

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return_info = True
        else:
            obs = out
            info = {}
            return_info = False

        obs = self._skip_reset_zoom(obs)
        obs = self._preprocess_obs(obs)

        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(obs.copy())

        stacked_obs = np.concatenate(list(self.frames), axis=-1)

        if return_info:
            return stacked_obs, info
        return stacked_obs

    def step(self, action):
        out = self.env.step(action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            self.frames.append(self._preprocess_obs(obs).copy())
            stacked_obs = np.concatenate(list(self.frames), axis=-1)
            return stacked_obs, reward, terminated, truncated, info

        if len(out) == 4:
            obs, reward, done, info = out
            self.frames.append(self._preprocess_obs(obs).copy())
            stacked_obs = np.concatenate(list(self.frames), axis=-1)
            return stacked_obs, reward, done, info

        raise RuntimeError(f"Expected env.step() to return 4 or 5 values, got {len(out)}")


class RecordEpisodeStatistics(gym.Wrapper):
    """
    Episode return/length tracker that works with EnvPool-style vector envs
    and Gymnasium-style APIs.
    """

    def __init__(self, env, deque_size=100):
        super().__init__(env)

        self.num_envs = int(getattr(env, "num_envs", 1))
        self.single_observation_space = getattr(env, "single_observation_space", env.observation_space)
        self.single_action_space = getattr(env, "single_action_space", env.action_space)

        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

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

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)
            return_gymnasium_api = True
        elif len(out) == 4:
            obs, reward, done, info = out
            done = np.asarray(done, dtype=bool)
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
            raise RuntimeError(f"Expected env.step(action) to return 4 or 5 values, got {len(out)}")

        reward = np.asarray(reward, dtype=np.float32)

        if info is None:
            info = {}
        elif not isinstance(info, dict):
            info = dict(info)

        self.episode_returns += reward
        self.episode_lengths += 1

        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        info["r"] = self.returned_episode_returns.copy()
        info["l"] = self.returned_episode_lengths.copy()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["done"] = done

        self.episode_returns[done] = 0.0
        self.episode_lengths[done] = 0

        if return_gymnasium_api:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CarRacingPQNQNetwork(nn.Module):
    """
    Your CarRacing conv trunk, converted into a discrete-action Q-network.

    Expected input from wrappers:
      [B, 96, 96, 16] for 4 stacked full RGB+road-mask frames.

    Output:
      [B, 9] Q-values, one per discrete action.
    """

    def __init__(self, envs, hidden_dim=1024, activation="relu"):
        super().__init__()

        obs_shape = envs.single_observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(f"Expected image observation shape [H,W,C], got {obs_shape}")

        h, w, c = obs_shape
        action_dim = int(envs.single_action_space.n)

        if h != 96 or w != 96:
            print(f"[CarRacingPQNQNetwork warning] expected full 96x96 CarRacing obs, got H={h}, W={w}.")
        if c != 16:
            print(f"[CarRacingPQNQNetwork warning] expected 4 stacked RGB+road-mask frames => C=16, got C={c}.")

        self.conv1 = layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4))
        conv1_h = (h - 8) // 4 + 1
        conv1_w = (w - 8) // 4 + 1
        self.ln1 = nn.LayerNorm([32, conv1_h, conv1_w])

        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        conv2_h = (conv1_h - 4) // 2 + 1
        conv2_w = (conv1_w - 4) // 2 + 1
        self.ln2 = nn.LayerNorm([64, conv2_h, conv2_w])

        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        conv3_h = conv2_h - 3 + 1
        conv3_w = conv2_w - 3 + 1
        self.ln3 = nn.LayerNorm([64, conv3_h, conv3_w])

        self.flatten = nn.Flatten()
        conv_flatten_dim = 64 * conv3_h * conv3_w
        self.trunk_fc = layer_init(nn.Linear(conv_flatten_dim, hidden_dim))
        self.trunk_ln = nn.LayerNorm(hidden_dim)

        if activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == "silu":
            self.act = nn.SiLU()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.q_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"[CarRacingPQNQNetwork] obs_shape={obs_shape}, hidden_dim={hidden_dim}, "
            f"action_dim={action_dim}, activation={activation}, total_params={total_params:,}"
        )

    def _normalize_input(self, x):
        x = x.float() / 255.0
        x = torch.permute(x, (0, 3, 1, 2))
        return x

    def forward(self, x):
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

        return self.q_head(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    duration = max(int(duration), 1)
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def make_optimizer(args: Args, q_network: nn.Module):
    if args.optimizer == "AdamW":
        return optim.AdamW(
            q_network.parameters(),
            lr=args.learning_rate,
            betas=(args.momentum, 0.999),
            eps=1e-5,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "Adam":
        return optim.Adam(
            q_network.parameters(),
            lr=args.learning_rate,
            betas=(args.momentum, 0.999),
            eps=1e-5,
        )
    if args.optimizer == "SGD":
        return optim.SGD(
            q_network.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}. Supported: AdamW, Adam, SGD")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.num_iterations <= 0:
        raise ValueError(
            f"total_timesteps={args.total_timesteps} is smaller than one batch "
            f"num_envs*num_steps={args.batch_size}. Increase total_timesteps."
        )

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

    # Seeding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"the device we're using is: {device}")

    # Env setup. Keep your CarRacing EnvPool setup, but map discrete action IDs
    # to continuous controls before stepping the actual environment.
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        seed=args.seed,
    )
    envs = DiscreteCarRacingActionWrapper(envs, light_gas=args.light_gas)
    envs = CarRacingFrameStack(
        envs,
        stack_size=4,
        road_threshold=args.road_threshold,
        reset_noop_steps=args.reset_noop_steps,
    )
    envs = RecordEpisodeStatistics(envs)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "PQN requires a discrete action space"
    print(
        "Discrete action table: "
        "0 noop, 1 left, 2 right, 3 gas, 4 brake, "
        "5 gas+left, 6 gas+right, 7 brake+left, 8 brake+right, "
        "9 light-gas+left, 10 light-gas+right"
    )

    q_network = CarRacingPQNQNetwork(
        envs,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
    ).to(device)

    optimizer = make_optimizer(args, q_network)

    # Preserve per-parameter-group LR ratios during annealing.
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    # Storage.
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    avg_returns = deque(maxlen=20)

    global_step = 0
    start_time = time.time()

    reset_out = envs.reset()
    if isinstance(reset_out, tuple):
        next_obs, reset_info = reset_out
    else:
        next_obs = reset_out
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for group in optimizer.param_groups:
                group["lr"] = frac * group["initial_lr"]

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                global_step,
            )

            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs, device=device), max_actions]

            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,), device=device)
            explore = torch.rand((args.num_envs,), device=device) < epsilon
            action = torch.where(explore, random_actions, max_actions)

            actions[step] = action

            step_out = envs.step(action.cpu().numpy())

            if len(step_out) == 5:
                next_obs_np, reward, next_terminated, next_truncated, info = step_out
                next_done_np = np.logical_or(next_terminated, next_truncated)
            elif len(step_out) == 4:
                next_obs_np, reward, next_done_np, info = step_out
            else:
                raise RuntimeError(f"Expected env.step() to return 4 or 5 values, got {len(step_out)}")

            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            for idx, done_flag in enumerate(next_done_np):
                if done_flag:
                    episodic_return = float(info["r"][idx])
                    episodic_length = int(info["l"][idx])
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    avg_returns.append(episodic_return)
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/avg_episodic_return", float(np.average(avg_returns)), global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        # Compute Q(lambda) targets.
        with torch.no_grad():
            returns = torch.zeros_like(rewards, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_q_values = q_network(next_obs)
                    next_value = torch.max(next_q_values, dim=-1).values
                    nextnonterminal = 1.0 - next_done.float()
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    next_value = values[t + 1]
                    returns[t] = rewards[t] + args.gamma * (
                        args.q_lambda * returns[t + 1]
                        + (1.0 - args.q_lambda) * next_value
                    ) * nextnonterminal

        # Flatten batch.
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimize Q-network.
        b_inds = np.arange(args.batch_size)
        losses = []
        q_means = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                chosen_q = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1)).squeeze(-1)
                loss = F.mse_loss(chosen_q, b_returns[mb_inds])

                optimizer.zero_grad()
                loss.backward()

                if not torch.isfinite(loss):
                    print("Non-finite TD loss detected:", loss.item())
                    raise RuntimeError("Stopping because TD loss became non-finite.")

                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)

                for name, param in q_network.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise RuntimeError(f"Non-finite gradient in {name}")

                optimizer.step()

                losses.append(loss.item())
                q_means.append(chosen_q.detach().mean().item())

        sps = int(global_step / (time.time() - start_time))
        current_lr = optimizer.param_groups[0]["lr"]
        epsilon_now = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        writer.add_scalar("charts/learning_rate", current_lr, global_step)
        writer.add_scalar("charts/epsilon", epsilon_now, global_step)
        writer.add_scalar("losses/td_loss", float(np.mean(losses)), global_step)
        writer.add_scalar("losses/q_values", float(np.mean(q_means)), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"SPS: {sps}, epsilon={epsilon_now:.4f}, td_loss={float(np.mean(losses)):.6f}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "q_network_state_dict": q_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "global_step": global_step,
            "run_name": run_name,
            "action_table": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [-1.0, args.light_gas, 0.0],
                    [1.0, args.light_gas, 0.0],
                ],
                dtype=np.float32,
            ),
            "observation_preprocessing": {
                "rgb": "full 96x96 RGB, uncropped",
                "road_mask_crop": "computed on obs[:, 0:84, :, :]",
                "road_mask_padding": "rows outside crop are zero-padded",
                "channels_per_frame": "full RGB + full-size road_mask",
                "road_mask": f"gray < {args.road_threshold}",
                "reset_noop_steps": args.reset_noop_steps,
            },
        },
        args.save_path,
    )
    abs_save_path = os.path.abspath(args.save_path)
    print(f"Saved final checkpoint to: {abs_save_path}")
    writer.add_text("checkpoint/final_agent_path", abs_save_path, global_step)

    envs.close()
    writer.close()

    if args.track:
        import wandb

        wandb.finish()
