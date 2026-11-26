import time
import math
import json
import zipfile
import tempfile
import shutil
from io import BytesIO
from typing import Any

import cloudpickle
import gymnasium as gym
import numpy as np
import torch
import random
from tqdm import tqdm

from .agents import build_agent
from .envs import make_vec_env
from simplyqrl.monitoring import (
    CoreTrainingMonitor,
    EvaluationMonitor,
    MonitorManager,
    OptimizationDiagnosticsMonitor,
    QuantumMonitor,
    ResourceMonitor,
)
from simplyqrl.reporting import ArtifactLogger, build_run_context, ensure_hardware_profile
from simplyqrl.runtime import resolve_device_policy
from simplyqrl.utils import count_trainable_params
from simplyqrl.optimization_diagnostics import OptimizationDiagnosticsProbe
from simplyqrl.optimization import (
    SUPPORTED_GRADIENT_CLIPPING_MODES,
    SUPPORTED_LEARNING_RATE_MODES,
    apply_gradient_clipping,
    build_adam_optimizer,
    current_branch_learning_rates,
    resolve_optimization_parameter_groups,
    set_annealed_learning_rates,
)



class PPO:
    """SimplyQRL PPO for discrete action spaces.

    SimplyQRL 1.5 keeps PPO as the common training loop for classical and hybrid
    actor-critic agents, while reporting is handled through monitors and artifact
    loggers instead of hardcoded TensorBoard/CSV calls.
    """

    def __init__(
        self,
        env_id="CartPole-v1",
        env=None,
        seed=1,
        agent_type="mlp",
        agent_config=None,
        learning_rate=2.5e-4,
        num_envs=1,
        num_steps=512,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=4,
        norm_adv=True,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gradient_clipping_mode="global",
        actor_max_grad_norm=None,
        critic_max_grad_norm=None,
        shared_max_grad_norm=None,
        learning_rate_mode="global",
        actor_learning_rate=None,
        critic_learning_rate=None,
        shared_learning_rate=None,
        target_kl=None,
        run_name="PPO",
        reporting_config=None,
        monitors=None,
        device_policy=None,
    ):
        self.env_id = env_id
        self.env = env
        self.seed = seed
        self.learning_rate = float(learning_rate)
        self.learning_rate_mode = str(learning_rate_mode).strip().lower()
        supported_learning_rate_modes = SUPPORTED_LEARNING_RATE_MODES
        if self.learning_rate_mode not in supported_learning_rate_modes:
            supported = ", ".join(sorted(supported_learning_rate_modes))
            raise ValueError(
                f"Unsupported learning_rate_mode={learning_rate_mode!r}. "
                f"Supported: {supported}."
            )
        self.actor_learning_rate = float(
            self.learning_rate if actor_learning_rate is None else actor_learning_rate
        )
        self.critic_learning_rate = float(
            self.learning_rate if critic_learning_rate is None else critic_learning_rate
        )
        self.shared_learning_rate = float(
            self.learning_rate if shared_learning_rate is None else shared_learning_rate
        )
        for name, value in {
            "learning_rate": self.learning_rate,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "shared_learning_rate": self.shared_learning_rate,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}.")
        if self.learning_rate_mode == "global":
            explicit_branch_rates = {
                "actor_learning_rate": actor_learning_rate,
                "critic_learning_rate": critic_learning_rate,
                "shared_learning_rate": shared_learning_rate,
            }
            mismatched = [
                name
                for name, raw in explicit_branch_rates.items()
                if raw is not None and not math.isclose(
                    float(raw), self.learning_rate, rel_tol=1e-12, abs_tol=0.0
                )
            ]
            if mismatched:
                raise ValueError(
                    "Branch-specific learning rates require "
                    "learning_rate_mode='separate_actor_critic': "
                    + ", ".join(mismatched)
                )
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = float(max_grad_norm)
        self.gradient_clipping_mode = str(gradient_clipping_mode).strip().lower()
        supported_clipping_modes = SUPPORTED_GRADIENT_CLIPPING_MODES
        if self.gradient_clipping_mode not in supported_clipping_modes:
            supported = ", ".join(sorted(supported_clipping_modes))
            raise ValueError(
                f"Unsupported gradient_clipping_mode={gradient_clipping_mode!r}. "
                f"Supported: {supported}."
            )
        self.actor_max_grad_norm = float(
            self.max_grad_norm if actor_max_grad_norm is None else actor_max_grad_norm
        )
        self.critic_max_grad_norm = float(
            self.max_grad_norm if critic_max_grad_norm is None else critic_max_grad_norm
        )
        self.shared_max_grad_norm = float(
            self.max_grad_norm if shared_max_grad_norm is None else shared_max_grad_norm
        )
        for name, value in {
            "max_grad_norm": self.max_grad_norm,
            "actor_max_grad_norm": self.actor_max_grad_norm,
            "critic_max_grad_norm": self.critic_max_grad_norm,
            "shared_max_grad_norm": self.shared_max_grad_norm,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}.")
        self.target_kl = target_kl
        self.agent_type = agent_type
        self.agent_config = agent_config
        self.run_name = run_name

        self.reporting_config = reporting_config or {}
        self.external_monitors = monitors or []
        self.device_policy = resolve_device_policy(device_policy)
        self._device = self.device_policy.torch_device

        if self._device.type == "cuda" and self._device.index is not None:
            torch.cuda.set_device(self._device.index)
        self.artifact_logger = None
        self.monitor_manager = MonitorManager([])
        self.evaluation_monitors = []
        self._reporting_closed = False
        self._envs_closed = False
        self.optimization_diagnostics_probe = None

        # Derived values
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches

        if self.minibatch_size <= 0:
            raise ValueError(
                "Invalid PPO minibatch configuration: "
                f"batch_size={self.batch_size}, num_minibatches={self.num_minibatches}."
            )

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.envs = make_vec_env(
            base=env if env is not None else env_id,
            num_envs=num_envs,
            seed=seed,
        )

        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action spaces supported"

        # Create actor-critic
        obs_shape = self.envs.single_observation_space.shape
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        n_actions = self.envs.single_action_space.n

        self.agent = build_agent(
            agent_type,
            self.obs_shape,
            n_actions,
            is_qnet=False,
            config=agent_config,
        )
        self.agent.to(self.device)
        self._apply_qnode_wrapper_policy()

        # Create the optimizer after final placement. Hybrid runs may deliberately
        # contain mixed CPU/CUDA parameters (CUDA CNN/MLP + CPU QNode boundary).
        # The grouped-LR path still uses one Adam and one joint PPO loss.
        self.optimization_parameter_groups = {}
        self.optimizer = None
        self.rebuild_optimizer()

        # Modular SimplyQRL agents preserve the original observation shape so
        # feature extractors such as CNNExtractor can consume visual inputs.
        self.obs = torch.zeros(
            (self.num_steps, self.num_envs) + self.obs_shape,
            dtype=torch.float32,
            device=self.device,
        )

        self.actions = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), device=self.device)

        self.global_step = 0
        self.episode_counter = 0

        self._setup_reporting()

    def _branch_learning_rates(self) -> dict[str, float]:
        return {
            "shared": self.shared_learning_rate,
            "actor": self.actor_learning_rate,
            "critic": self.critic_learning_rate,
        }

    def _branch_max_grad_norms(self) -> dict[str, float]:
        return {
            "shared": self.shared_max_grad_norm,
            "actor": self.actor_max_grad_norm,
            "critic": self.critic_max_grad_norm,
        }

    def _resolve_optimization_parameter_groups(self):
        return resolve_optimization_parameter_groups(
            self.agent,
            require_branch_groups=(
                self.gradient_clipping_mode != "global"
                or self.learning_rate_mode != "global"
            ),
        )

    def _build_optimizer(self):
        return build_adam_optimizer(
            self.agent,
            self.optimization_parameter_groups,
            learning_rate_mode=self.learning_rate_mode,
            learning_rate=self.learning_rate,
            branch_learning_rates=self._branch_learning_rates(),
        )

    def rebuild_optimizer(self):
        """Rebuild Adam from the agent's current trainable modules.

        Staged composition replaces actor/critic module objects. Rebuilding must
        therefore re-resolve parameter ownership and preserve the configured
        global or branch-specific learning-rate policy. Optimizer state is reset
        intentionally because the parameter objects changed.
        """

        self.optimization_parameter_groups = self._resolve_optimization_parameter_groups()
        self.optimizer = self._build_optimizer()

        if self.optimization_diagnostics_probe is not None:
            self.optimization_diagnostics_probe = self._create_optimization_diagnostics_probe()
        return self.optimizer

    def _set_annealed_learning_rates(self, fraction: float) -> None:
        set_annealed_learning_rates(
            self.optimizer,
            fraction,
            learning_rate_mode=self.learning_rate_mode,
            learning_rate=self.learning_rate,
            branch_learning_rates=self._branch_learning_rates(),
        )

    def _current_branch_learning_rates(self) -> dict[str, float | None]:
        return current_branch_learning_rates(
            self.optimizer,
            learning_rate_mode=self.learning_rate_mode,
        )

    def _apply_gradient_clipping(
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        return apply_gradient_clipping(
            self.agent,
            self.optimization_parameter_groups,
            gradient_clipping_mode=self.gradient_clipping_mode,
            max_grad_norm=self.max_grad_norm,
            branch_max_grad_norms=self._branch_max_grad_norms(),
        )

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------
    @property
    def device(self):
        return self._device

    def _apply_qnode_wrapper_policy(self) -> None:
        """Apply the internal QNode-wrapper placement policy for hybrid agents."""

        configurator = getattr(self.agent, "configure_qnode_wrappers", None)
        if callable(configurator):
            configurator(self.device)

    def _device_metadata(self) -> dict[str, Any]:
        metadata = {
            **self.device_policy.to_dict(),
            "ppo_device": str(self.device),
        }

        if hasattr(self.agent, "get_device_metadata"):
            metadata.update(self.agent.get_device_metadata())

        return metadata

    def _prepare_obs(self, obs_np):
        """Convert vector-env observations into an agent-ready tensor."""
        return torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

    def _prepare_inference_obs(self, obs_np):
        """Convert a single observation or batch into agent-ready tensor form."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if tuple(obs.shape) == self.obs_shape:
            obs = obs.unsqueeze(0)
        return obs

    @torch.no_grad()
    def predict(self, obs_np, *, deterministic: bool = True):
        """Predict actions for a single observation or batch of observations."""
        obs = self._prepare_inference_obs(obs_np)
        outputs = self.agent.get_teacher_outputs(obs, deterministic=deterministic)
        return outputs["actions"].detach().cpu().numpy()

    @torch.no_grad()
    def get_teacher_outputs(self, obs_np, *, deterministic: bool = True) -> dict[str, np.ndarray]:
        """Return features/logits/values/actions for KD dataset export."""
        obs = self._prepare_inference_obs(obs_np)
        outputs = self.agent.get_teacher_outputs(obs, deterministic=deterministic)
        return {
            key: value.detach().cpu().numpy()
            for key, value in outputs.items()
        }

    def _flatten_rollout_obs(self):
        """Flatten rollout time/env axes while preserving observation shape."""
        return self.obs.reshape((self.batch_size,) + self.obs_shape)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    @property
    def reporting_enabled(self):
        return bool(self.reporting_config.get("enabled", True))

    def _env_label(self) -> str:
        if self.env is None:
            return str(self.env_id)

        if isinstance(self.env, type):
            return self.env.__name__

        if callable(self.env):
            return getattr(self.env, "__name__", repr(self.env))

        return repr(self.env)

    def _effective_config(self) -> dict[str, Any]:
        """Return the effective run configuration saved by the artifact logger."""
        return {
            "run_name": self.run_name,
            "env_id": self.env_id if self.env is None else None,
            "env": self._env_label() if self.env is not None else None,
            "seed": self.seed,
            "agent_type": self.agent_type,
            "agent_config": self.agent_config,
            "training_config": {
                "learning_rate": self.learning_rate,
                "learning_rate_mode": self.learning_rate_mode,
                "actor_learning_rate": self.actor_learning_rate,
                "critic_learning_rate": self.critic_learning_rate,
                "shared_learning_rate": self.shared_learning_rate,
                "num_envs": self.num_envs,
                "num_steps": self.num_steps,
                "anneal_lr": self.anneal_lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "num_minibatches": self.num_minibatches,
                "update_epochs": self.update_epochs,
                "norm_adv": self.norm_adv,
                "clip_coef": self.clip_coef,
                "clip_vloss": self.clip_vloss,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
                "gradient_clipping_mode": self.gradient_clipping_mode,
                "actor_max_grad_norm": self.actor_max_grad_norm,
                "critic_max_grad_norm": self.critic_max_grad_norm,
                "shared_max_grad_norm": self.shared_max_grad_norm,
                "target_kl": self.target_kl,
                "batch_size": self.batch_size,
                "minibatch_size": self.minibatch_size,
            },
            "device_policy": self.device_policy.to_dict(),
            "reporting_config": self.reporting_config,
        }

    def _get_trainable_parameter_report(self) -> dict[str, Any]:
        if hasattr(self.agent, "get_parameter_breakdown"):
            return self.agent.get_parameter_breakdown().to_dict()
        return {"trainable_params_total": count_trainable_params(self.agent)}

    def _create_optimization_diagnostics_probe(self):
        optimization_config = self.reporting_config.get("optimization_diagnostics", {})
        sample_every_optimizer_steps = int(
            optimization_config.get("sample_every_optimizer_steps", 64)
        )
        return OptimizationDiagnosticsProbe(
            agent=self.agent,
            optimizer=self.optimizer,
            max_grad_norm=self.max_grad_norm,
            sample_every_optimizer_steps=sample_every_optimizer_steps,
            gradient_clipping_mode=self.gradient_clipping_mode,
            group_max_grad_norms=self._branch_max_grad_norms(),
        )

    def _setup_reporting(self) -> None:
        """Create default reporting artifacts and monitors.

        Reporting is enabled by default in SimplyQRL 1.5. Disable with:
            reporting_config={"enabled": False}
        """
        if not self.reporting_enabled:
            self.monitor_manager = MonitorManager(self.external_monitors)
            return

        effective_config = self._effective_config()

        output_dir = self.reporting_config.get("output_dir", "runs")
        experiment_name = self.reporting_config.get("experiment_name")
        regime = self.reporting_config.get("regime", self.agent_type)
        hardware_profile_id = self.reporting_config.get("hardware_profile_id")
        report_run_name = self.reporting_config.get("run_name", self.run_name)

        hardware_profile_config = self.reporting_config.get("hardware_profile", False)
        if hardware_profile_config:
            hp_config = (
                hardware_profile_config
                if isinstance(hardware_profile_config, dict)
                else {}
            )
            hardware_profile_id, _, _ = ensure_hardware_profile(
                output_dir=output_dir,
                experiment_name=experiment_name,
                repo_root=hp_config.get("repo_root"),
                overwrite=bool(hp_config.get("overwrite", False)),
            )

        context = build_run_context(
            config=effective_config,
            output_dir=output_dir,
            run_name=report_run_name,
            experiment_name=experiment_name,
            regime=regime,
            hardware_profile_id=hardware_profile_id,
        )

        self.artifact_logger = ArtifactLogger(context, effective_config)
        monitors = [CoreTrainingMonitor(self.artifact_logger)]

        evaluation_config = self.reporting_config.get("evaluation", {})
        if evaluation_config.get("enabled", False):
            evaluation_monitor = EvaluationMonitor(
                self.artifact_logger,
                eval_freq=int(evaluation_config["eval_freq"]),
                eval_episodes=int(evaluation_config.get("eval_episodes", 5)),
                max_eval_steps=int(evaluation_config.get("max_eval_steps", 1_000)),
                deterministic=bool(evaluation_config.get("deterministic", True)),
            )
            self.evaluation_monitors.append(evaluation_monitor)
            monitors.append(evaluation_monitor)

        quantum_config = self.reporting_config.get("quantum", {})
        if quantum_config.get("enabled", False):
            monitors.append(
                QuantumMonitor(
                    self.artifact_logger,
                    agent=self.agent,
                    include_zeros=bool(quantum_config.get("include_zeros", True)),
                )
            )

        resources_config = self.reporting_config.get("resources", {})
        if resources_config.get("enabled", False):
            monitors.append(
                ResourceMonitor(
                    self.artifact_logger,
                    sample_every_updates=int(resources_config.get("sample_every_updates", 1)),
                    include_system=bool(resources_config.get("include_system", True)),
                    include_process=bool(resources_config.get("include_process", True)),
                    include_torch_cuda=bool(resources_config.get("include_torch_cuda", True)),
                    torch_cuda_device_index=self.device_policy.cuda_device_index,
                    use_codecarbon=bool(resources_config.get("use_codecarbon", False)),
                    codecarbon_output_file=str(
                        resources_config.get("codecarbon_output_file", "codecarbon.csv")
                    ),
                    codecarbon_project_name=resources_config.get("codecarbon_project_name"),
                    codecarbon_measure_power_secs=int(
                        resources_config.get("codecarbon_measure_power_secs", 1)
                    ),
                    codecarbon_log_level=str(
                        resources_config.get("codecarbon_log_level", "error")
                    ),
                    fail_on_codecarbon_error=bool(
                        resources_config.get("fail_on_codecarbon_error", False)
                    ),
                )
            )

        optimization_config = self.reporting_config.get(
            "optimization_diagnostics", {}
        )
        if optimization_config.get("enabled", False):
            sample_every_optimizer_steps = int(
                optimization_config.get("sample_every_optimizer_steps", 64)
            )
            self.optimization_diagnostics_probe = (
                self._create_optimization_diagnostics_probe()
            )
            monitors.append(
                OptimizationDiagnosticsMonitor(
                    self.artifact_logger,
                    sample_every_optimizer_steps=sample_every_optimizer_steps,
                )
            )

        monitors.extend(self.external_monitors)
        self.monitor_manager = MonitorManager(monitors)

    def _close_reporting(self, status: str = "finished", final_summary: dict[str, Any] | None = None):
        if self.artifact_logger is not None and not self._reporting_closed:
            self.artifact_logger.close(status=status, final_summary=final_summary)
            self._reporting_closed = True

    def _close_envs(self):
        if not self._envs_closed:
            self.envs.close()
            self._envs_closed = True

    def _make_eval_env(self):
        """Create a fresh evaluation environment."""
        if self.env is None:
            return gym.make(self.env_id)

        if isinstance(self.env, type):
            return self.env()

        if callable(self.env):
            return self.env()

        if isinstance(self.env, gym.Env):
            if self.env.spec is not None and self.env.spec.id is not None:
                return gym.make(self.env.spec.id)

        raise ValueError(
            "Cannot create evaluation environment from the provided env. "
            "Pass env as an environment id, class or factory when using EvaluationMonitor."
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, total_timesteps: int = 100_000, progress_bar: bool = False):
        """Main PPO training loop."""
        next_obs_np, _ = self.envs.reset(seed=self.seed)
        next_obs = self._prepare_obs(next_obs_np)
        next_done = torch.zeros(self.num_envs, device=self.device)

        start_time = time.time()
        self.num_iterations = math.ceil(total_timesteps / self.batch_size)

        self.monitor_manager.on_training_start(
            {
                "global_step": self.global_step,
                "total_timesteps": total_timesteps,
                "seed": self.seed,
                "env_id": self._env_label(),
                "agent_type": self.agent_type,
                "trainable_params": self._get_trainable_parameter_report(),
                "device": self._device_metadata(),
            }
        )

        pbar = tqdm(total=total_timesteps, desc="Training", unit="step") if progress_bar else None
        training_status = "finished"

        try:
            for iteration in range(1, self.num_iterations + 1):
                # Anneal learning rate
                if self.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / self.num_iterations
                    self._set_annealed_learning_rates(frac)

                rollout_start_time = time.time()
                self.monitor_manager.on_rollout_start(
                    {
                        "global_step": self.global_step,
                        "update": iteration,
                    }
                )

                # Collect experience
                for step in range(self.num_steps):
                    self.global_step += self.num_envs
                    self.obs[step] = next_obs
                    self.dones[step] = next_done

                    with torch.no_grad():
                        action, logprob, _, value = self.agent.get_action_and_value(next_obs)

                    self.actions[step] = action
                    self.logprobs[step] = logprob
                    self.values[step] = value.flatten()

                    next_obs_np, reward, terminated, truncated, infos = self.envs.step(
                        action.cpu().numpy()
                    )

                    next_done_np = np.logical_or(terminated, truncated)
                    self.rewards[step] = torch.as_tensor(
                        reward,
                        dtype=torch.float32,
                        device=self.device,
                    ).view(-1)

                    next_obs = self._prepare_obs(next_obs_np)
                    next_done = torch.as_tensor(
                        next_done_np,
                        dtype=torch.float32,
                        device=self.device,
                    )

                    if pbar is not None:
                        pbar.update(self.num_envs)

                    # Episode metrics
                    if "episode" in infos and "_episode" in infos:
                        for i, done in enumerate(infos["_episode"]):
                            if done:
                                ep_return = float(infos["episode"]["r"][i])
                                ep_length = int(infos["episode"]["l"][i])

                                self.episode_counter += 1
                                self.monitor_manager.on_episode_end(
                                    {
                                        "global_step": self.global_step,
                                        "episode_return": ep_return,
                                        "episode_length": ep_length,
                                    }
                                )

                rollout_time = time.time() - rollout_start_time
                self.monitor_manager.on_rollout_end(
                    {
                        "global_step": self.global_step,
                        "update": iteration,
                        "rollout_time": rollout_time,
                    }
                )

                update_start_time = time.time()
                self.monitor_manager.on_update_start(
                    {
                        "global_step": self.global_step,
                        "update": iteration,
                    }
                )

                # Compute GAE & returns
                with torch.no_grad():
                    next_value = self.agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(self.rewards, device=self.device)
                    lastgaelam = 0

                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            nextvalues = self.values[t + 1]

                        delta = (
                            self.rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - self.values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                        )

                    returns = advantages + self.values

                # Flatten rollout batch while preserving observation shape for SimplyQRL 1.5 agents.
                b_obs = self._flatten_rollout_obs()
                b_logprobs = self.logprobs.reshape(-1)
                b_actions = self.actions.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = self.values.reshape(-1)

                # Optimize policy + value
                b_inds = np.arange(self.batch_size)
                clipfracs = []
                optimizer_steps_this_update = 0
                last_grad_norm = None

                # Values assigned during optimization and logged after the update
                pg_loss = None
                v_loss = None
                entropy_loss = None
                loss = None
                approx_kl = torch.tensor(0.0)
                old_approx_kl = torch.tensor(0.0)

                for _epoch in range(self.update_epochs):
                    np.random.shuffle(b_inds)

                    for start in range(0, self.batch_size, self.minibatch_size):
                        end = start + self.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                            b_obs[mb_inds],
                            b_actions.long()[mb_inds],
                        )

                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs.append(
                                ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                            )

                        mb_advantages = b_advantages[mb_inds]
                        if self.norm_adv:
                            mb_advantages = (
                                (mb_advantages - mb_advantages.mean())
                                / (mb_advantages.std() + 1e-8)
                            )

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio,
                            1 - self.clip_coef,
                            1 + self.clip_coef,
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -self.clip_coef,
                                self.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        # Entropy
                        entropy_loss = entropy.mean()

                        # Final loss
                        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                        self.optimizer.zero_grad()
                        loss.backward()

                        diagnostics_sampled = False
                        if self.optimization_diagnostics_probe is not None:
                            diagnostics_sampled = (
                                self.optimization_diagnostics_probe.begin_step(
                                    update=iteration,
                                    epoch=_epoch + 1,
                                    minibatch=(start // self.minibatch_size) + 1,
                                    global_step=self.global_step,
                                    losses={
                                        "loss_policy_minibatch": pg_loss.item(),
                                        "loss_value_minibatch": v_loss.item(),
                                        "loss_entropy_minibatch": entropy_loss.item(),
                                        "loss_total_minibatch": loss.item(),
                                    },
                                )
                            )

                        last_grad_norm, returned_group_norms = self._apply_gradient_clipping()
                        if diagnostics_sampled:
                            self.optimization_diagnostics_probe.capture_post_clip(
                                last_grad_norm,
                                returned_group_norms=returned_group_norms,
                            )

                        self.monitor_manager.on_backward_end(
                            {
                                "global_step": self.global_step,
                                "update": iteration,
                                "grad_norm": float(last_grad_norm),
                            }
                        )
                        self.optimizer.step()
                        optimizer_steps_this_update += 1

                        if diagnostics_sampled:
                            diagnostics_state = (
                                self.optimization_diagnostics_probe.finish_step()
                            )
                            if diagnostics_state is not None:
                                self.monitor_manager.on_optimizer_step_end(
                                    diagnostics_state
                                )

                    if self.target_kl is not None and approx_kl > self.target_kl:
                        break

                # Update metrics
                y_pred = b_values.detach().cpu().numpy()
                y_true = b_returns.detach().cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                update_time = time.time() - update_start_time
                elapsed = max(time.time() - start_time, 1e-12)
                sps = int(self.global_step / elapsed)

                branch_learning_rates = self._current_branch_learning_rates()
                self.monitor_manager.on_update_end(
                    {
                        "global_step": self.global_step,
                        "update": iteration,
                        "rollout_time": rollout_time,
                        "update_time": update_time,
                        "steps_per_second": sps,
                        "loss_policy": pg_loss.item() if pg_loss is not None else None,
                        "loss_value": v_loss.item() if v_loss is not None else None,
                        "loss_entropy": entropy_loss.item() if entropy_loss is not None else None,
                        "loss_total": loss.item() if loss is not None else None,
                        "old_approx_kl": old_approx_kl.item(),
                        "approx_kl": approx_kl.item(),
                        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                        "explained_variance": explained_var,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "learning_rate_mode": self.learning_rate_mode,
                        "learning_rate_actor": branch_learning_rates["actor"],
                        "learning_rate_critic": branch_learning_rates["critic"],
                        "learning_rate_shared": branch_learning_rates["shared"],
                        "optimizer_steps": optimizer_steps_this_update,
                        "grad_norm": float(last_grad_norm) if last_grad_norm is not None else None,
                    }
                )

                for evaluation_monitor in self.evaluation_monitors:
                    if evaluation_monitor.should_evaluate(self.global_step):
                        eval_state = evaluation_monitor.evaluate(
                            agent=self.agent,
                            env_factory=self._make_eval_env,
                            device=self.device,
                            global_step=self.global_step,
                            seed=self.seed,
                        )
                        self.monitor_manager.on_evaluation_end(eval_state)

        except Exception as exc:
            training_status = "failed"
            self.monitor_manager.on_training_error(
                {
                    "global_step": self.global_step,
                    "error": repr(exc),
                }
            )
            raise

        finally:
            wall_clock_elapsed = time.time() - start_time

            if training_status == "finished":
                self.monitor_manager.on_training_end(
                    {
                        "global_step": self.global_step,
                        "wall_clock_elapsed": wall_clock_elapsed,
                    }
                )

            if pbar is not None:
                pbar.close()

            self._close_envs()
            self._close_reporting(status=training_status)

        print("Training finished!")

    # ------------------------------------------------------------------
    # Hyperparameter export
    # ------------------------------------------------------------------
    def _export_hparams(self):
        return dict(
            env_id=self.env_id,
            seed=self.seed,
            agent_type=self.agent_type,
            learning_rate=self.learning_rate,
            learning_rate_mode=self.learning_rate_mode,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            shared_learning_rate=self.shared_learning_rate,
            num_envs=self.num_envs,
            num_steps=self.num_steps,
            anneal_lr=self.anneal_lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            num_minibatches=self.num_minibatches,
            update_epochs=self.update_epochs,
            norm_adv=self.norm_adv,
            clip_coef=self.clip_coef,
            clip_vloss=self.clip_vloss,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            gradient_clipping_mode=self.gradient_clipping_mode,
            actor_max_grad_norm=self.actor_max_grad_norm,
            critic_max_grad_norm=self.critic_max_grad_norm,
            shared_max_grad_norm=self.shared_max_grad_norm,
            target_kl=self.target_kl,
            device_policy={
                "torch_device": self.device_policy.requested_torch_device,
                "allow_cuda_fallback": self.device_policy.allow_cuda_fallback,
            },
        )

    # ------------------------------------------------------------------
    # Persistence helpers: secure (JSON+zip) + fallback (pickle)
    # ------------------------------------------------------------------
    def save(self, file_path: str, *, metadata: dict | None = None):
        tmp_dir = tempfile.mkdtemp()

        torch.save(self.agent.state_dict(), f"{tmp_dir}/policy.pth")
        torch.save(self.optimizer.state_dict(), f"{tmp_dir}/optim.pth")

        try:
            json.dumps(self.agent_config)
            with open(f"{tmp_dir}/agent_cfg.json", "w") as fcfg:
                json.dump(self.agent_config, fcfg)
            cfg_fmt = "json"
        except TypeError:
            with open(f"{tmp_dir}/agent_cfg.pkl", "wb") as fcfg:
                cloudpickle.dump(self.agent_config, fcfg)
            cfg_fmt = "pkl"

        meta = {
            "global_step": self.global_step,
            "hparams": self._export_hparams(),
            "cfg_format": cfg_fmt,
            **(metadata or {}),
        }

        with open(f"{tmp_dir}/meta.json", "w") as f:
            json.dump(meta, f)

        shutil.make_archive(file_path, "zip", tmp_dir)
        shutil.rmtree(tmp_dir)
        print(f"Saved checkpoint: {file_path}.zip")

    @classmethod
    def load(
        cls,
        file_path: str,
        env=None,
        device: str = "cpu",
        allow_pickle: bool = False,
        reporting_config: dict | None = None,
        run_name: str | None = None,
        hparams_overrides: dict | None = None,
        load_optimizer_state: bool = True,
    ):
        """Load a SimplyQRL ZIP checkpoint.

        Normal continuation restores policy and Adam state. Staged composition
        should pass ``load_optimizer_state=False`` because actor/critic modules
        are replaced and the optimizer must be rebuilt from the active config.
        """

        if not file_path.endswith(".zip"):
            file_path += ".zip"

        with zipfile.ZipFile(file_path, "r") as zf:
            meta = json.loads(zf.read("meta.json"))
            hparams = meta["hparams"].copy()

            fmt = meta.get("cfg_format", "json")
            if fmt == "json":
                cfg = json.loads(zf.read("agent_cfg.json"))
            else:
                if not allow_pickle:
                    raise RuntimeError(
                        "Checkpoint uses pickle for agent_config; pass allow_pickle=True to load it."
                    )
                cfg = cloudpickle.loads(zf.read("agent_cfg.pkl"))

            if hparams_overrides is not None:
                allowed_override_keys = {
                    "seed",
                    "learning_rate",
                    "learning_rate_mode",
                    "actor_learning_rate",
                    "critic_learning_rate",
                    "shared_learning_rate",
                    "num_envs",
                    "num_steps",
                    "anneal_lr",
                    "gamma",
                    "gae_lambda",
                    "num_minibatches",
                    "update_epochs",
                    "norm_adv",
                    "clip_coef",
                    "clip_vloss",
                    "ent_coef",
                    "vf_coef",
                    "max_grad_norm",
                    "gradient_clipping_mode",
                    "actor_max_grad_norm",
                    "critic_max_grad_norm",
                    "shared_max_grad_norm",
                    "target_kl",
                }
                unknown_override_keys = set(hparams_overrides) - allowed_override_keys
                if unknown_override_keys:
                    raise KeyError(
                        "Unsupported PPO.load hparams_overrides keys: "
                        + ", ".join(sorted(unknown_override_keys))
                    )
                hparams.update(dict(hparams_overrides))

            saved_device_policy = hparams.pop("device_policy", None)
            if saved_device_policy is not None:
                saved_device_policy = dict(saved_device_policy)
                # Backward compatibility with historical experimental checkpoints:
                # qnode_torch_device is no longer a user-facing policy key.
                saved_device_policy.pop("qnode_torch_device", None)

            if device is not None:
                requested_load_device = str(device)
                if saved_device_policy is None:
                    saved_device_policy = {"torch_device": requested_load_device}
                else:
                    saved_device_policy["torch_device"] = requested_load_device

            kwargs = dict(
                env=env,
                run_name=run_name or "PPO-loaded",
                agent_config=cfg,
                reporting_config=(
                    reporting_config
                    if reporting_config is not None
                    else {"enabled": False}
                ),
                device_policy=saved_device_policy,
                **hparams,
            )

            obj = cls(**kwargs)

            obj.agent.load_state_dict(
                torch.load(BytesIO(zf.read("policy.pth")), map_location=device)
            )
            if load_optimizer_state:
                obj.optimizer.load_state_dict(
                    torch.load(BytesIO(zf.read("optim.pth")), map_location=device)
                )
            obj.global_step = meta["global_step"]

        print(f"Loaded checkpoint: {file_path}")
        return obj

    def close(self):
        self._close_envs()
        self._close_reporting(status="finished")
