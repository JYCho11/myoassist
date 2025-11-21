"""
Curriculum Learning Callback for MyoAssist Imitation Learning
==============================================================

Implements curriculum learning strategy:
- Phase 1 (Early): Focus on balancing and forward progression
- Phase 2 (Mid): Gradually increase imitation reward weight
- Phase 3 (Late): Focus on reference motion tracking

Strategy:
- Initial: High weight on forward_reward, low on imitation
- Progress: Gradually shift weight to imitation rewards
- Final: High weight on imitation, maintain forward progress
"""

import numpy as np
from typing import Dict
from rl_train.utils.learning_callback import BaseCustomLearningCallback
from rl_train.envs.myoassist_leg_imitation import ImitationCustomLearningCallback
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb


class CurriculumImitationCallback(ImitationCustomLearningCallback):
    """
    Curriculum Learning Callback with WandB integration
    
    Curriculum Strategy:
    - Phase 1 (0-30%): Balancing focus
      - forward_reward: 1.0 → 0.5
      - qpos_imitation: 0.1 → 0.3
      - qvel_imitation: 0.1 → 0.3
      
    - Phase 2 (30-70%): Transition phase  
      - forward_reward: 0.5 → 0.2
      - qpos_imitation: 0.3 → 0.6
      - qvel_imitation: 0.3 → 0.6
      
    - Phase 3 (70-100%): Imitation focus
      - forward_reward: 0.2 (maintain)
      - qpos_imitation: 0.6 → 0.8
      - qvel_imitation: 0.6 → 0.8
    """
    
    def __init__(self,
                 *,
                 log_rollout_freq: int,
                 evaluate_freq: int,
                 log_handler: train_log_handler.TrainLogHandler,
                 original_reward_weights: ImitationTrainSessionConfig.EnvParams.RewardWeights,
                 auto_reward_adjust_params: ImitationTrainSessionConfig.AutoRewardAdjustParams,
                 total_timesteps: int,
                 curriculum_config: Dict = None,
                 use_wandb: bool = True,
                 wandb_project: str = "myoassist-curriculum",
                 wandb_run_name: str = None,
                 verbose=1):
        """
        Initialize Curriculum Learning Callback
        
        Args:
            curriculum_config: Dict with curriculum parameters
                - phase1_end: End of phase 1 (default: 0.3)
                - phase2_end: End of phase 2 (default: 0.7)
                - forward_weight_schedule: [start, phase1_end, phase2_end, final]
                - imitation_weight_schedule: [start, phase1_end, phase2_end, final]
            use_wandb: Enable WandB logging
            wandb_project: WandB project name
            wandb_run_name: WandB run name
        """
        super().__init__(
            log_rollout_freq=log_rollout_freq,
            evaluate_freq=evaluate_freq,
            log_handler=log_handler,
            original_reward_weights=original_reward_weights,
            auto_reward_adjust_params=auto_reward_adjust_params,
            verbose=verbose
        )
        
        self.total_timesteps = total_timesteps
        self.use_wandb = use_wandb
        
        # Default curriculum configuration
        self.curriculum_config = curriculum_config or {
            'phase1_end': 0.3,
            'phase2_end': 0.7,
            'forward_weight_schedule': [1.0, 0.5, 0.2, 0.2],
            'imitation_pos_weight_schedule': [0.1, 0.3, 0.6, 0.8],
            'imitation_vel_weight_schedule': [0.1, 0.3, 0.6, 0.8],
        }
        
        # Store original weights for reference
        self.original_weights = {}
        for key in dir(original_reward_weights):
            if not key.startswith('_'):
                self.original_weights[key] = getattr(original_reward_weights, key)
        
        # WandB initialization
        print(f"[DEBUG] use_wandb={self.use_wandb}, wandb_project={wandb_project}, wandb_run_name={wandb_run_name}")
        if self.use_wandb:
            print(f"[WandB] Initializing WandB...")
            print(f"[WandB]   Project: {wandb_project}")
            print(f"[WandB]   Run Name: {wandb_run_name}")
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,  # Can be None, wandb will auto-generate
                config={
                    'total_timesteps': total_timesteps,
                    'curriculum_config': self.curriculum_config,
                    'original_reward_weights': self.original_weights,
                },
                sync_tensorboard=True
            )
            print(f"[WandB] Initialization complete! Run URL: {wandb.run.url}")
        else:
            print(f"[WandB] Disabled (use_wandb={self.use_wandb})")
    
    def _get_curriculum_progress(self) -> float:
        """Get current curriculum progress (0.0 to 1.0)"""
        return min(1.0, self.num_timesteps / self.total_timesteps)
    
    def _interpolate_weight(self, schedule: list, progress: float) -> float:
        """
        Interpolate weight based on curriculum schedule
        
        Args:
            schedule: [start, phase1_end, phase2_end, final]
            progress: Current progress (0.0 to 1.0)
        """
        phase1_end = self.curriculum_config['phase1_end']
        phase2_end = self.curriculum_config['phase2_end']
        
        if progress <= phase1_end:
            # Phase 1: start → phase1_end
            t = progress / phase1_end
            return schedule[0] + t * (schedule[1] - schedule[0])
        elif progress <= phase2_end:
            # Phase 2: phase1_end → phase2_end
            t = (progress - phase1_end) / (phase2_end - phase1_end)
            return schedule[1] + t * (schedule[2] - schedule[1])
        else:
            # Phase 3: phase2_end → final
            t = (progress - phase2_end) / (1.0 - phase2_end)
            return schedule[2] + t * (schedule[3] - schedule[2])
    
    def _update_curriculum_weights(self):
        """Update reward weights based on curriculum progress"""
        progress = self._get_curriculum_progress()
        
        # Get interpolated weights
        forward_weight = self._interpolate_weight(
            self.curriculum_config['forward_weight_schedule'],
            progress
        )
        imitation_pos_weight = self._interpolate_weight(
            self.curriculum_config['imitation_pos_weight_schedule'],
            progress
        )
        imitation_vel_weight = self._interpolate_weight(
            self.curriculum_config['imitation_vel_weight_schedule'],
            progress
        )
        
        # Update forward reward weight
        self._reward_weights.forward_reward = forward_weight
        
        # Update imitation weights for all joints
        for key in dir(self._reward_weights.qpos_imitation_rewards):
            if not key.startswith('_'):
                original_value = self.original_weights['qpos_imitation_rewards'].get(key, 0.0)
                if original_value > 0:
                    setattr(
                        self._reward_weights.qpos_imitation_rewards,
                        key,
                        original_value * imitation_pos_weight
                    )
        
        for key in dir(self._reward_weights.qvel_imitation_rewards):
            if not key.startswith('_'):
                original_value = self.original_weights['qvel_imitation_rewards'].get(key, 0.0)
                if original_value > 0:
                    setattr(
                        self._reward_weights.qvel_imitation_rewards,
                        key,
                        original_value * imitation_vel_weight
                    )
        
        # Update environments with new weights
        subprocvec_env: SubprocVecEnv = self.model.get_env()
        subprocvec_env.env_method('set_reward_weights', self._reward_weights)
        
        return {
            'curriculum/progress': progress,
            'curriculum/forward_weight': forward_weight,
            'curriculum/imitation_pos_weight': imitation_pos_weight,
            'curriculum/imitation_vel_weight': imitation_vel_weight,
        }
    
    def _init_callback(self):
        """Initialize callback"""
        super()._init_callback()
        
        # Initialize curriculum
        if self.verbose > 0:
            print("="*60)
            print("Curriculum Learning Initialized")
            print(f"Total timesteps: {self.total_timesteps}")
            print(f"Phase 1 (Balancing): 0% - {self.curriculum_config['phase1_end']*100:.0f}%")
            print(f"Phase 2 (Transition): {self.curriculum_config['phase1_end']*100:.0f}% - {self.curriculum_config['phase2_end']*100:.0f}%")
            print(f"Phase 3 (Imitation): {self.curriculum_config['phase2_end']*100:.0f}% - 100%")
            print("="*60)
    
    def _on_step(self) -> bool:
        """Called after each environment step"""
        # Update curriculum weights
        curriculum_metrics = self._update_curriculum_weights()
        
        # ========== LOG TO WANDB (EVERY STEP) ==========
        if self.use_wandb and wandb.run is not None:
            # Collect basic metrics
            log_dict = {**curriculum_metrics}
            
            # Add current reward accumulation (cumulative during rollout)
            # Apply weights to show actual contribution to total reward
            total_reward_live = 0.0
            if hasattr(self, 'reward_accumulate') and self.reward_accumulate:
                for key, value in self.reward_accumulate.items():
                    if isinstance(value, dict):
                        # Nested dict (e.g., qpos_imitation_rewards)
                        for sub_key, sub_value in value.items():
                            # Apply weight to show actual contribution
                            weight = self._reward_weights[key].get(sub_key, 1.0) if hasattr(self._reward_weights[key], 'get') else 1.0
                            weighted_value = sub_value * weight
                            log_dict[f'reward_live/{key}/{sub_key}'] = weighted_value
                            log_dict[f'reward_live_raw/{key}/{sub_key}'] = sub_value  # Also log raw value
                            total_reward_live += weighted_value
                    else:
                        # Scalar value
                        weight = self._reward_weights.get(key, 1.0) if hasattr(self._reward_weights, 'get') else 1.0
                        weighted_value = value * weight
                        log_dict[f'reward_live/{key}'] = weighted_value
                        log_dict[f'reward_live_raw/{key}'] = value  # Also log raw value
                        total_reward_live += weighted_value
                
                # Add total reward
                log_dict['reward_live/total'] = total_reward_live
            
            # Add training metrics if available from logger
            if self.logger is not None and hasattr(self.logger, 'name_to_value'):
                for key, value in self.logger.name_to_value.items():
                    # Log all metrics (train, rollout, time)
                    if isinstance(value, (int, float)):
                        log_dict[key] = value
            
            wandb.log(log_dict, step=self.num_timesteps)
        # ========== END WANDB LOGGING ==========
        
        # Call parent on_step
        return super()._on_step()
    
    def _on_rollout_end(self, write_log: bool = True):
        """Called at the end of each rollout"""
        log_data = super()._on_rollout_end(write_log=write_log)
        
        if log_data is None:
            return
        
        # ========== CALCULATE REWARD COMPONENTS (FLATTEN NESTED DICTS) ==========
        reward_components = {}
        total_reward = 0.0
        
        print(f"\n[DEBUG REWARD ROLLOUT] ============================================")
        print(f"[DEBUG REWARD] reward_accumulate keys: {list(self.reward_accumulate.keys())}")
        print(f"[DEBUG REWARD] reward_accumulate length: {len(self.reward_accumulate)}")
        
        for key, value in self.reward_accumulate.items():
            if isinstance(value, dict):
                # Nested dict (e.g., qpos_imitation_rewards)
                dict_sum_raw = sum(value.values())
                dict_sum_weighted = 0.0
                print(f"[DEBUG REWARD] {key} (dict): raw_sum={dict_sum_raw:.2f}, items={len(value)}")
                for sub_key, sub_value in value.items():
                    # Apply weight
                    weight = self._reward_weights[key].get(sub_key, 1.0) if hasattr(self._reward_weights[key], 'get') else 1.0
                    weighted_value = sub_value * weight
                    dict_sum_weighted += weighted_value
                    reward_components[f'reward/{key}/{sub_key}'] = weighted_value
                    reward_components[f'reward_raw/{key}/{sub_key}'] = sub_value  # Also log raw
                    total_reward += weighted_value
                print(f"[DEBUG REWARD] {key} weighted_sum={dict_sum_weighted:.6f}")
            else:
                # Scalar value (e.g., forward_reward, muscle_activation_penalty)
                weight = self._reward_weights.get(key, 1.0) if hasattr(self._reward_weights, 'get') else 1.0
                weighted_value = value * weight
                print(f"[DEBUG REWARD] {key} (scalar): raw={value:.4f}, weight={weight}, weighted={weighted_value:.6f}")
                reward_components[f'reward/{key}'] = weighted_value
                reward_components[f'reward_raw/{key}'] = value  # Also log raw
                total_reward += weighted_value
        
        reward_components['reward/total'] = total_reward
        print(f"[DEBUG REWARD] Total reward: {total_reward:.2f}, components: {len(reward_components)}")
        print(f"[DEBUG REWARD] ============================================\n")
        # ========== END REWARD COMPONENTS ==========
        
        # Get curriculum metrics
        progress = self._get_curriculum_progress()
        curriculum_metrics = {
            'curriculum/progress': progress,
            'curriculum/phase': 1 if progress < self.curriculum_config['phase1_end'] 
                               else (2 if progress < self.curriculum_config['phase2_end'] else 3),
        }
        
        # Get training metrics from locals
        if hasattr(self, 'locals') and 'infos' in self.locals:
            # Extract episode info
            for info in self.locals['infos']:
                if 'episode' in info:
                    reward_components['episode/reward'] = info['episode']['r']
                    reward_components['episode/length'] = info['episode']['l']
        
        # ========== COLLECT TRAINING METRICS FROM LOGGER ==========
        # Get all metrics from stable-baselines3 logger
        training_metrics = {}
        if self.logger is not None:
            print(f"[DEBUG] Logger available, keys: {list(self.logger.name_to_value.keys())[:5]}")  # Show first 5 keys
            # Extract from logger's name_to_value dict
            for key in self.logger.name_to_value.keys():
                try:
                    value = self.logger.name_to_value[key]
                    # Include train/, rollout/, time/ metrics
                    if any(prefix in key for prefix in ['train/', 'rollout/', 'time/']):
                        training_metrics[key] = value
                except:
                    pass
            print(f"[DEBUG] Collected {len(training_metrics)} training metrics")
        else:
            print(f"[DEBUG] Logger is None!")
        # ========== END TRAINING METRICS COLLECTION ==========
        
        # Log to WandB
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                **reward_components,
                **curriculum_metrics,
                **training_metrics,  # Add training metrics (loss, policy_grad, etc)
                'time/total_timesteps': self.num_timesteps,
            }, step=self.num_timesteps)
        
        return log_data


def create_curriculum_callback(
    log_rollout_freq: int,
    evaluate_freq: int,
    log_handler: train_log_handler.TrainLogHandler,
    reward_weights: ImitationTrainSessionConfig.EnvParams.RewardWeights,
    auto_reward_adjust_params: ImitationTrainSessionConfig.AutoRewardAdjustParams,
    total_timesteps: int,
    curriculum_config: Dict = None,
    use_wandb: bool = True,
    wandb_project: str = "myoassist-curriculum",
    wandb_run_name: str = None,
    verbose: int = 1
) -> CurriculumImitationCallback:
    """
    Factory function to create curriculum callback
    
    Usage:
        callback = create_curriculum_callback(
            log_rollout_freq=8,
            evaluate_freq=16,
            log_handler=log_handler,
            reward_weights=reward_weights,
            auto_reward_adjust_params=auto_adjust_params,
            total_timesteps=3e7,
            use_wandb=True,
            wandb_run_name="S004_curriculum_v1"
        )
    """
    return CurriculumImitationCallback(
        log_rollout_freq=log_rollout_freq,
        evaluate_freq=evaluate_freq,
        log_handler=log_handler,
        original_reward_weights=reward_weights,
        auto_reward_adjust_params=auto_reward_adjust_params,
        total_timesteps=total_timesteps,
        curriculum_config=curriculum_config,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        verbose=verbose
    )
