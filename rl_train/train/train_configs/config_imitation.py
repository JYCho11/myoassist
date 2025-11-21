from rl_train.train.train_configs.config import TrainSessionConfigBase
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ImitationTrainSessionConfig(TrainSessionConfigBase):
    @dataclass
    class AutoRewardAdjustParams:
        learning_rate: float = 0.001
    auto_reward_adjust_params: AutoRewardAdjustParams = field(default_factory=AutoRewardAdjustParams)
    
    # ========== CURRICULUM LEARNING PARAMS (모듈화) ==========
    @dataclass
    class CurriculumParams:
        """Curriculum learning configuration - can be enabled/disabled via JSON"""
        enabled: bool = False
        phase1_end: float = 0.3
        phase2_end: float = 0.7
        forward_weight_schedule: list = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.3])
        imitation_pos_weight_schedule: list = field(default_factory=lambda: [0.2, 0.4, 0.8, 1.0])
        imitation_vel_weight_schedule: list = field(default_factory=lambda: [0.2, 0.4, 0.8, 1.0])
    curriculum_params: CurriculumParams = field(default_factory=CurriculumParams)
    
    # ========== WANDB LOGGING PARAMS (모듈화) ==========
    @dataclass
    class WandbParams:
        """WandB logging configuration - can be enabled/disabled via JSON"""
        enabled: bool = False
        project: str = "myoassist-imitation"
        run_name: Optional[str] = None
        entity: Optional[str] = None
        tags: list = field(default_factory=list)
        notes: str = ""
    wandb_params: WandbParams = field(default_factory=WandbParams)
    # ========== END NEW PARAMS ==========

    @dataclass
    class EnvParams(TrainSessionConfigBase.EnvParams):
        @dataclass
        class RewardWeights(TrainSessionConfigBase.EnvParams.RewardWeights):
            qpos_imitation_rewards:dict = field(default_factory=dict)
            qvel_imitation_rewards:dict = field(default_factory=dict)
            
            end_effector_imitation_reward: float = 0.3

            

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        flag_random_ref_index: bool = False
        out_of_trajectory_threshold: float = 1
        reference_data_path: str = ""

        reference_data_keys: list[str] = field(default_factory=list[str])
    env_params: EnvParams = field(default_factory=EnvParams)