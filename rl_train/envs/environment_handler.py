import numpy as np
import json
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from myosuite.utils import gym
from rl_train.utils.data_types import DictionableDataclass
import os
from rl_train.train.train_configs.config import TrainSessionConfigBase
class EnvironmentHandler:
    @staticmethod
    def create_environment(config, is_rendering_on:bool, is_evaluate_mode:bool = False, use_dummy_vec_env:bool = False):

        ref_data_dict = EnvironmentHandler.load_reference_data(config)
    
        # Base gym.make arguments
        gym_make_args = {
            'seed': config.env_params.seed,
            'model_path': config.env_params.model_path,
            'env_params': config.env_params,
            'is_evaluate_mode': is_evaluate_mode
        }
        
        # Add reference_data only if it exists
        if ref_data_dict is not None:
            gym_make_args['reference_data'] = ref_data_dict
        
        try:
            if is_rendering_on or config.env_params.num_envs == 1:
                print(f"{config.env_params.env_id=}")
                env = gym.make(config.env_params.env_id, **gym_make_args).unwrapped
                if is_rendering_on:
                    env.mujoco_render_frames = True
                config.env_params.num_envs = 1
                config.ppo_params.n_steps = config.ppo_params.batch_size
            else:
                # DummyVecEnv: threading 기반 (live rendering 가능, 약간 느림)
                # SubprocVecEnv: multiprocessing 기반 (live rendering 불가, 빠름)
                vec_env_cls = DummyVecEnv if use_dummy_vec_env else SubprocVecEnv
                env = vec_env_cls([lambda: (gym.make(config.env_params.env_id, 
                                                    **gym_make_args)).unwrapped 
                                for _ in range(config.env_params.num_envs)])
                if use_dummy_vec_env:
                    print(f"[ENV] Using DummyVecEnv (threading) - live rendering ENABLED")
        except Exception as e:
            new_message = str(e)[:1000]
            e.args = (new_message,)
            raise e
        return env

    @staticmethod
    def load_reference_data(config):
        # Check if config has reference_data_path attribute
        print("===================================================================")
        if not hasattr(config.env_params, 'reference_data_path'):
            print("No reference data path provided.")
            print("===================================================================")
            return None
            
        if not config.env_params.reference_data_path:
            print("No reference data path provided.")
            print("===================================================================")
            return None
        print(f"Loading reference data from {config.env_params.reference_data_path}")
        print("===================================================================")
        if config.env_params.reference_data_path.endswith(".npz"):
            ref_data_npz = np.load(config.env_params.reference_data_path, allow_pickle=True)
            ref_data_dict = {key: ref_data_npz[key].item() for key in ref_data_npz.files}
        elif config.env_params.reference_data_path.endswith(".json"):
            with open(config.env_params.reference_data_path, 'r') as f:
                ref_data_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use either .npz or .json.")

        if "resampled_series_data" not in ref_data_dict:
            ref_data_dict["resampled_series_data"] = {}
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(0, original_data_length - 1, original_data_length)

                new_sample_rate = config.env_params.control_framerate
                new_length = int(original_data_length * new_sample_rate / original_sample_rate)
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(new_x, original_x, ref_data_dict["series_data"][key])
                ref_data_dict["metadata"]["resampled_data_length"] = new_length
                ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate

        return ref_data_dict

    def get_config_type_from_session_id(session_id):
        # from rl_train.envs import myo_leg_18_reward_per_step
        from rl_train.train.train_configs.config import TrainSessionConfigBase
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
        # Create appropriate config based on env_id
        print(f"session_id: {session_id}")
        if session_id == 'myoAssistLeg-v0':
            return TrainSessionConfigBase
        elif session_id in ['myoAssistLegImitation-v0']:
            return ImitationTrainSessionConfig
        elif session_id == 'myoAssistLegImitationExo-v0':
            return ExoImitationTrainSessionConfig
        raise ValueError(f"Invalid session id: {session_id}")
        

    @staticmethod
    def get_session_config_from_path(config_path, class_type):
        print(f"Loading config from {config_path}")
        config_file_path = config_path
        with open(config_file_path, 'r') as f:
            config_dict = json.load(f)
            session_config = DictionableDataclass.create(class_type, config_dict)
        return session_config

    @staticmethod
    def get_callback(config, train_log_handler):
        from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
        
        from rl_train.envs import myoassist_leg_imitation
        from rl_train.utils import learning_callback
        
        if isinstance(config, ImitationTrainSessionConfig):
            # ========== ORIGINAL CODE (주석처리) ==========
            # custom_callback = myoassist_leg_imitation.ImitationCustomLearningCallback(
            #     log_rollout_freq=config.logger_params.logging_frequency,
            #     evaluate_freq=config.logger_params.evaluate_frequency,
            #     log_handler=train_log_handler,
            #     original_reward_weights=config.env_params.reward_keys_and_weights,
            #     auto_reward_adjust_params=config.auto_reward_adjust_params,
            # )
            # ========== END ORIGINAL CODE ==========
            
            # ========== NEW CODE: Curriculum Learning Support (모듈화) ==========
            # Check if curriculum learning is enabled
            use_curriculum = hasattr(config, 'curriculum_params') and getattr(config.curriculum_params, 'enabled', False)
            use_wandb = hasattr(config, 'wandb_params') and getattr(config.wandb_params, 'enabled', False)
            
            print(f"[DEBUG ENV_HANDLER] use_curriculum={use_curriculum}, use_wandb={use_wandb}")
            if hasattr(config, 'wandb_params'):
                print(f"[DEBUG ENV_HANDLER] wandb_params exists: enabled={config.wandb_params.enabled}, project={config.wandb_params.project}, run_name={config.wandb_params.run_name}")
            
            if use_curriculum:
                # Use curriculum learning callback
                from rl_train.utils.curriculum_callback import CurriculumImitationCallback
                
                # Prepare curriculum config
                curriculum_config = {
                    'phase1_end': config.curriculum_params.phase1_end,
                    'phase2_end': config.curriculum_params.phase2_end,
                    'forward_weight_schedule': config.curriculum_params.forward_weight_schedule,
                    'imitation_pos_weight_schedule': config.curriculum_params.imitation_pos_weight_schedule,
                    'imitation_vel_weight_schedule': config.curriculum_params.imitation_vel_weight_schedule,
                }
                
                # Prepare WandB config
                wandb_config = {}
                if use_wandb:
                    wandb_config = {
                        'project': config.wandb_params.project,
                        'run_name': config.wandb_params.run_name,
                    }
                    if hasattr(config.wandb_params, 'entity') and config.wandb_params.entity:
                        wandb_config['entity'] = config.wandb_params.entity
                
                print(f"Using Curriculum Learning Callback")
                print(f"  Phase 1: 0-{curriculum_config['phase1_end']*100:.0f}% (Balancing)")
                print(f"  Phase 2: {curriculum_config['phase1_end']*100:.0f}-{curriculum_config['phase2_end']*100:.0f}% (Transition)")
                print(f"  Phase 3: {curriculum_config['phase2_end']*100:.0f}-100% (Imitation)")
                if use_wandb:
                    print(f"  WandB: project={wandb_config.get('project')}, run={wandb_config.get('run_name')}")
                
                print(f"[DEBUG ENV_HANDLER] Creating CurriculumImitationCallback with use_wandb={use_wandb}")
                custom_callback = CurriculumImitationCallback(
                    log_rollout_freq=config.logger_params.logging_frequency,
                    evaluate_freq=config.logger_params.evaluate_frequency,
                    log_handler=train_log_handler,
                    original_reward_weights=config.env_params.reward_keys_and_weights,
                    auto_reward_adjust_params=config.auto_reward_adjust_params,
                    total_timesteps=int(config.total_timesteps),
                    curriculum_config=curriculum_config,
                    use_wandb=use_wandb,
                    wandb_project=wandb_config.get('project', 'myoassist-curriculum'),
                    wandb_run_name=wandb_config.get('run_name', None),
                )
            else:
                # Use standard imitation callback (original behavior)
                print(f"Using Standard Imitation Callback")
                custom_callback = myoassist_leg_imitation.ImitationCustomLearningCallback(
                    log_rollout_freq=config.logger_params.logging_frequency,
                    evaluate_freq=config.logger_params.evaluate_frequency,
                    log_handler=train_log_handler,
                    original_reward_weights=config.env_params.reward_keys_and_weights,
                    auto_reward_adjust_params=config.auto_reward_adjust_params,
                )
            # ========== END NEW CODE ==========
        else:
            custom_callback = learning_callback.BaseCustomLearningCallback(
                log_rollout_freq=config.logger_params.logging_frequency,
                evaluate_freq=config.logger_params.evaluate_frequency,
                log_handler=train_log_handler,
            )

        return custom_callback
    @staticmethod
    def get_stable_baselines3_model(config:TrainSessionConfigBase, env, trained_model_path:str|None=None, tensorboard_log:str|None=None):
        import stable_baselines3
        from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy
        from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy
        if config.env_params.env_id in ["myoAssistLegImitationExo-v0"]:
            policy_class = HumanExoActorCriticPolicy
            print(f"Using HumanExoActorCriticPolicy")
        else:
            policy_class = HumanActorCriticPolicy
            print(f"Using HumanActorCriticPolicy")
        if trained_model_path is not None:
            print(f"Loading trained model from {trained_model_path}")
            model = stable_baselines3.PPO.load(trained_model_path,
                                            env=env,
                                            custom_objects = {"policy_class": policy_class},
                                            tensorboard_log=tensorboard_log,
                                            )
        elif config.env_params.prev_trained_policy_path:
            print(f"Loading previous trained policy from {config.env_params.prev_trained_policy_path}")
            # when should I reset the (value)network?
            model = stable_baselines3.PPO.load(config.env_params.prev_trained_policy_path,
                                            env=env,
                                            custom_objects = {"policy_class": policy_class},

                                            # policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                                            verbose=2,
                                            tensorboard_log=tensorboard_log,
                                            **DictionableDataclass.to_dict(config.ppo_params),
                                            )
            # print(f"Resetting network: {config.custom_policy_params.reset_shared_net_after_load=}, {config.custom_policy_params.reset_policy_net_after_load=}, {config.custom_policy_params.reset_value_net_after_load=}")
            model.policy.reset_network(reset_shared_net=config.policy_params.custom_policy_params.reset_shared_net_after_load,
                                    reset_policy_net=config.policy_params.custom_policy_params.reset_policy_net_after_load,
                                    reset_value_net=config.policy_params.custom_policy_params.reset_value_net_after_load)
        else:
            model = stable_baselines3.PPO(
                policy=policy_class,
                env=env,
                policy_kwargs=DictionableDataclass.to_dict(config.policy_params),
                verbose=2,
                tensorboard_log=tensorboard_log,
                **DictionableDataclass.to_dict(config.ppo_params),
            )
        return model
    @staticmethod
    def updateconfig_from_model_policy(config, model):
        pass
        # config.policy_info.extractor_policy_net = f"{model.policy.mlp_extractor.policy_net}"
        # config.policy_info.extractor_value_net = f"{model.policy.mlp_extractor.value_net}"
        # config.policy_info.action_net = f"{model.policy.action_net}"
        # config.policy_info.value_net = f"{model.policy.value_net}"
        # config.policy_info.ortho_init = f"{model.policy.ortho_init}"
        # config.policy_info.share_features_extractor = f"{model.policy.share_features_extractor}"
