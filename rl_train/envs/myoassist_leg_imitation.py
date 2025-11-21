import collections
import numpy as np
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils import train_log_handler
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_train.utils.learning_callback import BaseCustomLearningCallback
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
import wandb
import mujoco
import mujoco.viewer as mjv
################################################################


class ImitationCustomLearningCallback(BaseCustomLearningCallback):
    
    def __init__(self, *,
                 log_rollout_freq: int,
                 evaluate_freq: int,
                 log_handler:train_log_handler.TrainLogHandler,
                 original_reward_weights:ImitationTrainSessionConfig.EnvParams.RewardWeights,
                 auto_reward_adjust_params:ImitationTrainSessionConfig.AutoRewardAdjustParams,
                 verbose=1):
        super().__init__(log_rollout_freq=log_rollout_freq, 
                         evaluate_freq=evaluate_freq,
                         log_handler=log_handler,
                         verbose=verbose)
        self._reward_weights = original_reward_weights
        self._auto_reward_adjust_params = auto_reward_adjust_params
        

    def _init_callback(self):
        super()._init_callback()

        self.reward_accumulate = DictionableDataclass.create(ImitationTrainSessionConfig.EnvParams.RewardWeights)
        self.reward_accumulate = DictionableDataclass.to_dict(self.reward_accumulate)
        for key in self.reward_accumulate.keys():
            self.reward_accumulate[key] = 0
    #called after all envs step done
    def _on_step(self) -> bool:
        # print("======================self.locals    ======================")
        # # pprint.pprint(self.locals)
        # print(f"DEBUG:: {len(self.locals['infos'])=}")
        # for idx, info in enumerate(self.locals['infos']):
        #     print(f"DEBUG:: {idx=} {info['rwd_dict']=}")
        # print("======================self.locals    ======================")

        subprocvec_env:SubprocVecEnv = self.model.get_env()
        # print(f"DEBUG:: {subprocvec_env=}")
        # print(f"DEBUG:: {subprocvec_env.env_method('subproc_env_test', 'This is param from learning callback')=}")
        for info in self.locals["infos"]:
            for key in self.reward_accumulate.keys():
                self.reward_accumulate[key] += info["rwd_dict"][key]
        

        super()._on_step()
            
        return True
    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def _on_rollout_end(self, write_log: bool = True) -> "ImitationTrainCheckpointData":
        log_data_base = super()._on_rollout_end(write_log=False)
        if log_data_base is None:
            return
        log_data = ImitationTrainCheckpointData(
            **log_data_base.__dict__,
            reward_weights=DictionableDataclass.to_dict(self._reward_weights),
            reward_accumulate=self.reward_accumulate.copy(),
        )
        if write_log:
            self.train_log_handler.add_log_data(log_data)
            self.train_log_handler.write_json_file()
        
        ################################################################################## wandb
        total_steps_in_rollout = self.training_env.num_envs * self.model.n_steps
        
        custom_metrics = {}
        total_reward = 0.0
        
        for key, value in self.reward_accumulate.items():
            # 개별 보상 항목의 평균값 계산
            avg_reward = value / total_steps_in_rollout
            custom_metrics[f"rewards_breakdown/{key}"] = avg_reward
            # 총 보상에 합산
            total_reward += avg_reward
        custom_metrics["rewards/total_reward"] = total_reward
        wandb.log(custom_metrics)
        ##################################################################################

        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)




class MyoAssistLegImitation(MyoAssistLegBase):
    
    ################################################################################################################### Rendering
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._viewer = None
        self._viewer_alive = False
        self._joint_adr_cache = {} 
        
    # --- NEW: 카메라 상태 보존/복원 ---
    def get_camera_state(self):
        if self._viewer is None:
            return None
        try:
            cam = self._viewer.cam
            return dict(
                azimuth=float(cam.azimuth),
                elevation=float(cam.elevation),
                distance=float(cam.distance),
                lookat=np.array(cam.lookat, dtype=float).copy(),
            )
        except Exception:
            return None

    def set_camera_state(self, state:dict|None):
        if self._viewer is None or state is None:
            return
        try:
            cam = self._viewer.cam
            cam.azimuth   = state.get("azimuth",   cam.azimuth)
            cam.elevation = state.get("elevation", cam.elevation)
            cam.distance  = state.get("distance",  cam.distance)
            la = state.get("lookat", None)
            if la is not None:
                cam.lookat[:] = la
        except Exception:
            pass

    def render_live(self, paused: bool = False, cam_state: dict | None = None):
        M = self.sim.model.ptr if hasattr(self.sim.model, "ptr") else self.sim.model
        D = self.sim.data.ptr  if hasattr(self.sim.data,  "ptr") else self.sim.data

        if (self._viewer is None) or (not getattr(self._viewer, "is_alive", True)):
            self._viewer = mjv.launch_passive(M, D)
            self._viewer_alive = True
            if cam_state is not None:
                self.set_camera_state(cam_state)

        if paused:
            return  # 프레임 정지: sync()도 호출 안 함 → CPU 부담 거의 0

        # --- 여기서부터 ref 고스트 + 에러 오버레이 ---
        try:
            # 1) ref qpos로 별도의 MjData를 만들어 forward
            if not hasattr(self, "_ref_data_cache"):
                self._ref_data_cache = mujoco.MjData(self.sim.model.ptr)
            
            # [수정] 현재 인덱스의 Reference Pose를 정확히 가져옴
            ref_qpos = self._get_ref_qpos_now()
            
            if ref_qpos is not None:
                self._ref_data_cache.qpos[:] = ref_qpos
                mujoco.mj_forward(self.sim.model.ptr, self._ref_data_cache)

                # 2) ref 마커(고스트) 표시
                ghost_sites = ["r_knee_site","r_ankle_site","l_knee_site","l_ankle_site","pelvis_site"]
                self._add_ref_markers(self._ref_data_cache, ghost_sites, rgba=(0.0,1.0,0.0,0.6), size=0.02)

            # 3) 텍스트 오버레이
            key_joints = [
                "pelvis_tilt", "pelvis_list", "pelvis_rotation",
                "pelvis_tx", "pelvis_ty", "pelvis_tz",
                "hip_flexion_r","hip_adduction_r","hip_rotation_r",
                "knee_angle_r","ankle_angle_r",
                "hip_flexion_l","hip_adduction_l","hip_rotation_l",
                "knee_angle_l","ankle_angle_l"
            ]
            pairs = self._joint_err_with_units(key_joints)
            if pairs and hasattr(self._viewer, "add_overlay"):
                self._viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "tracking", "joint          actual    ref     |err|   unit")
                for jn, act_v, ref_v, err_v, unit in pairs:
                    self._viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT, "",
                        f"{jn:14s}{act_v:9.2f}{ref_v:9.2f}{err_v:9.2f}  {unit}"
                    )
        except Exception as e:
            # print(f"Render Error: {e}") 
            pass

        # 마지막에 동기화
        try:
            # self._viewer.sync_data(self.sim.model.ptr, self.sim.data.ptr)
            self._viewer.sync()
        except Exception:
            try: self._viewer.close()
            except Exception: pass
            self._viewer = None
            self._viewer_alive = False

    def close_live_view(self):
        if self._viewer is not None:
            try: self._viewer.close()
            except Exception: pass
            self._viewer = None
            self._viewer_alive = False

    def _get_ref_qpos_now(self):
        """
        [수정됨] Reference Data와 현재 imitation index를 사용하여 
        실제 Reference Pose Vector를 구성하여 반환합니다.
        """
        if (self._reference_data is None) or (self._imitation_index is None):
            return None
        
        # 1. 기본 포즈(키 프레임)로 초기화
        ref_vec = self.sim.model.key_qpos[0].copy()
        
        # 2. 캐시된 주소를 이용해 Reference 값 덮어쓰기
        idx = self._imitation_index
        series = self._reference_data["series_data"]
        
        for key, adr in self._joint_adr_cache.items():
            try:
                # Reference 데이터에서 값 가져오기
                val = series[f"q_{key}"][idx]
                ref_vec[adr] = val
            except KeyError:
                pass
                
        return ref_vec

    def _joint_err_with_units(self, joints: list[str]):
        """각 관절의 (이름, actual, ref, err, unit)을 반환."""
        m = self.sim.model; d = self.sim.data
        ref_qpos = self._get_ref_qpos_now()
        if ref_qpos is None:
            return []

        out = []
        for jn in joints:
            # Cache 이용
            if jn not in self._joint_adr_cache:
                jid = mujoco.mj_name2id(m.ptr, mujoco.mjtObj.mjOBJ_JOINT, jn.encode())
                if jid >= 0:
                     self._joint_adr_cache[jn] = m.jnt_qposadr[jid]
                else:
                    continue # 없는 관절
            
            adr = self._joint_adr_cache[jn]
            # jid를 다시 구해야 type을 알 수 있음 (최적화를 위해 type도 캐싱하면 좋으나 생략)
            jid = mujoco.mj_name2id(m.ptr, mujoco.mjtObj.mjOBJ_JOINT, jn.encode())
            jtype = m.jnt_type[jid] 

            act = float(d.qpos[adr])
            ref = float(ref_qpos[adr])

            if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                act_v = act * 180.0/np.pi; ref_v = ref * 180.0/np.pi; unit = "deg"
            elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
                act_v = act * 1000.0; ref_v = ref * 1000.0; unit = "mm"
            else:
                act_v = act; ref_v = ref; unit = ""

            out.append((jn, act_v, ref_v, abs(act_v - ref_v), unit))
        return out

    def _add_ref_markers(self, data_ref, site_names: list[str], rgba=(0.0,1.0,0.0,0.6), size=0.012):
        if self._viewer is None:
            return
        m = self.sim.model
        # site id 찾기
        sids = []
        for sn in site_names:
            try:
                sid = mujoco.mj_name2id(m.ptr, mujoco.mjtObj.mjOBJ_SITE, sn.encode())
            except Exception:
                sid = -1
            sids.append(sid)
        # 마커 찍기
        for sid in sids:
            if sid < 0: 
                continue
            pos = data_ref.site_xpos[sid]
            self._viewer.add_marker(pos=pos, size=[size, size, size], rgba=rgba)

    ################################################################################################################### Rendering
    
    def _setup(self,*,
            env_params:ImitationTrainSessionConfig.EnvParams,
            reference_data:dict|None = None,
            loop_reference_data:bool = False,
            **kwargs,
        ):
        self._flag_random_ref_index = env_params.flag_random_ref_index
        self._out_of_trajectory_threshold = env_params.out_of_trajectory_threshold
        self.reference_data_keys = env_params.reference_data_keys
        self._loop_reference_data = loop_reference_data
        self._reward_keys_and_weights:ImitationTrainSessionConfig.EnvParams.RewardWeights = env_params.reward_keys_and_weights

        self.setup_reference_data(data=reference_data)

        super()._setup(env_params=env_params,
                       **kwargs,
                       )

        
        
    def set_reward_weights(self, reward_keys_and_weights:TrainSessionConfigBase.EnvParams.RewardWeights):
        self._reward_keys_and_weights = reward_keys_and_weights
    # override from MujocoEnv
    def get_obs_dict(self, sim):
        return super().get_obs_dict(sim)

    def _get_qpos_diff(self) -> dict:

        def get_qpos_diff_one(key:str):
            diff = self.sim.data.joint(f"{key}").qpos[0].copy() - self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
            return diff
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qpos_imitation_rewards:
            name_diff_dict[q_key] = get_qpos_diff_one(q_key)
        return name_diff_dict
    def _get_qvel_diff(self):
        speed_ratio_to_target_velocity = self._target_velocity / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]

        def get_qvel_diff_one(key:str):
            diff = self.sim.data.joint(f"{key}").qvel[0].copy() - self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio_to_target_velocity
            return diff
        name_diff_dict = {}
        for q_key in self._reward_keys_and_weights.qvel_imitation_rewards:
            # joint_weight = self._reward_keys_and_weights.qvel_imitation_rewards[q_key]
            name_diff_dict[q_key] = get_qvel_diff_one(q_key)
        return name_diff_dict
    def _get_qpos_diff_nparray(self):
        return np.array([diff for diff in self._get_qpos_diff().values()])
    def _get_end_effector_diff(self):
        # body_pos = self.sim.data.body('pelvis').xpos.copy()
        # diff_array = []
        # for mapping in self.ANCHOR_SIM_TO_REF.values():
        #     sim_anchor = self.sim.data.joint(mapping.sim_name).xanchor.copy() - body_pos
        #     ref_anchor = self._reference_data[mapping.ref_name][self._imitation_index]
        #     diff = np.linalg.norm(sim_anchor - ref_anchor)
        #     diff_array.append(diff)
        # return diff_array
        return np.array([0])
    
    def _calculate_imitation_rewards(self, obs_dict):
        base_reward, base_info = super()._calculate_base_reward(obs_dict)

        q_diff_dict = self._get_qpos_diff()
        dq_diff_dict = self._get_qvel_diff()
        anchor_diff_array = self._get_end_effector_diff()

        # Calculate joint position rewards
        q_reward_dict = {}
        for joint_name, diff in q_diff_dict.items():
            q_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))

        dq_reward_dict = {}
        for joint_name, diff in dq_diff_dict.items():
            dq_reward_dict[joint_name] = self.dt * np.exp(-8 * np.square(diff))
        
        # Calculate end effector reward
        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array)))

        # Calculate joint imitation rewards sum
        qpos_imitation_rewards = np.sum([q_reward_dict[key] * self._reward_keys_and_weights.qpos_imitation_rewards[key] for key in q_reward_dict.keys()])
        qvel_imitation_rewards = np.sum([dq_reward_dict[key] * self._reward_keys_and_weights.qvel_imitation_rewards[key] for key in dq_reward_dict.keys()])

        # Add new key-value pairs to the base_reward dictionary
        base_reward.update({
            'qpos_imitation_rewards': qpos_imitation_rewards,
            'qvel_imitation_rewards': qvel_imitation_rewards,
            'end_effector_imitation_reward': anchor_reward
        })

        # Use the updated base_reward as imitation_rewards
        imitation_rewards = base_reward
        info = base_info
        return imitation_rewards, info
    

    # override from MujocoEnv
    def get_reward_dict(self, obs_dict):
        # Calculate common rewards
        imitation_rewards, info = self._calculate_imitation_rewards(obs_dict)

        # Construct reward dictionary
        # Automatically add all imitation_rewards items to rwd_dict
        rwd_dict = collections.OrderedDict((key, imitation_rewards[key]) for key in imitation_rewards)

        # Add additional fixed keys
        rwd_dict.update({
            'sparse': 0,
            'solved': False,
            'done': self._get_done(),
        })
        # Calculate final reward
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items() if key in rwd_dict], axis=0)
        
        return rwd_dict
    

    def _follow_reference_motion(self, is_x_follow:bool):
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qpos = self._reference_data["series_data"][f"q_{key}"][self._imitation_index]
            if not is_x_follow and key == 'pelvis_tx':
                self.sim.data.joint(f"{key}").qpos = 0
            # if key == 'pelvis_ty':
            #     self.sim.data.joint(f"{key}").qpos += 0.05
        speed_ratio_to_target_velocity = self._target_velocity / self._reference_data["series_data"]["dq_pelvis_tx"][self._imitation_index]
        for key in self.reference_data_keys:
            self.sim.data.joint(f"{key}").qvel = self._reference_data["series_data"][f"dq_{key}"][self._imitation_index] * speed_ratio_to_target_velocity
    def imitation_step(self, is_x_follow:bool, specific_index:int|None = None):
        if specific_index is None:
            self._imitation_index += 1
            if self._imitation_index >= self._reference_data_length:
                self._imitation_index = 0
        else:
            self._imitation_index = specific_index
        self._follow_reference_motion(is_x_follow)
        # should call this but I don't know why
        # next_obs, reward, terminated, truncated, info = super().step(np.zeros(self.sim.model.nu))
        # return (next_obs, reward, False, False, info)
        self.forward()
        return self._imitation_index
        # pass
    
    # override
    def step(self, a, **kwargs):
        if self._imitation_index is not None:
            self._imitation_index += 1
            if self._imitation_index < self._reference_data_length:
                is_out_of_index = False
            else:
                if self._loop_reference_data:
                    self._imitation_index = 0
                    is_out_of_index = False
                else:
                    is_out_of_index = True
                    self._imitation_index = self._reference_data_length - 1
        else:
            is_out_of_index = True
        
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        if is_out_of_index:
            reward = 0
            truncated = True
        else:
            q_diff_nparray:np.ndarray = self._get_qpos_diff_nparray()
            is_out_of_trajectory = np.any(np.abs(q_diff_nparray) >self._out_of_trajectory_threshold)
            terminated = terminated or is_out_of_trajectory
        
        return (next_obs, reward, terminated, truncated, info)
        
    
    def setup_reference_data(self, data:dict|None):
        self._reference_data = data
        self._imitation_index = None
        if data is not None:
            # self._follow_reference_motion(False)
            self._reference_data_length = self._reference_data["metadata"]["resampled_data_length"]
        else:
            raise ValueError("Reference data is not set")

    def reset(self, **kwargs):
        rng = np.random.default_rng()# TODO: refactoring random to use seed
        
        if self._flag_random_ref_index:
            self._imitation_index = rng.integers(0, int(self._reference_data_length * 0.8))
        else:
            self._imitation_index = 0
        # generate random targets
        # new_qpos = self.generate_qpos()# TODO: should set qvel too.
        # self.sim.data.qpos = new_qpos
        self._follow_reference_motion(False)
        
        obs = super().reset(reset_qpos= self.sim.data.qpos, reset_qvel=self.sim.data.qvel, **kwargs)
        return obs

    # override
    def _initialize_pose(self):
        super()._initialize_pose()