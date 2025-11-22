import sys
import os
# Add current workspace to path to use local modifications
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rl_train.train.train_configs.config as myoassist_config
import rl_train.utils.train_log_handler as train_log_handler
from rl_train.utils.data_types import DictionableDataclass
import json
from datetime import datetime
from rl_train.envs.environment_handler import EnvironmentHandler
import subprocess

################################################################################################################### Rendering
import os, sys, time, threading
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, configure
import numbers

def _getch_nonblock():
    if os.name == "nt":  # Windows
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            return ch
        return None
    else:  # POSIX
        import select, sys, termios, tty
        dr, _, _ = select.select([sys.stdin], [], [], 0.03)
        if dr:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            return ch
        return None
    
class LiveRenderToggleCallback(BaseCallback):
    """
    키:
      - 'o' : 창 ON/OFF (viewer open/close)
      - 'v' : 일시정지/재개 (sync 끔/켬)
      - 'm' : 다음 env
      - 'n' : 이전 env
      - 'q' : 키 리스너 종료
    주의: 
      1) 키 스레드에서는 '명령만 큐에 push'하고, 실제 env_method 호출은
         _on_step() (메인 학습 스레드)에서만 수행 -> 경쟁/랜덤 크래시 방지
      2) SubprocVecEnv (multiprocessing) 사용 시 MuJoCo viewer 생성 불가!
         -> DummyVecEnv (threading) 사용 필요 또는 render_off=True로 비활성화
    """
    def __init__(self, num_envs:int, start_index:int=0, render_every_n_steps:int=1, render_off:bool=False, verbose:int=1):
        super().__init__(verbose)
        self.num_envs = int(num_envs)
        self.curr_idx = int(start_index)
        self.render_off = render_off  # True면 렌더링 명령 무시

        self.enabled_window = False
        self.paused = False

        self.render_interval = max(1, int(render_every_n_steps))
        self._last_render_step = -1

        self._stop = False
        self._th = threading.Thread(target=self._key_loop, daemon=True)

        self._cmd_q = deque()
        self._lock = threading.Lock()

    # ---------- 키 스레드: 명령만 큐에 넣는다 ----------
    def _key_loop(self):
        def _getch_nonblock():
            if os.name == "nt":
                import msvcrt
                if msvcrt.kbhit(): return msvcrt.getwch()
                return None
            else:
                import select, termios, tty
                dr, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not dr: return None
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                return ch

        while not self._stop:
            ch = _getch_nonblock()
            if not ch:
                time.sleep(0.01)
                continue
            c = ch.lower()
            with self._lock:
                if c == 'o':
                    self._cmd_q.append(("TOGGLE_WINDOW", None))
                elif c == 'v':
                    self._cmd_q.append(("TOGGLE_PAUSE", None))
                elif c == 'm':
                    self._cmd_q.append(("SWITCH_ENV", +1))
                elif c == 'n':
                    self._cmd_q.append(("SWITCH_ENV", -1))
                elif c == 'q':
                    self._cmd_q.append(("STOP_KEYS", None))
            time.sleep(0.005)

    def _drain_cmds(self):
        cmds = []
        with self._lock:
            while self._cmd_q:
                cmds.append(self._cmd_q.popleft())
        return cmds

    # ---------- SB3 콜백 라이프사이클 ----------
    def _on_training_start(self) -> None:
        # SubprocVecEnv 체크: multiprocessing이면 렌더링 불가
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        if isinstance(self.training_env, SubprocVecEnv):
            if self.verbose:
                print("[LiveRender] WARNING: SubprocVecEnv detected - live rendering disabled")
                print("[LiveRender] TIP: Use --flag_live_render to enable DummyVecEnv for live rendering")
            self.render_off = True
        elif isinstance(self.training_env, DummyVecEnv):
            if self.verbose:
                print("[LiveRender] DummyVecEnv detected - live rendering ENABLED")
        
        if not self.render_off:
            self._th.start()
            if self.verbose:
                print("[LiveRender] Keyboard controls active: o(open/close), v(pause), m/n(next/prev env), q(stop)")
        else:
            if self.verbose:
                print("[LiveRender] Keyboard controls disabled (SubprocVecEnv mode)")

    def _on_training_end(self) -> None:
        self._stop = True
        if self.enabled_window:
            try:
                self.training_env.env_method("close_live_view", indices=[self.curr_idx])
            except Exception:
                pass

    # ---------- 메인 학습 스레드에서만 env_method를 호출 ----------
    def _on_step(self) -> bool:
        if self.render_off:
            return True  # 렌더링 비활성화 시 아무것도 안 함
        
        # 1) 키 명령 처리
        for typ, arg in self._drain_cmds():
            try:
                if typ == "TOGGLE_WINDOW":
                    if self.enabled_window:
                        # close
                        try:
                            self.training_env.env_method("close_live_view", indices=[self.curr_idx])
                        except Exception:
                            pass
                        self.enabled_window = False
                        if self.verbose: print("[LiveRender] window OFF")
                    else:
                        # open
                        self.enabled_window = True
                        if self.verbose: print("[LiveRender] window ON (env %d)" % self.curr_idx)
                        try:
                            self.training_env.env_method("render_live", paused=self.paused, indices=[self.curr_idx])
                        except Exception as e:
                            if self.verbose: print(f"[LiveRender] open failed: {e}")
                            self.enabled_window = False

                elif typ == "TOGGLE_PAUSE":
                    self.paused = not self.paused
                    if self.verbose: print(f"[LiveRender] paused={'ON' if self.paused else 'OFF'}")

                elif typ == "SWITCH_ENV":
                    delta = int(arg)
                    prev = self.curr_idx
                    self.curr_idx = (self.curr_idx + delta) % self.num_envs
                    if self.verbose: print(f"[LiveRender] env {prev} -> {self.curr_idx}")
                    if self.enabled_window:
                        # close prev -> short sleep -> open new
                        try:
                            self.training_env.env_method("close_live_view", indices=[prev])
                        except Exception:
                            pass
                        time.sleep(0.01)
                        try:
                            self.training_env.env_method("render_live", paused=self.paused, indices=[self.curr_idx])
                        except Exception as e:
                            if self.verbose: print(f"[LiveRender] switch open failed: {e}")
                            # 창 상태를 정합하게 유지
                            self.enabled_window = False

                elif typ == "STOP_KEYS":
                    if self.verbose: print("[LiveRender] key listener stopped")
                    self._stop = True

            except Exception as e:
                # 어떤 에러도 학습을 죽이지 않도록 삼킨다
                if self.verbose: print(f"[LiveRender] cmd {typ} error: {e}")

        # 2) 주기적 프레임 업데이트 (창 ON & pause OFF 일 때만)
        if self.enabled_window and (not self.paused):
            if (self.num_timesteps - self._last_render_step) >= self.render_interval:
                try:
                    self.training_env.env_method("render_live", paused=False, indices=[self.curr_idx])
                    self._last_render_step = self.num_timesteps
                except Exception as e:
                    if self.verbose: print(f"[LiveRender] render tick failed: {e}")
        return True
################################################################################################################### Rendering

def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        return {
            "commit": commit,
            "branch": branch
        }
    except:
        return {
            "commit": "unknown",
            "branch": "unknown"
        }

# Version information
VERSION = {
    "version": "0.3.0",  # MAJOR.MINOR.PATCH
    **get_git_info()
}
def ppo_evaluate_with_rendering(config):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on=True, is_evaluate_mode=True)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    obs, info = env.reset()
    for _ in range(config.evaluate_param_list[0]["num_timesteps"]):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        if truncated:
            obs, info = env.reset()

    env.close()
def ppo_train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler, use_live_render=False):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on, use_dummy_vec_env=use_live_render)
    
    # Set tensorboard log directory
    tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
    
    model = EnvironmentHandler.get_stable_baselines3_model(config, env, tensorboard_log=tensorboard_log_dir)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)

    session_config_dict["code_version"] = VERSION
    with open(os.path.join(log_dir, 'session_config.json'), 'w', encoding='utf-8') as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)

    # ========== WANDB INITIALIZATION (FRIEND'S STYLE) ==========
    # Initialize WandB if enabled in config
    wandb_run = None
    if hasattr(config, 'wandb_params') and config.wandb_params.enabled:
        import wandb
        print(f"[WandB] Initializing from run_train.py...")
        print(f"[WandB]   Project: {config.wandb_params.project}")
        print(f"[WandB]   Run Name: {config.wandb_params.run_name}")
        
        wandb_run = wandb.init(
            project=config.wandb_params.project,
            name=config.wandb_params.run_name,
            config=session_config_dict,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        print(f"[WandB] Initialization complete! Run URL: {wandb.run.url}")
    else:
        print(f"[WandB] Disabled or not configured in config file")
    # ========== END WANDB INITIALIZATION ==========

    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler)

    # ========== ADD LIVE RENDERING TOGGLE CALLBACK ==========
    # Create list of callbacks
    from stable_baselines3.common.callbacks import CallbackList
    
    callbacks = [custom_callback]
    
    # Add WandB callback if wandb is initialized
    if wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback
        print(f"[WandB] Adding WandbCallback...")
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{wandb_run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
        print(f"[WandB] WandbCallback added")
    
    # Add live render toggle if not in rendering mode
    # NOTE: Auto-disabled if SubprocVecEnv (multiprocessing) is used
    #       Requires DummyVecEnv (threading) for MuJoCo viewer support
    if not is_rendering_on:
        live_render_callback = LiveRenderToggleCallback(
            num_envs=config.env_params.num_envs,
            start_index=0,
            render_every_n_steps=4,
            render_off=False,  # Will auto-detect SubprocVecEnv and disable
            verbose=1
        )
        callbacks.append(live_render_callback)
    
    callback_list = CallbackList(callbacks)
    # ========== END LIVE RENDERING ==========

    # Start training with tensorboard logging
    model.learn(
        reset_num_timesteps=False, 
        total_timesteps=train_time_step, 
        log_interval=1, 
        callback=callback_list, 
        progress_bar=True,
        tb_log_name="PPO"
    )
    env.close()
    print("learning done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    parser.add_argument("--flag_rendering", type=bool, default=False, action=argparse.BooleanOptionalAction, help="rendering(True/False)")
    parser.add_argument("--flag_realtime_evaluate", type=bool, default=False, action=argparse.BooleanOptionalAction, help="realtime evaluate(True/False)")
    parser.add_argument("--flag_live_render", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Enable live rendering toggle with DummyVecEnv (slower but allows 'o' key rendering)")

    args, unknown_args = parser.parse_known_args()
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, myoassist_config.TrainSessionConfigBase)
    DictionableDataclass.add_arguments(default_config, parser, prefix="config.")
    args = parser.parse_args()

    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)


    DictionableDataclass.set_from_args(config, args, prefix="config.")


    log_dir = os.path.join("rl_train","results", f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    train_log_handler = train_log_handler.TrainLogHandler(log_dir)

    if args.flag_realtime_evaluate:
        ppo_evaluate_with_rendering(config)
    else:
        ppo_train_with_parameters(config,
                                train_time_step=config.total_timesteps,
                                is_rendering_on=args.flag_rendering,
                                train_log_handler=train_log_handler,
                                use_live_render=args.flag_live_render)
    