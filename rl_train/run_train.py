import numpy as np
import rl_train.train.train_configs.config as myoassist_config
import rl_train.utils.train_log_handler as train_log_handler
from rl_train.utils.data_types import DictionableDataclass
import json
import os
from datetime import datetime
from rl_train.envs.environment_handler import EnvironmentHandler
import subprocess
import torch
import wandb # [추가]
from wandb.integration.sb3 import WandbCallback # [추가]
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
    주의: 키 스레드에서는 '명령만 큐에 push'하고, 실제 env_method 호출은
         _on_step() (메인 학습 스레드)에서만 수행 -> 경쟁/랜덤 크래시 방지
    """
    def __init__(self, num_envs:int, start_index:int=0, render_every_n_steps:int=1, verbose:int=1):
        super().__init__(verbose)
        self.num_envs = int(num_envs)
        self.curr_idx = int(start_index)

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
        self._th.start()
        if self.verbose:
            print("[LiveRender] keys: o(open/close), v(pause), m/n(next/prev env), q(stop)")

    def _on_training_end(self) -> None:
        self._stop = True
        if self.enabled_window:
            try:
                self.training_env.env_method("close_live_view", indices=[self.curr_idx])
            except Exception:
                pass

    # ---------- 메인 학습 스레드에서만 env_method를 호출 ----------
    def _on_step(self) -> bool:
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
###################################################################################################################

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
    
def ppo_train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env, tensorboard_log=train_log_handler.log_dir)

    try:
        print(f"[TORCH] cuda.is_available = {torch.cuda.is_available()}")
    except Exception as e:
        print(f"[TORCH] not importable ({e})")
    print(f"[PPO] model.device   = {getattr(model, 'device', None)}")
    print(f"[PPO] policy.device  = {getattr(getattr(model, 'policy', None), 'device', None)}")
        
        
    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)
    session_config_dict["code_version"] = VERSION
    
    run = wandb.init(
        project="MyoAssist-Imitation",
        config=session_config_dict,  # 전체 config 자동 로깅
        sync_tensorboard=True,       # SB3 기본 로그와 연동
        monitor_gym=True,            # 비디오 저장 (필요시)
        save_code=True,              # 코드 백업
        name=f"session_{datetime.now().strftime('%Y%m%d-%H%M%S')}" # Run 이름
    )
    
    with open(os.path.join(log_dir, 'session_config.json'), 'w', encoding='utf-8') as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)


    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler)
    
########################################################################################################## rendering, wandb
    live_cb = LiveRenderToggleCallback(num_envs=config.env_params.num_envs, start_index=0)
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    final_callback = [custom_callback, live_cb, wandb_callback]
##########################################################################################################

    model.learn(reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=final_callback, progress_bar=True)
    env.close()
    print("learning done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    parser.add_argument("--flag_rendering", type=bool, default=False, action=argparse.BooleanOptionalAction, help="rendering(True/False). Default: False to avoid opening GUI on train start.")
    parser.add_argument("--flag_realtime_evaluate", type=bool, default=False, action=argparse.BooleanOptionalAction, help="realtime evaluate(True/False)")

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
                                train_log_handler=train_log_handler)
    