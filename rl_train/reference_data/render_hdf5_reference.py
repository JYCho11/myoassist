#!/usr/bin/env python3
"""
Visualize HDF5-converted reference motion
"""
import numpy as np
import argparse
import mujoco
import imageio
from pathlib import Path


def render_reference_motion(npz_path, model_path, output_path,
                            num_frames=800, height_offset=0.95,
                            view_type='diagonal'):
    """Render reference motion from NPZ file

    Args:
        height_offset: Vertical offset to lift model above ground (meters)
                      Default 0.95m is approximately pelvis height for standing
        view_type: Camera view type. Can be 'diagonal', 'side', or 'front'.
    """

    print(f'Loading reference: {npz_path}')
    # allow_pickle=True → metadata/series_data dict 구조도 읽을 수 있게
    data = np.load(npz_path, allow_pickle=True)

    # ------------------------------------------------------------------
    # 1) reference joint 순서 정의 (MuJoCo joint 이름 매핑용)
    # ------------------------------------------------------------------
    ref_joint_order = [
        ('q_pelvis_tx', 'pelvis_tx'),
        ('q_pelvis_ty', 'pelvis_ty'),
        ('q_pelvis_tz', 'pelvis_tz'),
        ('q_pelvis_tilt', 'pelvis_tilt'),      # q_ref[3] → qpos[3]
        ('q_pelvis_list', 'pelvis_list'),      # q_ref[4] → qpos[4]
        ('q_pelvis_rotation', 'pelvis_rotation'),
        ('q_hip_flexion_r', 'hip_flexion_r'),
        ('q_hip_adduction_r', 'hip_adduction_r'),
        ('q_hip_rotation_r', 'hip_rotation_r'),
        ('q_knee_angle_r', 'knee_angle_r'),
        ('q_ankle_angle_r', 'ankle_angle_r'),
        ('q_hip_flexion_l', 'hip_flexion_l'),
        ('q_hip_adduction_l', 'hip_adduction_l'),
        ('q_hip_rotation_l', 'hip_rotation_l'),
        ('q_knee_angle_l', 'knee_angle_l'),
        ('q_ankle_angle_l', 'ankle_angle_l')
    ]

    # ------------------------------------------------------------------
    # 2) 두 가지 포맷 지원:
    #    (A) q_ref + joint_names 직접 있는 경우 (기존 포맷)
    #    (B) metadata/series_data dict 구조 (학습용 npz)
    # ------------------------------------------------------------------
    if 'q_ref' in data.files and 'joint_names' in data.files:
        # --- 기존 포맷 ---
        q_ref = data['q_ref']
        joint_names = data['joint_names']
        print("  Detected format: legacy (q_ref + joint_names)")
    elif 'series_data' in data.files:
        # --- 새 포맷: series_data dict → q_ref 구성 ---
        print("  Detected format: series_data dict")

        series_data = data['series_data'].item()
        # 길이 T 추정: q_pelvis_tx 또는 다른 q_* 중 하나에서 길이 사용
        # (없을 가능성은 거의 없다고 가정하지만, 방어적으로 작성)
        candidate_keys = [k for k in series_data.keys() if k.startswith('q_')]
        if len(candidate_keys) == 0:
            raise ValueError("series_data 안에 'q_'로 시작하는 키가 없습니다. npz 구조를 확인하세요.")

        T = len(np.asarray(series_data[candidate_keys[0]]).squeeze())
        q_ref = np.zeros((T, len(ref_joint_order)), dtype=float)
        joint_names_list = []

        print("  Building q_ref from series_data keys...")
        for col_idx, (ref_key, _) in enumerate(ref_joint_order):
            if ref_key in series_data:
                arr = np.asarray(series_data[ref_key]).squeeze()
                if arr.ndim != 1:
                    # (T,1) 같은 경우를 위해 reshape
                    arr = arr.reshape(-1)
                if len(arr) != T:
                    raise ValueError(
                        f"series_data['{ref_key}'] 길이 {len(arr)}가 다른 q_* 길이 {T}와 다릅니다."
                    )
                q_ref[:, col_idx] = arr
                print(f"    filled column {col_idx:2d} from series_data['{ref_key}'], "
                      f"range=({arr.min():+.3f}, {arr.max():+.3f})")
            else:
                # 해당 키가 없으면 0으로 두되 경고만 출력
                print(f"    [경고] series_data에 '{ref_key}'가 없습니다. "
                      f"q_ref[:, {col_idx}]는 0으로 유지됩니다.")
            joint_names_list.append(ref_key)

        joint_names = np.array(joint_names_list)
    else:
        raise ValueError(
            "지원하지 않는 NPZ 포맷입니다. "
            "① q_ref + joint_names 또는 ② series_data dict 구조여야 합니다."
        )

    print(f'  Frames: {q_ref.shape[0]}')
    print(f'  DOF: {q_ref.shape[1]}')
    print(f'  Joints: {list(joint_names)}')
    print(f'  Height offset: {height_offset:.3f} m')

    # ------------------------------------------------------------------
    # 3) MuJoCo 모델 로드 및 joint → qpos 인덱스 매핑
    # ------------------------------------------------------------------
    print(f'Loading model: {model_path}')
    model = mujoco.MjModel.from_xml_path(model_path)
    data_mj = mujoco.MjData(model)

    # Create joint name to qpos index mapping
    joint_to_qpos = {}
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        print(f'  Joint {i:2d}: {jnt_name}')
        qpos_addr = model.jnt_qposadr[i] # qpos index
        joint_to_qpos[jnt_name] = qpos_addr

    print(f'\n{"="*80}')
    print(f'QPOS MAPPING EXPLANATION')
    print(f'{"="*80}')
    print(f'Our reference data (q_ref) has {q_ref.shape[1]} DOF in a specific order.')
    print(f'MuJoCo model has {model.nq} qpos values (including auto-calculated wrapping points).')
    print(f'We need to map each q_ref column to the correct qpos index.\n')

    print(f'[Joint → qpos mapping]:')
    print(f'{"q_ref index":<15} {"Joint name":<25} {"→":<3} {"qpos index":<15} {"MuJoCo joint"}')
    print(f'{"-"*80}')

    # Build mapping: ref_data column → qpos index (ref_joint_order 기준)
    ref_to_qpos = []
    for ref_idx, (ref_name, mujoco_name) in enumerate(ref_joint_order):
        if ref_idx < q_ref.shape[1] and mujoco_name and mujoco_name in joint_to_qpos:
            qpos_idx = joint_to_qpos[mujoco_name]
            ref_to_qpos.append((ref_idx, qpos_idx, mujoco_name))
            print(f'  q_ref[{ref_idx:2d}]      {ref_name:<25} →   qpos[{qpos_idx:2d}]        {mujoco_name}')

    print(f'{"-"*80}')
    print(f'Note: qpos indices are NOT sequential because muscle wrapping points')
    print(f'      (e.g., knee_r_translation1/2) are interspersed between main joints.')
    print(f'{"="*80}\n')

    # ------------------------------------------------------------------
    # 4) 렌더러 / 카메라 설정
    # ------------------------------------------------------------------
    renderer = mujoco.Renderer(model, height=720, width=1280)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)

    if view_type == 'side':
        print("\nSetting camera to: Side View")
        camera.azimuth = 90
        camera.elevation = -10
        camera.distance = 4.0
        camera.lookat[:] = [0, 0.8, 0]
    elif view_type == 'front':
        print("\nSetting camera to: Front View")
        camera.azimuth = 180
        camera.elevation = -10
        camera.distance = 3.5
        camera.lookat[:] = [0, 0.0, 0]
    elif view_type == 'top':
        print("\nSetting camera to: top View")
        camera.azimuth = 180
        camera.elevation = -90
        camera.distance = 3.5
        camera.lookat[:] = [0, 0.0, 0]        
    else:
        print("\nSetting camera to: Diagonal View (default)")
        camera.azimuth = 135
        camera.elevation = -20
        camera.distance = 5.0
        camera.lookat[:] = [0, 0.5, 0]

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

    # 팔 geom 투명 처리
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and any(part in geom_name.lower()
                             for part in ['humer', 'ulna', 'radius', 'hand', 'arm']):
            model.geom_rgba[i, 3] = 0.0  # alpha=0

    print(f'\nCamera settings:')
    print(f'  View angle: {view_type} (azimuth={camera.azimuth}°, elevation={camera.elevation}°)')
    print(f'  Distance: {camera.distance}m')
    print(f'  Transparency: Enabled (can see through floor)')

    # ------------------------------------------------------------------
    # 5) 프레임 루프 돌면서 렌더링
    # ------------------------------------------------------------------
    print(f'\nRendering {num_frames} frames...')
    frames = []

    frame_skip = max(1, q_ref.shape[0] // num_frames)

    for i in range(0, min(num_frames * frame_skip, q_ref.shape[0]), frame_skip):
        # stand keyframe을 base pose로 사용
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        data_mj.qpos[:] = model.key_qpos[key_id]

        # qpos mapping 적용
        for ref_idx, qpos_idx, jnt_name in ref_to_qpos:
            if ref_idx < q_ref.shape[1] and qpos_idx < model.nq:
                data_mj.qpos[qpos_idx] = q_ref[i, ref_idx]

        # pelvis_ty (index 1)에 height offset 적용
        if 1 < len(data_mj.qpos):
            # ref_joint_order에서 q_pelvis_ty가 1번째라 가정 (위에서 순서 보장)
            data_mj.qpos[1] = q_ref[i, 1]

        # 팔 neutral pose
        arm_joints = {
            40: 0.0,   # r_shoulder_abd
            41: 0.0,   # r_shoulder_rot
            42: 0.5,   # r_shoulder_flex
            43: 0.8,   # r_elbow_flex
            47: 0.0,   # l_shoulder_abd
            48: 0.0,   # l_shoulder_rot
            49: 0.5,   # l_shoulder_flex
            50: 0.8,   # l_elbow_flex
        }
        for qpos_idx, angle in arm_joints.items():
            if qpos_idx < len(data_mj.qpos):
                data_mj.qpos[qpos_idx] = angle

        mujoco.mj_forward(model, data_mj)

        renderer.update_scene(data_mj, camera=camera, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

        if (i // frame_skip) % 30 == 0:
            print(f'  Frame {i // frame_skip}/{num_frames}...')

    # ------------------------------------------------------------------
    # 6) 비디오 저장
    # ------------------------------------------------------------------
    print(f'Saving video: {output_path}')
    effective_fps = num_frames / 60.0  # ~60초 영상
    print(f'  Video FPS: {effective_fps:.1f} (target duration: ~60 seconds)')
    imageio.mimsave(output_path, frames, fps=effective_fps)

    # joint range 출력
    print('\nJoint ranges:')
    for i, name in enumerate(joint_names):
        vals = q_ref[:, i]
        print(f'  {name:20s}: [{vals.min():+.3f}, {vals.max():+.3f}] rad')

    print(f'\n✅ Done! Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='S004_level_08mps',
                        help='NPZ file name or path')

    # project root 추론
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # myotuning/
    default_model_path = project_root / 'models' / '26muscle_3D' / 'myoLeg26_BASELINE.xml'
    default_data_dir = project_root / 'rl_train' / 'reference_data'

    parser.add_argument('--model', type=str,
                        default=str(default_model_path),
                        help='MuJoCo model XML path')
    parser.add_argument('--frames', type=int, default=2500,
                        help='Number of frames to render')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path')
    parser.add_argument('--height', type=float, default=0.0,
                        help='Height offset to lift model above ground (meters)')
    parser.add_argument('--view', type=str, default='diagonal',
                        choices=['diagonal', 'side', 'front', 'top'],
                        help='Camera view type for rendering')

    args = parser.parse_args()

    data_filename = args.data if args.data.endswith('.npz') else f'{args.data}.npz'
    npz_path = Path(data_filename)
    if not npz_path.exists():
        npz_path = default_data_dir / data_filename
        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find reference file: {data_filename}")

    if args.output is None:
        args.output = f'ref_{npz_path.stem}_{args.view}.mp4'

    render_reference_motion(npz_path, args.model, args.output,
                            args.frames, args.height, args.view)


if __name__ == '__main__':
    main()
