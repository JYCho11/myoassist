import numpy as np
import h5py

# 파일 경로 설정
npz_source_file = 'S004_level_08mps.npz'  # 관절 데이터 원본
h5_source_file = 'S004_level_08mps.h5'    # Gait Event(Heel Strike) 추출용
output_file = 'S004_segmented_final.npz'  # 최종 결과 파일

print("1. NPZ 데이터 로딩 중...")
source_npz = np.load(npz_source_file, allow_pickle=True)
output_data = {}

# ---------------------------------------------------------
# [Step 1] 관절 데이터(series_data) 평탄화 (Unpacking)
# ---------------------------------------------------------
if 'series_data' in source_npz:
    series_dict = source_npz['series_data'].item()
    for key, value in series_dict.items():
        output_data[key] = value
    print(f" - 관절 데이터 {len(series_dict)}개 추출 완료")
else:
    print(" - 경고: series_data를 찾을 수 없습니다.")

# ---------------------------------------------------------
# [Step 2] H5 파일에서 Gait Cycle(Heel Strike) 추출
# ---------------------------------------------------------
print("2. H5 파일에서 Heel Strike 이벤트 계산 중...")
segments_list = []

try:
    with h5py.File(h5_source_file, 'r') as f:
        # 지면 반력(GRF) 데이터 경로 (오른발 수직 힘: Fz)
        # 일반적으로 보행 사이클은 오른발 뒤꿈치가 닿는 순간(RHS)을 기준으로 나눕니다.
        grf_right_z = f['MoCap/grf_measured/right/force/Fz'][:]
        time_array = f['MoCap/grf_measured/time'][:]
        
        # Heel Strike 감지 알고리즘
        # 조건: 힘이 20N 미만이었다가 20N 이상으로 바뀌는 순간 (Rising Edge)
        threshold = 20.0 
        
        # 이전 프레임은 임계값보다 작고(&), 현재 프레임은 임계값보다 큰 인덱스 찾기
        heel_strike_indices = np.where(
            (grf_right_z[:-1] < threshold) & (grf_right_z[1:] >= threshold)
        )[0] + 1
        
        # 해당 인덱스의 시간 정보를 segments_list로 저장
        segments_list = time_array[heel_strike_indices].tolist()
        
        print(f" - 감지된 보행 주기(Heel Strikes) 수: {len(segments_list)}개")
        print(f" - 첫 3개 타임스탬프: {segments_list[:3]}")

except Exception as e:
    print(f" - H5 처리 중 오류 발생: {e}")
    # 오류 시 전체 구간을 하나의 세그먼트로 처리
    if 'metadata' in source_npz:
        md = source_npz['metadata'].item()
        duration = md.get('data_length', 0) / md.get('sample_rate', 100)
        segments_list = [0.0, duration]
        print(" - 오류로 인해 전체 구간을 하나의 세그먼트로 설정했습니다.")

# ---------------------------------------------------------
# [Step 3] 메타데이터 통합 및 저장
# ---------------------------------------------------------
print("3. 데이터 병합 및 저장 중...")

# 메타데이터 생성 ('metadata' -> 'meta_data' 이름 변경)
meta_data = {}
if 'metadata' in source_npz:
    source_metadata = source_npz['metadata'].item()
    # 기존 메타데이터 복사
    for k, v in source_metadata.items():
        meta_data[k] = v

# 추출한 segments_list 추가
meta_data['segments_list'] = segments_list
output_data['meta_data'] = meta_data

# 최종 저장
np.savez(output_file, **output_data)

print(f"\n[완료] '{output_file}' 파일이 성공적으로 생성되었습니다.")
print("이 파일은 segmented.npz와 동일한 구조(Key)를 가지며, H5 파일의 Gait 정보를 포함합니다.")