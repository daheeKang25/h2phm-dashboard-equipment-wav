import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import time

# 페이지 기본 설정
st.set_page_config(page_title="압축기 음향 실시간 모니터링", layout="wide")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # 윈도우 (맥은 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 데이터 로드 및 사전 분석 (캐싱으로 속도 최적화)
# ---------------------------------------------------------
@st.cache_data
def process_audio(file_path):
    # 16,000Hz로 오디오 로드
    y, sr = librosa.load(file_path, sr=16000)
    
    # 전체 멜 스펙트로그램 사전 계산
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 전체 RMS 에너지 사전 계산
    rms = librosa.feature.rms(y=y)[0]
    
    # 총 재생 시간(초) 및 1초당 프레임 수 계산 (기본 hop_length=512)
    total_sec = len(y) // sr
    frames_per_sec = sr / 512  
    
    return y, sr, S_dB, rms, total_sec, frames_per_sec

# ---------------------------------------------------------
# 2. 세션 상태(Session State) 초기화
# ---------------------------------------------------------
if 'current_sec' not in st.session_state:
    st.session_state.current_sec = 1
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'selected_audio' not in st.session_state:
    st.session_state.selected_audio = ""

# ---------------------------------------------------------
# 3. 사이드바 (제어부)
# ---------------------------------------------------------
st.sidebar.title("🎛️ 실시간 분석 제어부")

DATA_DIR = "data"
available_files = [
    "normal_id_00.wav", 
    "normal_id_02.wav",
    "anomaly_id_01.wav", 
    "anomaly_id_02.wav", 
    "anomaly_id_03.wav",
    "anomaly_id_04.wav", 
    "anomaly_id_05.wav"
]

selected_file = st.sidebar.selectbox("모니터링할 파일 선택", available_files)
file_path = os.path.join(DATA_DIR, selected_file)

# 파일이 바뀌면 시간(인덱스) 초기화
if selected_file != st.session_state.selected_audio:
    st.session_state.selected_audio = selected_file
    st.session_state.current_sec = 1
    st.session_state.is_running = False

threshold = st.sidebar.slider("알람 임계값 (RMS)", min_value=0.0, max_value=0.3, value=0.05, step=0.01)

# 전송 시작/정지 버튼
col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("▶ 모니터링 시작"):
    st.session_state.is_running = True
if col_btn2.button("⏸ 일시 정지"):
    st.session_state.is_running = False

# ---------------------------------------------------------
# 4. 메인 대시보드 화면
# ---------------------------------------------------------
st.title("🎧 수소발전소 압축기 실시간 음향 모니터링")

if os.path.exists(file_path):
    # 오디오 처리 (캐싱됨)
    y, sr, S_dB_full, rms_full, total_sec, frames_per_sec = process_audio(file_path)
    
    # 사용자가 슬라이더로 직접 시점 이동 가능
    st.session_state.current_sec = st.sidebar.slider(
        "시간(초) 탐색", 
        min_value=1, max_value=total_sec, value=st.session_state.current_sec
    )
    
    st.sidebar.markdown("---")
    if st.session_state.is_running:
        st.sidebar.success("🟢 실시간 음향 센서 스트리밍 중...")
    else:
        st.sidebar.warning("🟡 센서 스트리밍 대기 중")

    # 현재 초(sec)까지의 프레임 계산 후 데이터 슬라이싱 (1초 단위 갱신)
    current_frame = int(st.session_state.current_sec * frames_per_sec)
    
    S_dB_current = S_dB_full[:, :current_frame]
    rms_current = rms_full[:current_frame]
    times_current = librosa.frames_to_time(np.arange(len(rms_current)), sr=sr)
    
    col1, col2 = st.columns(2)
    
    # 좌측: 스펙트로그램 실시간 렌더링
    with col1:
        st.subheader(f"📊 스펙트로그램 (0 ~ {st.session_state.current_sec}초)")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        
        # x축을 총 길이(total_sec)로 고정하여 차트가 왼쪽부터 서서히 차오르는 효과 구현
        img = librosa.display.specshow(S_dB_current, x_axis='time', y_axis='mel', sr=sr, ax=ax1, cmap='magma')
        ax1.set_xlim(0, total_sec)
        
        fig1.colorbar(img, ax=ax1, format='%+2.0f dB')
        st.pyplot(fig1)
        
    # 우측: RMS 추세 실시간 렌더링
    with col2:
        st.subheader(f"📈 음향 에너지 추세 (0 ~ {st.session_state.current_sec}초)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        
        ax2.plot(times_current, rms_current, label="RMS 에너지", color='teal')
        ax2.axhline(y=threshold, color='red', linestyle='--', label="임계값")
        
        # x축 고정 및 y축 여유 공간 확보
        ax2.set_xlim(0, total_sec)
        ax2.set_ylim(0, max(0.2, np.max(rms_full) + 0.05)) 
        
        ax2.set_xlabel("시간 (초)")
        ax2.set_ylabel("RMS")
        ax2.legend(loc="upper right")
        st.pyplot(fig2)
        
        # 현재 화면(시간대)의 최대 RMS 기반 실시간 알람 판정
        if len(rms_current) > 0:
            current_max_rms = np.max(rms_current)
            if current_max_rms > threshold:
                st.error(f"🚨 **이상 감지:** 임계값 초과! (최대 RMS: {current_max_rms:.3f})")
            else:
                st.success(f"✅ **상태 정상:** 안정적입니다. (최대 RMS: {current_max_rms:.3f})")

    # ---------------------------------------------------------
    # 5. 1초 자동 갱신 루프 (Data Pump)
    # ---------------------------------------------------------
    if st.session_state.is_running:
        if st.session_state.current_sec < total_sec:
            time.sleep(1) # 1초 대기하여 실제 시간 흐름 모사
            st.session_state.current_sec += 1
            st.rerun() # 화면 다시 그리기
        else:
            st.session_state.is_running = False
            st.sidebar.info("오디오 끝에 도달하여 모니터링이 종료되었습니다.")
            st.rerun()

else:
    st.error(f"⚠️ '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")