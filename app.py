import streamlit as st
import numpy as np
import librosa
import os
import time
import plotly.graph_objects as go

# 1. 페이지 설정 (이모지 제거, 와이드 레이아웃 유지)
st.set_page_config(page_title="압축기 예지보전 시스템", layout="wide")

# CSS를 통한 산업용 UI 커스텀
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .sensor-box {
        background-color: #303030FF; border: 1px solid #333; 
        border-radius: 4px; padding: 10px; margin-bottom: 10px;
    }
    .sensor-title { font-size: 16px; color: #FFFFFF; margin-bottom: 2px; }
    .sensor-value { font-size: 20px; font-weight: 600; color: #00E676; }
    .log-box {
        background-color: #1A1A1A; border-left: 3px solid #FF1744;
        padding: 8px; margin-bottom: 5px; font-size: 13px; color: #FF8A80;
    }
    h1, h2, h3 { color: #E0E0E0; }
    </style>
""", unsafe_allow_html=True)

# 2. 데이터 로드 및 사전 분석
@st.cache_data
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    rms = librosa.feature.rms(y=y)[0]
    total_sec = len(y) // sr
    frames_per_sec = sr / 512  
    return y, sr, S_dB, rms, total_sec, frames_per_sec

# 3. 세션 상태 초기화
if 'current_sec' not in st.session_state: st.session_state.current_sec = 1
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'selected_audio' not in st.session_state: st.session_state.selected_audio = ""
if 'alarm_log' not in st.session_state: st.session_state.alarm_log = []

# 4. 사이드바 (제어부)
with st.sidebar:
    st.title("데이터 전송 제어부")
    
    DATA_DIR = "data"
    available_files = [
        "normal_id_00.wav", "normal_id_02.wav",
        "anomaly_id_01.wav", "anomaly_id_02.wav", 
        "anomaly_id_03.wav", "anomaly_id_04.wav", "anomaly_id_05.wav"
    ]
    
    selected_file = st.selectbox("모니터링 파일 선택", available_files)
    file_path = os.path.join(DATA_DIR, selected_file)

    if selected_file != st.session_state.selected_audio:
        st.session_state.selected_audio = selected_file
        st.session_state.current_sec = 1
        st.session_state.is_running = False
        st.session_state.alarm_log = []

    threshold = st.slider("알람 임계값 (RMS)", min_value=0.0, max_value=0.3, value=0.05, step=0.01)

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("▶ 모니터링 시작"): st.session_state.is_running = True
    if col_btn2.button("⏸ 일시 정지"): st.session_state.is_running = False

    st.markdown("---")
    status_html = "<span style='color:#00FF00;'>센서 스트리밍 중</span>" if st.session_state.is_running else "<span style='color:#FFA500;'>스트리밍 대기</span>"
    st.markdown(f"**통신 상태:** {status_html}", unsafe_allow_html=True)
    
    if os.path.exists(file_path):
        _, _, _, _, total_sec, _ = process_audio(file_path)
        st.session_state.current_sec = st.slider("시간(초) 탐색", min_value=1, max_value=total_sec, value=st.session_state.current_sec)

# 5. 메인 레이아웃 분할
st.title("수소발전소 압축기 실시간 음향 모니터링")

if os.path.exists(file_path):
    y, sr, S_dB_full, rms_full, total_sec, frames_per_sec = process_audio(file_path)
    
    current_frame = int(st.session_state.current_sec * frames_per_sec)
    S_dB_current = S_dB_full[:, :current_frame]
    rms_current = rms_full[:current_frame]
    times_current = librosa.frames_to_time(np.arange(len(rms_current)), sr=sr)
    
    # 상태 판정 로직
    current_max_rms = np.max(rms_current) if len(rms_current) > 0 else 0
    is_anomaly = current_max_rms > threshold
    
    if is_anomaly:
        msg = f"파열음 감지 (RMS: {current_max_rms:.3f})"
        log_entry = f"[{st.session_state.current_sec}s] {msg}"
        if log_entry not in st.session_state.alarm_log:
            st.session_state.alarm_log.insert(0, log_entry)
            
    st.session_state.alarm_log = st.session_state.alarm_log[:10]

    # --- 3단 그리드 구성 ---
    col_kpi, col_charts, col_logs = st.columns([2, 6, 2])
    
    # 좌측: 핵심 KPI
    with col_kpi:
        st.subheader("음향 지표")
        def metric_card(title, value, unit="", color="#00E676"):
            st.markdown(f"""
            <div class="sensor-box">
                <div class="sensor-title">{title}</div>
                <div class="sensor-value" style="color:{color};">{value} {unit}</div>
            </div>
            """, unsafe_allow_html=True)
            
        metric_card("현재 재생 시간", f"{st.session_state.current_sec}", "s")
        
        # 현재 화면 안의 최고 RMS에 따라 색상 변경
        rms_color = "#FF1744" if is_anomaly else "#00E676"
        metric_card("구간 최대 강도 (Max RMS)", f"{current_max_rms:.4f}", "", color=rms_color)
        metric_card("알람 임계값 (Threshold)", f"{threshold:.2f}", "", color="#888888")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if is_anomaly:
            st.markdown("### 설비 상태: <span style='color:#FF1744;'>이상 발생 (FAULT)</span>", unsafe_allow_html=True)
        else:
            st.markdown("### 설비 상태: <span style='color:#00E676;'>정상 운전 (NORMAL)</span>", unsafe_allow_html=True)

    # 중앙: Plotly 시각화 차트
    with col_charts:
        # 1. 멜 스펙트로그램 (Plotly Heatmap)
        heatmap_x = librosa.frames_to_time(np.arange(S_dB_current.shape[1]), sr=sr)
        fig_spec = go.Figure(data=go.Heatmap(
            x=heatmap_x, z=S_dB_current, colorscale='magma', showscale=False
        ))
        fig_spec.update_layout(
            title="멜 스펙트로그램 분석",
            template="plotly_dark", height=250, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="시간 (s)", yaxis_title="주파수 대역 (Mel)",
            xaxis=dict(range=[0, total_sec]) # X축을 전체 시간으로 고정
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        
        # 2. RMS 에너지 추세 (이상 발생 시 빨간색 변경)
        fig_rms = go.Figure()
        fig_rms.add_trace(go.Scatter(
            x=times_current, y=rms_current, mode='lines', 
            name="RMS", line=dict(color=rms_color, width=2)
        ))
        fig_rms.add_hline(y=threshold, line_dash="dash", line_color="#FF1744", annotation_text="임계값")
        
        fig_rms.update_layout(
            title="음향 에너지 (RMS) 추세",
            template="plotly_dark", height=250, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="시간 (s)", yaxis_title="RMS",
            xaxis=dict(range=[0, total_sec]),
            yaxis=dict(range=[0, max(0.2, np.max(rms_full) + 0.05)])
        )
        st.plotly_chart(fig_rms, use_container_width=True)

    # 우측: 알람 로그
    with col_logs:
        st.subheader("알람 발생 이력")
        if st.session_state.alarm_log:
            for log in st.session_state.alarm_log:
                st.markdown(f'<div class="log-box">{log}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#888; font-size:12px;">현재 발생한 알람이 없습니다.</div>', unsafe_allow_html=True)

    # 6. 실시간 업데이트 루프 (Data Pump)
    if st.session_state.is_running:
        if st.session_state.current_sec < total_sec:
            time.sleep(1) 
            st.session_state.current_sec += 1
            st.rerun() 
        else:
            st.session_state.is_running = False
            st.sidebar.warning("오디오 모니터링이 종료되었습니다.")
            st.rerun()
else:
    st.error("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
