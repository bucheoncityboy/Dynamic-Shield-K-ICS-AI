"""
K-ICS 동적 환헤지 Operations Dashboard
====================================
3가지 핵심 운영 기능을 시각화하는 종합 대시보드

1. 일일 운용: 실시간 국면 인식 → 헤지 비율 조정
2. 위기 대응: VIX 급등 감지 → Emergency 모드 전환
3. 규제 준수: K-ICS 모니터링 → 자동 대응

실행: streamlit run dashboard/operations_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 기존 모듈 임포트 시도
try:
    from core.kics_real import RatioKICSEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

try:
    from core.agent import DynamicShieldAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(
    page_title="K-ICS 동적 환헤지 Control Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CSS 스타일
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .status-normal { background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 5px; }
    .status-transition { background-color: #FFC107; color: black; padding: 0.5rem 1rem; border-radius: 5px; }
    .status-panic { background-color: #F44336; color: white; padding: 0.5rem 1rem; border-radius: 5px; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .safe-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 세션 상태 초기화
# ==========================================
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'history' not in st.session_state:
    st.session_state.history = {
        'timestamp': [],
        'vix': [],
        'hedge_ratio': [],
        'kics_ratio': [],
        'correlation': [],
        'regime': [],
        'actions': []
    }
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'emergency_mode' not in st.session_state:
    st.session_state.emergency_mode = False


# ==========================================
# 시뮬레이션 데이터 생성
# ==========================================
def generate_market_state(step, scenario='normal'):
    """시장 상태 시뮬레이션"""
    np.random.seed(step)
    
    # 시나리오별 VIX 시뮬레이션
    if scenario == 'crisis':
        base_vix = 35 + 15 * np.sin(step * 0.1)
        vix = base_vix + np.random.normal(0, 5)
    elif scenario == 'volatile':
        base_vix = 25 + 10 * np.sin(step * 0.2)
        vix = base_vix + np.random.normal(0, 3)
    else:  # normal
        base_vix = 15 + 5 * np.sin(step * 0.05)
        vix = base_vix + np.random.normal(0, 2)
    
    vix = max(10, min(80, vix))  # 범위 제한
    
    # Correlation: VIX와 약한 음의 상관관계
    correlation = -0.3 + 0.4 * np.random.random() - (vix - 20) * 0.01
    correlation = max(-0.9, min(0.5, correlation))
    
    return vix, correlation


def determine_regime(vix):
    """VIX 기반 Regime 결정"""
    if vix >= 30:
        return 'PANIC', '🔴'
    elif vix >= 20:
        return 'TRANSITION', '🟡'
    else:
        return 'NORMAL', '🟢'


def calculate_optimal_hedge(vix, correlation, kics_ratio, current_hedge):
    """최적 헤지 비율 계산 (간단한 룰 기반)"""
    regime, _ = determine_regime(vix)
    
    # Safety Layer 오버라이드
    if kics_ratio < 100:
        return 1.0, "CRITICAL: K-ICS < 100%, FORCE 100%"
    elif kics_ratio < 120:
        target = min(current_hedge + 0.1, 1.0)
        return target, f"DANGER: K-ICS = {kics_ratio:.0f}%, Increasing Hedge"
    
    # Regime 기반 타겟
    if regime == 'PANIC':
        target = 1.0 if current_hedge < 0.9 else current_hedge
        action = "PANIC: Rapid Hedge Increase"
    elif regime == 'TRANSITION':
        target = 0.7 if current_hedge < 0.7 else current_hedge
        action = "TRANSITION: Gradual Increase"
    else:
        # Normal: 비용 절감
        if correlation < -0.3:
            target = max(current_hedge - 0.05, 0.3)
            action = "NORMAL: Natural Hedge Effect, Reducing"
        else:
            target = max(current_hedge - 0.02, 0.4)
            action = "NORMAL: Cost Optimization"
    
    return target, action


def calculate_kics_ratio(hedge_ratio, correlation):
    """K-ICS 비율 계산 (간단한 모델)"""
    # 기본 K-ICS: 150% 기준
    base_kics = 150
    
    # 헤지 비율 효과: 높을수록 안정적 but 비용 증가
    hedge_effect = hedge_ratio * 20  # 0~20%p
    
    # Correlation 효과: 음의 상관관계일 때 분산 효과
    corr_effect = -correlation * 15  # -7.5 ~ +13.5%p
    
    # 변동성 요소
    noise = np.random.normal(0, 5)
    
    kics = base_kics + hedge_effect + corr_effect + noise
    return max(80, min(200, kics))


# ==========================================
# 헤더
# ==========================================
st.markdown('<div class="main-header">🛡️ K-ICS 동적 환헤지 Control Center</div>', unsafe_allow_html=True)

# 현재 시간 표시
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    status_text = "🔴 EMERGENCY" if st.session_state.emergency_mode else "🟢 OPERATIONAL"
    st.markdown(f"**System Status:** {status_text}")
with col3:
    st.markdown(f"**Step:** {st.session_state.current_step}")


# ==========================================
# 사이드바: 컨트롤 패널
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # 시나리오 선택
    scenario = st.selectbox(
        "📊 Market Scenario",
        ['normal', 'volatile', 'crisis'],
        format_func=lambda x: {'normal': '🟢 Normal', 'volatile': '🟡 Volatile', 'crisis': '🔴 Crisis'}[x]
    )
    
    # 시뮬레이션 컨트롤
    st.subheader("🎮 Simulation Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start" if not st.session_state.simulation_running else "⏸️ Pause"):
            st.session_state.simulation_running = not st.session_state.simulation_running
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.history = {
                'timestamp': [], 'vix': [], 'hedge_ratio': [],
                'kics_ratio': [], 'correlation': [], 'regime': [], 'actions': []
            }
            st.session_state.current_step = 0
            st.session_state.emergency_mode = False
            st.rerun()
    
    # 수동 VIX 주입
    st.subheader("🧪 Manual VIX Injection")
    manual_vix = st.slider("VIX Override", 10, 80, 20)
    if st.button("💉 Inject VIX"):
        st.session_state.manual_vix_override = manual_vix
        st.success(f"VIX set to {manual_vix}")
    
    # 임계값 설정
    st.subheader("⚡ Thresholds")
    vix_transition = st.number_input("VIX Transition", value=20, min_value=15, max_value=30)
    vix_panic = st.number_input("VIX Panic", value=30, min_value=25, max_value=50)
    kics_danger = st.number_input("K-ICS Danger", value=120, min_value=100, max_value=150)


# ==========================================
# 시뮬레이션 스텝 실행
# ==========================================
if st.session_state.simulation_running:
    step = st.session_state.current_step
    
    # 시장 상태 생성
    if hasattr(st.session_state, 'manual_vix_override'):
        vix = st.session_state.manual_vix_override
        delattr(st.session_state, 'manual_vix_override')
    else:
        vix, correlation = generate_market_state(step, scenario)
    
    _, correlation = generate_market_state(step, scenario)
    
    # 현재 헤지 비율
    current_hedge = st.session_state.history['hedge_ratio'][-1] if st.session_state.history['hedge_ratio'] else 0.5
    
    # K-ICS 계산
    kics_ratio = calculate_kics_ratio(current_hedge, correlation)
    
    # 최적 헤지 결정
    new_hedge, action = calculate_optimal_hedge(vix, correlation, kics_ratio, current_hedge)
    
    # Emergency 모드 판단
    st.session_state.emergency_mode = vix >= vix_panic or kics_ratio < 100
    
    # Regime 결정
    regime, regime_icon = determine_regime(vix)
    
    # 히스토리 업데이트
    st.session_state.history['timestamp'].append(datetime.now())
    st.session_state.history['vix'].append(vix)
    st.session_state.history['hedge_ratio'].append(new_hedge)
    st.session_state.history['kics_ratio'].append(kics_ratio)
    st.session_state.history['correlation'].append(correlation)
    st.session_state.history['regime'].append(regime)
    st.session_state.history['actions'].append(action)
    
    # 최근 100개만 유지
    for key in st.session_state.history:
        if len(st.session_state.history[key]) > 100:
            st.session_state.history[key] = st.session_state.history[key][-100:]
    
    st.session_state.current_step += 1


# ==========================================
# 현재 상태 가져오기
# ==========================================
if st.session_state.history['vix']:
    current_vix = st.session_state.history['vix'][-1]
    current_hedge = st.session_state.history['hedge_ratio'][-1]
    current_kics = st.session_state.history['kics_ratio'][-1]
    current_corr = st.session_state.history['correlation'][-1]
    current_regime, regime_icon = determine_regime(current_vix)
    latest_action = st.session_state.history['actions'][-1]
else:
    current_vix = 15
    current_hedge = 0.5
    current_kics = 150
    current_corr = 0
    current_regime = 'NORMAL'
    regime_icon = '🟢'
    latest_action = "Waiting for simulation..."


# ==========================================
# Section 1: 일일 운용 (Daily Operations)
# ==========================================
st.header("📈 Section 1: 일일 운용 (Daily Operations)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Regime 상태
    regime_colors = {'NORMAL': '#4CAF50', 'TRANSITION': '#FFC107', 'PANIC': '#F44336'}
    st.markdown(f"""
    <div style="background-color: {regime_colors.get(current_regime, '#9E9E9E')}; 
                padding: 1rem; border-radius: 10px; text-align: center; color: white;">
        <h3 style="margin:0;">Regime</h3>
        <h1 style="margin:0;">{regime_icon} {current_regime}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric(
        label="VIX Index",
        value=f"{current_vix:.1f}",
        delta=f"{current_vix - 20:.1f}" if current_vix > 20 else None,
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Hedge Ratio",
        value=f"{current_hedge*100:.0f}%",
        delta=f"{(current_hedge - 0.5)*100:+.0f}%p vs 50%"
    )

with col4:
    st.metric(
        label="Correlation",
        value=f"{current_corr:.2f}",
        help="주식-환율 상관계수 (음수: Natural Hedge 효과)"
    )

# Regime 상태 표시기
st.subheader("🚦 Regime State Diagram")
regime_col1, regime_col2, regime_col3 = st.columns(3)

for col, (regime_name, vix_range, hedge_target) in zip(
    [regime_col1, regime_col2, regime_col3],
    [('NORMAL', 'VIX < 20', '40-50%'), 
     ('TRANSITION', '20 ≤ VIX < 30', '60-70%'), 
     ('PANIC', 'VIX ≥ 30', '90-100%')]
):
    is_current = current_regime == regime_name
    bg_color = regime_colors.get(regime_name, '#9E9E9E') if is_current else '#E0E0E0'
    text_color = 'white' if is_current else '#666'
    border = '3px solid #000' if is_current else 'none'
    
    with col:
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 1rem; border-radius: 10px; 
                    text-align: center; color: {text_color}; border: {border};">
            <h4 style="margin:0;">{regime_name}</h4>
            <p style="margin:0.5rem 0;">{vix_range}</p>
            <p style="margin:0;">Target: {hedge_target}</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# Section 2: 위기 대응 (Emergency Response)
# ==========================================
st.header("🚨 Section 2: 위기 대응 (Emergency Response)")

# Emergency 경보
if st.session_state.emergency_mode:
    st.markdown(f"""
    <div class="alert-box">
        <h3>⚠️ EMERGENCY MODE ACTIVATED</h3>
        <p><strong>VIX:</strong> {current_vix:.1f} | <strong>K-ICS:</strong> {current_kics:.0f}%</p>
        <p><strong>Action:</strong> {latest_action}</p>
        <p>Safety Layer Override Active - Forcing Maximum Hedge</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="safe-box">
        <h3>✅ SYSTEM OPERATIONAL</h3>
        <p>All parameters within normal range. AI-driven optimization active.</p>
    </div>
    """, unsafe_allow_html=True)

# VIX 모니터링 차트
if len(st.session_state.history['vix']) > 1:
    fig_vix = go.Figure()
    
    # VIX 라인
    fig_vix.add_trace(go.Scatter(
        y=st.session_state.history['vix'],
        mode='lines',
        name='VIX',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # 임계값 라인
    fig_vix.add_hline(y=vix_transition, line_dash="dash", line_color="orange", 
                      annotation_text="Transition (20)")
    fig_vix.add_hline(y=vix_panic, line_dash="dash", line_color="red", 
                      annotation_text="Panic (30)")
    
    # Panic 구간 음영
    fig_vix.add_hrect(y0=vix_panic, y1=80, fillcolor="red", opacity=0.1, 
                      layer="below", line_width=0)
    fig_vix.add_hrect(y0=vix_transition, y1=vix_panic, fillcolor="orange", opacity=0.1, 
                      layer="below", line_width=0)
    
    fig_vix.update_layout(
        title="📊 VIX Real-time Monitoring",
        yaxis_title="VIX Index",
        xaxis_title="Time Step",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig_vix, use_container_width=True)

# De-risking 진행률
if st.session_state.emergency_mode and current_hedge < 1.0:
    progress = current_hedge / 1.0
    st.subheader("🔄 De-risking Progress")
    st.progress(progress)
    st.caption(f"Current: {current_hedge*100:.0f}% → Target: 100%")


# ==========================================
# Section 3: 규제 준수 (Regulatory Compliance)
# ==========================================
st.header("📋 Section 3: 규제 준수 (Regulatory Compliance)")

col1, col2 = st.columns([1, 2])

with col1:
    # K-ICS 게이지
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_kics,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "K-ICS Ratio (%)"},
        delta={'reference': 150, 'position': "top"},
        gauge={
            'axis': {'range': [80, 200], 'tickwidth': 1},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [80, 100], 'color': "#FFCDD2"},  # 위험
                {'range': [100, 150], 'color': "#FFF9C4"},  # 안전
                {'range': [150, 200], 'color': "#C8E6C9"}   # 최적
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # 상태 설명
    if current_kics < 100:
        st.error("⚠️ CRITICAL: K-ICS < 100% - 규제 위반 위험!")
    elif current_kics < 120:
        st.warning("🟡 CAUTION: K-ICS < 120% - 주의 필요")
    elif current_kics < 150:
        st.info("🟢 SAFE: K-ICS 100-150% - 안전 구간")
    else:
        st.success("✅ OPTIMAL: K-ICS > 150% - 최적 상태")

with col2:
    # 시계열 차트 (3개 지표)
    if len(st.session_state.history['vix']) > 1:
        fig_combined = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Hedge Ratio (%)", "K-ICS Ratio (%)", "Correlation")
        )
        
        # Hedge Ratio
        fig_combined.add_trace(
            go.Scatter(y=[h*100 for h in st.session_state.history['hedge_ratio']], 
                      mode='lines', name='Hedge', line=dict(color='#4CAF50')),
            row=1, col=1
        )
        
        # K-ICS
        fig_combined.add_trace(
            go.Scatter(y=st.session_state.history['kics_ratio'], 
                      mode='lines', name='K-ICS', line=dict(color='#2196F3')),
            row=2, col=1
        )
        fig_combined.add_hline(y=100, line_dash="dash", line_color="red", row=2, col=1)
        fig_combined.add_hline(y=150, line_dash="dash", line_color="green", row=2, col=1)
        
        # Correlation
        fig_combined.add_trace(
            go.Scatter(y=st.session_state.history['correlation'], 
                      mode='lines', name='Correlation', line=dict(color='#FF9800')),
            row=3, col=1
        )
        fig_combined.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        fig_combined.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_combined, use_container_width=True)


# ==========================================
# 액션 로그
# ==========================================
st.subheader("📝 Action Log")

if st.session_state.history['actions']:
    # 최근 10개 액션만 표시
    recent_actions = list(zip(
        st.session_state.history['timestamp'][-10:],
        st.session_state.history['regime'][-10:],
        st.session_state.history['actions'][-10:]
    ))[::-1]  # 최신순
    
    log_df = pd.DataFrame(recent_actions, columns=['Time', 'Regime', 'Action'])
    log_df['Time'] = log_df['Time'].apply(lambda x: x.strftime('%H:%M:%S'))
    
    st.dataframe(log_df, use_container_width=True, hide_index=True)
else:
    st.info("No actions yet. Start the simulation to see action logs.")


# ==========================================
# 자동 새로고침
# ==========================================
if st.session_state.simulation_running:
    time.sleep(1)
    st.rerun()
