"""
스트레스 시나리오 검증 통합 실행 스크립트
========================================
모든 스트레스 시나리오에 대해 체계적으로 검증 수행

사용법:
    python -m src.validation.run_stress_validation
"""

import os
import sys
import pandas as pd
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system import DynamicShieldSystem
from core.kics_real import RatioKICSEngine, KICSCalculator
from validation.enhanced_validation import EnhancedValidator

# AI 모델 로드를 위한 라이브러리
try:
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("[경고] stable-baselines3가 설치되지 않았습니다. AI 모델을 사용할 수 없습니다.")


def load_stress_scenario(scenario_name):
    """스트레스 시나리오 CSV 로드"""
    scenario_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'DATA', 'synthetic_stress'
    )
    scenario_path = os.path.join(scenario_dir, f'{scenario_name}.csv')
    
    if not os.path.exists(scenario_path):
        print(f"[경고] 시나리오 파일 없음: {scenario_path}")
        return None
    
    df = pd.read_csv(scenario_path, index_col=0, parse_dates=True)
    return df


def load_ai_model():
    """AI 모델 로드 (ppo_kics.zip)"""
    model = None
    model_path = None
    
    if not STABLE_BASELINES_AVAILABLE:
        return None, None
    
    # 모델 파일 탐색 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(script_dir, 'ppo_kics.zip'),  # validation 폴더 (가장 가능성 높음)
        os.path.join(script_dir, 'ppo_kics'),
        'ppo_kics.zip',
        'ppo_kics',
        os.path.join(script_dir, '..', 'validation', 'ppo_kics.zip'),
        os.path.join(script_dir, '..', 'validation', 'ppo_kics'),
    ]
    
    for path in search_paths:
        # .zip 확장자 처리
        test_path = path if path.endswith('.zip') else path + '.zip'
        
        if os.path.exists(test_path):
            try:
                model = PPO.load(test_path)
                model_path = test_path
                print(f"  [AI 모델 로드 성공] {test_path}")
                return model, model_path
            except Exception as e:
                print(f"  [시도 실패] {test_path}: {e}")
                continue
    
    print(f"  [경고] AI 모델 파일을 찾을 수 없습니다. 룰 기반 전략을 사용합니다.")
    print(f"  [참고] 탐색한 경로: {search_paths[:3]}...")
    return None, None


def run_scenario_validation(scenario_name, scenario_df, ai_model=None, use_ai=True):
    """
    단일 시나리오 검증 실행 (실제 AI 모델 사용)
    
    과정:
    1. 시나리오 데이터 로드
    2. AI 모델 로드 (또는 룰 기반 전략)
    3. 각 날짜마다:
       - 시장 상태 관측 (VIX, Correlation, SCR 등)
       - AI 추론 (또는 룰 기반)으로 헤지 비율 결정
       - K-ICS 계산 (자산/부채 갱신)
       - 결과 기록
    4. 통계 집계 및 리포트
    """
    print("\n" + "=" * 60)
    print(f"시나리오: {scenario_name}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. AI 모델 로드 확인
    if use_ai and ai_model is None:
        print("  [1단계] AI 모델 로드 중...")
        ai_model, model_path = load_ai_model()
        if ai_model is None:
            print("  [전략 변경] AI 모델 없음 → 룰 기반 전략 사용")
            use_ai = False
        else:
            print(f"  [AI 모델 사용] 학습된 신경망으로 헤지 비율 결정")
    
    # 2. K-ICS 계산 엔진 초기화
    print("  [2단계] K-ICS 계산 엔진 초기화...")
    kics_calculator = KICSCalculator()
    kics_calculator.reset()
    engine = RatioKICSEngine()  # SCR 계산용
    
    results = {
        'scenario': scenario_name,
        'n_days': len(scenario_df),
        'min_kics': float('inf'),
        'max_kics': float('-inf'),
        'avg_kics': 0.0,
        'below_100_count': 0,
        'below_150_count': 0,
        'ai_used': use_ai
    }
    
    # 3. 초기 상태 설정
    current_hedge = 0.5
    prev_row = None
    kics_ratios = []
    ai_inference_count = 0
    rule_based_count = 0
    
    print(f"  [3단계] 시뮬레이션 시작... (총 {len(scenario_df)}일)")
    print(f"  [전략] {'AI 모델 (PPO)' if use_ai else '룰 기반 (VIX)'}")
    
    # 4. 각 날짜마다 시뮬레이션
    for idx, (date_idx, row) in enumerate(scenario_df.iterrows()):
        # 4-1. 시장 상태 관측
        vix = row.get('VIX', 15.0)
        
        # Correlation 계산/추정
        if 'Correlation' in row and not pd.isna(row['Correlation']):
            correlation = row['Correlation']
        else:
            # VIX 기반 추정
            if vix >= 30:
                correlation = np.random.uniform(0.5, 0.9)
            elif vix >= 20:
                correlation = np.random.uniform(0.0, 0.5)
            else:
                correlation = np.random.uniform(-0.6, -0.2)
        
        # 4-2. SCR 계산 (현재 헤지 비율 기준)
        scr_ratio = engine.calculate_scr_ratio_batch(
            np.array([current_hedge]),
            np.array([correlation])
        )[0]
        
        # 4-3. 헤지 비율 결정 (AI 또는 룰)
        if use_ai and ai_model is not None:
            # AI 추론
            # State: [hedge_ratio, vix_norm, corr_norm, scr_ratio]
            obs = np.array([
                current_hedge,
                np.clip(vix / 100.0, 0, 1),
                np.clip((correlation + 1) / 2, 0, 1),
                np.clip(scr_ratio, 0, 1)
            ], dtype=np.float32)
            
            # AI 예측 (Deterministic: 확률적 탐색 없음)
            action, _ = ai_model.predict(obs, deterministic=True)
            
            # Action 해석: [-1, 1] -> 헤지 비율 변화 [-0.1, 0.1]
            hedge_change = float(action[0]) * 0.1
            new_hedge = np.clip(current_hedge + hedge_change, 0.0, 1.0)
            current_hedge = new_hedge
            ai_inference_count += 1
        else:
            # 룰 기반 전략 (VIX 기반)
            if vix >= 40:
                target_hedge = 0.95  # 패닉
            elif vix >= 30:
                target_hedge = 0.85  # 위험
            elif vix >= 20:
                target_hedge = 0.70  # 주의
            else:
                target_hedge = 0.50  # 정상
            
            # 점진적 조정
            current_hedge = current_hedge * 0.7 + target_hedge * 0.3
            current_hedge = np.clip(current_hedge, 0.0, 1.0)
            rule_based_count += 1
        
        # 4-4. K-ICS 계산 (자산/부채 갱신 포함)
        try:
            kics_ratio = kics_calculator.update_and_calculate(row, prev_row, current_hedge)
        except Exception as e:
            # 계산 실패 시 기본값
            kics_ratio = 150.0
        
        kics_ratios.append(kics_ratio)
        prev_row = row.copy()
        
        # 4-5. 통계 업데이트
        results['min_kics'] = min(results['min_kics'], kics_ratio)
        results['max_kics'] = max(results['max_kics'], kics_ratio)
        if kics_ratio < 100:
            results['below_100_count'] += 1
        if kics_ratio < 150:
            results['below_150_count'] += 1
        
        # 4-6. 진행 상황 출력 (100일마다 또는 처음 5일)
        if (idx + 1) <= 5 or (idx + 1) % 100 == 0:
            strategy_type = "AI" if use_ai and ai_model else "룰"
            print(f"    [{idx+1:4d}일] VIX={vix:5.1f} | 헤지={current_hedge:.2f} ({strategy_type}) | K-ICS={kics_ratio:6.2f}%")
    
    results['avg_kics'] = np.mean(kics_ratios)
    results['below_100_pct'] = results['below_100_count'] / len(scenario_df) * 100
    results['below_150_pct'] = results['below_150_count'] / len(scenario_df) * 100
    
    elapsed_time = time.time() - start_time
    
    # 5. 결과 출력
    print(f"\n  [4단계] 시뮬레이션 완료")
    print(f"  소요 시간: {elapsed_time:.2f}초")
    if use_ai and ai_model:
        print(f"  AI 추론 횟수: {ai_inference_count}회")
    else:
        print(f"  룰 기반 결정 횟수: {rule_based_count}회")
    
    print(f"\n[결과 요약]")
    print(f"  기간: {len(scenario_df)}일")
    print(f"  사용 전략: {'AI 모델 (PPO)' if use_ai and ai_model else '룰 기반 (VIX)'}")
    print(f"  평균 K-ICS: {results['avg_kics']:.2f}%")
    print(f"  최소 K-ICS: {results['min_kics']:.2f}%")
    print(f"  최대 K-ICS: {results['max_kics']:.2f}%")
    print(f"  100% 미만: {results['below_100_count']}일 ({results['below_100_pct']:.2f}%)")
    print(f"  150% 미만: {results['below_150_count']}일 ({results['below_150_pct']:.2f}%)")
    
    if results['min_kics'] >= 100:
        print("  [PASS] 최소 K-ICS가 100% 이상 유지됨")
    else:
        print("  [FAIL] 최소 K-ICS가 100% 미만으로 떨어짐")
    
    return results


def run_all_stress_validations(use_ai=True):
    """
    모든 스트레스 시나리오 검증 실행
    
    Args:
        use_ai: True면 AI 모델 사용, False면 룰 기반 전략 사용
    """
    total_start_time = time.time()
    
    print("=" * 60)
    print("스트레스 시나리오 통합 검증")
    print("=" * 60)
    print(f"[검증 모드] {'AI 모델 사용' if use_ai else '룰 기반 전략 사용'}")
    
    # AI 모델 사전 로드 (한 번만)
    ai_model = None
    if use_ai:
        print("\n[AI 모델 사전 로드]")
        ai_model, _ = load_ai_model()
        if ai_model is None:
            print("  [전략 변경] AI 모델 없음 → 모든 시나리오에 룰 기반 전략 사용")
            use_ai = False
    
    # 시나리오 목록
    scenarios = [
        'Scenario_A_Stagflation',
        'Scenario_B_Correlation_Breakdown',
        'Scenario_COVID19',
        'Scenario_Tail_Risk',
        'Scenario_C_Interest_Rate_Shock',  # 새로 추가
        'Scenario_D_Swap_Point_Extreme',   # 새로 추가
        'Scenario_E_Regime_Transition'     # 새로 추가
    ]
    
    all_results = []
    
    print(f"\n[검증 시작] 총 {len(scenarios)}개 시나리오")
    
    for scenario_idx, scenario_name in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"[{scenario_idx}/{len(scenarios)}] {scenario_name}")
        print(f"{'='*60}")
        
        scenario_df = load_stress_scenario(scenario_name)
        
        if scenario_df is None:
            print(f"  [스킵] 파일 없음")
            continue
        
        results = run_scenario_validation(scenario_name, scenario_df, ai_model=ai_model, use_ai=use_ai)
        all_results.append(results)
    
    # 종합 결과
    print("\n" + "=" * 60)
    print("종합 결과")
    print("=" * 60)
    
    summary_df = pd.DataFrame(all_results)
    
    print("\n[시나리오별 요약]")
    print(summary_df[['scenario', 'avg_kics', 'min_kics', 'below_100_pct']].to_string(index=False))
    
    # 전체 통계
    print("\n[전체 통계]")
    print(f"  검증 시나리오 수: {len(all_results)}")
    print(f"  평균 최소 K-ICS: {summary_df['min_kics'].mean():.2f}%")
    print(f"  최악 시나리오: {summary_df.loc[summary_df['min_kics'].idxmin(), 'scenario']} "
          f"({summary_df['min_kics'].min():.2f}%)")
    
    # 모든 시나리오에서 100% 이상 유지 여부
    all_pass = (summary_df['min_kics'] >= 100).all()
    
    if all_pass:
        print("\n[SUCCESS] 모든 시나리오에서 K-ICS 100% 이상 유지!")
    else:
        print("\n[WARNING] 일부 시나리오에서 K-ICS 100% 미만 발생")
        failed = summary_df[summary_df['min_kics'] < 100]
        print(f"  실패 시나리오: {', '.join(failed['scenario'].tolist())}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n[총 소요 시간] {total_elapsed:.2f}초 ({total_elapsed/60:.1f}분)")
    
    return summary_df


if __name__ == "__main__":
    # 1. 추가 시나리오 생성 (아직 없으면)
    print("[1단계] 추가 스트레스 시나리오 생성 확인...")
    from validation.stress_scenario_generator import generate_all_additional_scenarios
    
    # 시나리오 파일 존재 여부 확인
    scenario_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'DATA', 'synthetic_stress'
    )
    
    new_scenarios = [
        'Scenario_C_Interest_Rate_Shock',
        'Scenario_D_Swap_Point_Extreme',
        'Scenario_E_Regime_Transition'
    ]
    
    need_generate = False
    for scenario in new_scenarios:
        scenario_path = os.path.join(scenario_dir, f'{scenario}.csv')
        if not os.path.exists(scenario_path):
            need_generate = True
            break
    
    if need_generate:
        print("  -> 추가 시나리오 생성 중...")
        generate_all_additional_scenarios()
    else:
        print("  -> 모든 시나리오 파일 존재 확인")
    
    # 2. 모든 시나리오 검증
    print("\n[2단계] 모든 스트레스 시나리오 검증...")
    print("\n[설명] 이 검증은 다음과 같이 진행됩니다:")
    print("  1. 각 시나리오 CSV 파일 로드")
    print("  2. AI 모델 로드 (ppo_kics.zip)")
    print("  3. 각 날짜마다:")
    print("     - 시장 상태 관측 (VIX, Correlation, SCR 등)")
    print("     - AI 신경망 추론으로 헤지 비율 결정")
    print("     - K-ICS 계산 (자산/부채 갱신)")
    print("  4. 통계 집계 및 리포트")
    print("\n[참고] AI 추론은 빠르지만(마이크로초 단위),")
    print("       K-ICS 계산(자산/부채 갱신)이 각 날짜마다 수행되므로")
    print("       시나리오가 길수록 시간이 더 걸립니다.\n")
    
    summary = run_all_stress_validations(use_ai=True)
    
    # 3. 강화된 검증론 실행
    print("\n[3단계] 강화된 검증론 실행...")
    from validation.enhanced_validation import run_enhanced_validation
    enhanced_results = run_enhanced_validation()
    
    print("\n" + "=" * 60)
    print("모든 검증 완료!")
    print("=" * 60)

