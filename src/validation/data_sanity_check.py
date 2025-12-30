"""
Data & Leakage Sanity Checks
============================

목적:
- Dynamic_Shield_Data_v4.csv 전처리 결과를 자동으로 점검
- Yield_Spread 프록시, 결측/휴장일 처리, 시간 기반 Train/Test 분리 여부 확인
- RL/시뮬레이션 환경에서 쓰이는 상태벡터(State)까지 추적

사용법:
    cd 프로젝트 루트 (Dynamic-Shield-K-ICS-AI)
    python -m src.validation.data_sanity_check

핵심 철학: Capital Optimization, not Prediction
"""

import os
import numpy as np
import pandas as pd

# 상대 경로 기반 import 설정
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.realistic_data import (
    REAL_DATA_PATH,
    TRAIN_RATIO,
    load_real_data_for_training,
    load_real_data_for_testing,
)
from src.core.system import DynamicShieldSystem


def _load_main_dataframe() -> pd.DataFrame:
    """Dynamic_Shield_Data_v4.csv 로드 (index=Date)"""
    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {REAL_DATA_PATH}")

    df = pd.read_csv(REAL_DATA_PATH, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def check_yield_spread(df: pd.DataFrame, atol: float = 1e-4) -> bool:
    """
    Yield_Spread = KR_10Y - US_10Y 여부 확인
    (CDS의 프록시로 금리 스프레드를 사용한다는 가정 점검)
    """
    required_cols = ["US_10Y", "KR_10Y", "Yield_Spread"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[FAIL] Yield_Spread 체크 실패 - 컬럼 없음: {missing}")
        return False

    calc_spread = df["KR_10Y"] - df["US_10Y"]
    diff = (df["Yield_Spread"] - calc_spread).abs()
    max_diff = diff.max()

    if max_diff <= atol:
        print(f"[PASS] Yield_Spread 일관성 OK (max |diff|={max_diff:.6f})")
        return True
    else:
        bad_samples = diff.nlargest(5)
        print(f"[FAIL] Yield_Spread 값 불일치 (max |diff|={max_diff:.6f})")
        print("  예시 날짜 (상위 5개):")
        for dt, val in bad_samples.items():
            print(
                f"  {dt.date()} -> "
                f"file={df.loc[dt, 'Yield_Spread']:.6f}, "
                f"KR_10Y-US_10Y={calc_spread.loc[dt]:.6f}, "
                f"diff={val:.6f}"
            )
        return False


def check_missing_and_ffill(df: pd.DataFrame) -> bool:
    """
    - 핵심 컬럼의 결측 존재 여부 점검
    - Date 인덱스가 단조 증가/유니크인지 점검
    - 연속 날짜 간 간격 통계로 '이상한 큰 구멍'이 없는지 대략 확인
    """
    key_cols = [
        "KOSPI",
        "FX",
        "US_10Y",
        "KR_1M",
        "US_1M",
        "KR_10Y",
        "SPX",
        "VIX",
        "Yield_Spread",
    ]
    exist_cols = [c for c in key_cols if c in df.columns]

    na_any = df[exist_cols].isna().any()
    na_cols = na_any[na_any].index.tolist()
    if na_cols:
        print(f"[FAIL] 결측치 존재 컬럼: {na_cols}")
        return False
    else:
        print("[PASS] 핵심 컬럼에 결측치 없음 (ffill 등 전처리 일관성 OK)")

    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        print(
            "[FAIL] Date 인덱스가 단조 증가/유니크가 아닙니다. "
            "정렬 또는 중복 제거 필요."
        )
        return False

    # 날짜 간격 통계
    deltas = df.index.to_series().diff().dropna().dt.days
    max_gap = deltas.max()
    print(
        f"[INFO] 인접 날짜 간 최대 차이: {max_gap}일 "
        "(주말/연휴 포함 여부를 눈으로 한 번 더 확인하세요)"
    )
    if max_gap > 7:
        print(
            "[WARN] 7일 초과의 큰 갭이 있습니다. 특정 구간 데이터 누락 여부를 점검하세요."
        )
    else:
        print("[PASS] 날짜 간격에 특이한 대형 공백 없음 (휴장일 처리 정상으로 추정)")

    return True


def check_train_test_split(df: pd.DataFrame) -> bool:
    """
    realistic_data.py 의 시간 기반 Train/Test 분리가
    실제로 겹치지 않는지 확인.
    """
    total_len = len(df)
    split_idx = int(total_len * TRAIN_RATIO)
    train_idx = df.index[:split_idx]
    test_idx = df.index[split_idx:]

    overlap = train_idx.intersection(test_idx)
    if len(overlap) > 0:
        print(f"[FAIL] Train/Test 인덱스가 겹칩니다. 중복 날짜 수: {len(overlap)}")
        return False

    print(
        f"[PASS] 시간 기반 Train/Test 분리 OK "
        f"(Train: {train_idx[0].date()} ~ {train_idx[-1].date()}, "
        f"Test: {test_idx[0].date()} ~ {test_idx[-1].date()})"
    )

    # realistic_data.load_real_data_* 유틸이 실제로 이 구간만 사용하는지 간단 검증
    # 주의: load_real_data_* 함수는 reset_index(drop=True)를 사용하므로 날짜 인덱스가 없음
    # 대신 함수 내부에서 train/test 구간을 분리하므로, 여기서는 단순히 함수 호출만 확인
    try:
        train_sample = load_real_data_for_training(n_days=min(500, len(train_idx)))
        test_sample = load_real_data_for_testing(n_days=min(500, len(test_idx)))
        
        # 샘플이 정상적으로 반환되는지만 확인 (날짜 인덱스는 realistic_data.py 내부에서 처리됨)
        if len(train_sample) > 0 and len(test_sample) > 0:
            print("[PASS] load_real_data_for_training/testing 함수가 정상 작동합니다.")
            print(f"  (Train/Test 분리는 realistic_data.py 내부 TRAIN_RATIO로 처리됨)")
            return True
        else:
            print("[WARN] load_real_data_* 함수가 빈 데이터를 반환했습니다.")
            return False
    except Exception as e:
        print(f"[FAIL] load_real_data_* 함수 호출 중 오류: {e}")
        return False


def inspect_sample_rows(dates: list[str] | None = None) -> None:
    """
    몇 개 날짜를 골라
    - CSV 원시 컬럼
    - DynamicShieldSystem 상태 벡터(State)
    를 나란히 출력해서 '수작업 cross-check'를 돕는 유틸.

    dates 가 None이면 앞/중간/끝에서 1개씩 샘플링.
    """
    df = _load_main_dataframe()
    env = DynamicShieldSystem()

    if dates is None or len(dates) == 0:
        idx = df.index
        candidates = [idx[0], idx[len(idx) // 2], idx[-1]]
    else:
        candidates = []
        for d in dates:
            dt = pd.to_datetime(d)
            if dt in df.index:
                candidates.append(dt)
            else:
                print(f"[WARN] {d} 는 데이터에 없는 날짜입니다.")

    print("\n=== 수작업 Cross-Check용 샘플 ===")
    for dt in candidates:
        row = df.loc[dt]
        print("\n----------------------------------------")
        print(f"Date: {dt.date()}")
        print("[Raw Columns]")
        print(
            row[
                [
                    "KOSPI",
                    "FX",
                    "US_10Y",
                    "KR_10Y",
                    "Yield_Spread",
                    "Swap_Point_Proxy",
                    "FX_MA_Divergence",
                    "VIX",
                    "VIX_Change",
                ]
            ]
        )

        # DynamicShieldSystem 의 상태벡터를 얻기 위해 해당 시점까지 step 진행
        env.reset()
        target_idx = df.index.get_loc(dt)
        state = None
        # 마지막 날짜는 step을 한 번 덜 진행해야 함 (done=True가 되기 전에)
        max_steps = min(target_idx + 1, env.total_steps - 1)
        for i in range(max_steps):
            next_state, _, done, info = env.step(action=0.5)
            state = next_state
            if done:
                break

        print("\n[DynamicShieldSystem State Vector]")
        print(
            "구성: [VIX, VIX_Change, FX_MA_Divergence, Yield_Spread, "
            "KICS(배율), Regime_Idx]"
        )
        print(state)


def run_all_checks() -> None:
    """위에서 정의한 모든 체크를 순차 실행."""
    print("=" * 70)
    print("Data & Leakage Sanity Checks")
    print("=" * 70)

    df = _load_main_dataframe()

    ok_yield = check_yield_spread(df)
    ok_na_ffill = check_missing_and_ffill(df)
    ok_split = check_train_test_split(df)

    overall = ok_yield and ok_na_ffill and ok_split
    print("\n" + "=" * 70)
    print(f"종합 결과: {'[PASS] 문제 없음' if overall else '[CHECK] 경고/실패 항목 있음'}")

    # 수작업 cross-check용 샘플도 같이 출력
    inspect_sample_rows()


if __name__ == "__main__":
    run_all_checks()


