import numpy as np

class KICSCalculator:
    """
    K-ICS (Korean Insurance Capital Standard) 비율 계산 엔진 (Real-time)
    
    [기능]
    1. 시장 데이터(row)와 헤지 비율(Action)을 받아 포트폴리오 가치를 갱신
    2. 자산(Assets)과 부채(Liabilities)의 듀레이션 갭에 따른 자본 변동 계산
    3. 규제 충격 시나리오를 적용해 요구자본(SCR) 및 K-ICS 비율 산출
    """
    def __init__(self, initial_assets=None, initial_liabilities=None):
        # Config 로드 시도
        try:
            from config_loader import ConfigLoader
            loader = ConfigLoader()
            kics_config = loader.get_kics_config()
            
            # 설정 파일에서 기본값 로드
            self.initial_assets = initial_assets or kics_config.get('initial_assets', 10000.0)
            self.initial_liabilities = initial_liabilities or kics_config.get('initial_liabilities', 9000.0)
            
            # 포트폴리오 비중
            weights = kics_config.get('portfolio_weights', {})
            self.w_equity = weights.get('equity', 0.3)
            self.w_bond = weights.get('bond', 0.5)
            self.w_fx = weights.get('fx', 0.2)
            
            # 듀레이션
            duration = kics_config.get('duration', {})
            self.dur_asset = duration.get('asset', 8.0)
            self.dur_liab = duration.get('liability', 10.0)
            
            # 규제 충격 시나리오
            stress = kics_config.get('stress_scenarios', {})
            self.equity_shock = stress.get('equity_shock', 0.30)
            self.fx_shock = stress.get('fx_shock', 0.10)
            self.rate_shock = stress.get('rate_shock', 0.01)
        except (ImportError, FileNotFoundError, KeyError):
            # 폴백: 기본값 사용
            self.initial_assets = initial_assets or 10000.0
            self.initial_liabilities = initial_liabilities or 9000.0
            self.w_equity = 0.3
            self.w_bond = 0.5
            self.w_fx = 0.2
            self.dur_asset = 8.0
            self.dur_liab = 10.0
            self.equity_shock = 0.30
            self.fx_shock = 0.10
            self.rate_shock = 0.01
        
        # 현재 상태 (매일 갱신됨)
        self.assets = self.initial_assets
        self.liabilities = self.initial_liabilities   
        
    def reset(self):
        """시뮬레이션 초기화 시 호출"""
        self.assets = self.initial_assets
        self.liabilities = self.initial_liabilities

    def update_and_calculate(self, row, prev_row, hedge_ratio):
        """
        매일(Step) 호출되어 자산/부채를 갱신하고 K-ICS 비율을 반환
        """
        if prev_row is None:
            # 첫날은 변동 없음, 초기 비율 반환
            return self._compute_ratio(hedge_ratio)
            
        # 1. 시장 변동폭 계산
        # (1) 주식 수익률
        if prev_row['KOSPI'] > 0:
            ret_equity = (row['KOSPI'] - prev_row['KOSPI']) / prev_row['KOSPI']
        else:
            ret_equity = 0.0
            
        # (2) 환율 수익률
        if prev_row['FX'] > 0:
            ret_fx = (row['FX'] - prev_row['FX']) / prev_row['FX']
        else:
            ret_fx = 0.0
            
        # (3) 금리 변동 (단위: %) -> 가격 변동 근사: -Duration * Delta_Yield
        # 미국채 10년물(US_10Y) 변동분을 벤치마크로 사용
        delta_yield = (row['US_10Y'] - prev_row['US_10Y']) * 0.01 
        
        # (4) 헤지 비용 (Minus Carry)
        # Swap_Point_Proxy는 연율화된 스왑 포인트 대용치라고 가정하고 일할 계산
        daily_hedge_cost_rate = (row.get('Swap_Point_Proxy', 0.0) / 100) / 365 
        
        # 2. 자산 가치 갱신 (Mark-to-Market)
        # (1) 주식 자산 변동
        val_equity = self.assets * self.w_equity * (1 + ret_equity)
        
        # (2) 채권 자산 변동 (금리 민감도 적용)
        val_bond = self.assets * self.w_bond * (1 - self.dur_asset * delta_yield)
        
        # (3) 외화 자산 변동 (헤지 반영)
        # - Open Position (헤지 안 함): 환율 변동에 노출
        # - Hedged Position (헤지 함): 환율 변동 없음, 대신 헤지 비용 지불
        open_ratio = 1.0 - hedge_ratio
        val_fx_open = self.assets * self.w_fx * open_ratio * (1 + ret_fx)
        val_fx_hedged = self.assets * self.w_fx * hedge_ratio * (1 - daily_hedge_cost_rate)
        
        # 총 자산 업데이트
        new_assets = val_equity + val_bond + val_fx_open + val_fx_hedged
        
        # 3. 부채 가치 갱신 (금리 연동)
        new_liabilities = self.liabilities * (1 - self.dur_liab * delta_yield)
        
        # 상태 업데이트
        self.assets = new_assets
        self.liabilities = new_liabilities
        
        # 4. K-ICS 비율 산출
        return self._compute_ratio(hedge_ratio)
        
    def _compute_ratio(self, hedge_ratio):
        """현재 자산/부채 상태에서 K-ICS 비율 계산 (규제 충격 시나리오 적용)"""
        available_capital = self.assets - self.liabilities
        
        if available_capital <= 0:
            return 0.0 # 자본 잠식 (최악의 상황)
            
        # 요구자본(SCR) 계산 (간이 모델)
        # K-ICS 규제 충격 시나리오 (Config에서 로드)
        
        # (1) 주식 리스크 (충격)
        risk_equity = self.assets * self.w_equity * self.equity_shock
        
        # (2) 외환 리스크 (헤지된 부분은 리스크 0으로 간주)
        risk_fx = self.assets * self.w_fx * (1 - hedge_ratio) * self.fx_shock
        
        # (3) 금리 리스크 (듀레이션 갭만큼 노출)
        gap = self.dur_liab - self.dur_asset
        risk_rate = self.assets * self.w_bond * gap * self.rate_shock 
        
        # 통합 리스크 (단순 합산 - 보수적 관점)
        total_risk = risk_equity + risk_fx + risk_rate
        
        if total_risk <= 0: return 999.0 # 리스크 없음 (매우 안전)
        
        kics_ratio = (available_capital / total_risk) * 100
        return kics_ratio


class RatioKICSEngine:
    """
    K-ICS SCR Ratio 계산 엔진 (Batch 연산용)
    
    kics_surrogate.py의 AI Surrogate 학습을 위한 인터페이스.
    헤지 비율과 상관관계를 입력받아 SCR 비율을 배치로 계산.
    """
    def __init__(self, initial_assets=None, initial_liabilities=None):
        # Config 로드 시도 (KICSCalculator와 동일한 로직)
        try:
            from config_loader import ConfigLoader
            loader = ConfigLoader()
            kics_config = loader.get_kics_config()
            
            # 설정 파일에서 기본값 로드
            self.initial_assets = initial_assets or kics_config.get('initial_assets', 10000.0)
            self.initial_liabilities = initial_liabilities or kics_config.get('initial_liabilities', 9000.0)
            
            # 포트폴리오 비중
            weights = kics_config.get('portfolio_weights', {})
            self.w_equity = weights.get('equity', 0.3)
            self.w_bond = weights.get('bond', 0.5)
            self.w_fx = weights.get('fx', 0.2)
            
            # 듀레이션
            duration = kics_config.get('duration', {})
            self.dur_asset = duration.get('asset', 8.0)
            self.dur_liab = duration.get('liability', 10.0)
            
            # 규제 충격 시나리오
            stress = kics_config.get('stress_scenarios', {})
            self.equity_shock = stress.get('equity_shock', 0.30)
            self.fx_shock = stress.get('fx_shock', 0.10)
            self.rate_shock = stress.get('rate_shock', 0.01)
        except (ImportError, FileNotFoundError, KeyError):
            # 폴백: 기본값 사용
            self.initial_assets = initial_assets or 10000.0
            self.initial_liabilities = initial_liabilities or 9000.0
            self.w_equity = 0.3
            self.w_bond = 0.5
            self.w_fx = 0.2
            self.dur_asset = 8.0
            self.dur_liab = 10.0
            self.equity_shock = 0.30
            self.fx_shock = 0.10
            self.rate_shock = 0.01
        
    def calculate_scr_ratio_batch(self, hedge_ratios, correlations):
        """
        배치 단위로 SCR 비율 계산 (K-ICS 표준모형 방식: 제곱근 합산)
        
        Args:
            hedge_ratios: 헤지 비율 배열 (0~1)
            correlations: 시장 상관관계 배열 (-1~1, 주식-환율 상관계수)
            
        Returns:
            SCR 비율 배열 (0~1 범위로 정규화)
        """
        hedge_ratios = np.asarray(hedge_ratios)
        correlations = np.asarray(correlations)
        
        available_capital = self.initial_assets - self.initial_liabilities
        
        # 주식 리스크 (충격)
        risk_equity = self.initial_assets * self.w_equity * self.equity_shock
        
        # 외환 리스크 (헤지된 부분은 리스크 감소)
        risk_fx = self.initial_assets * self.w_fx * (1 - hedge_ratios) * self.fx_shock
        
        # 금리 리스크 (듀레이션 갭 기반) - 독립적으로 합산
        dur_gap = self.dur_liab - self.dur_asset
        risk_rate = self.initial_assets * self.w_bond * dur_gap * self.rate_shock
        
        # ========================================
        # K-ICS 표준모형: 제곱근 합산 (Diversification Effect)
        # Risk_market = sqrt(R_eq² + R_fx² + 2*ρ*R_eq*R_fx)
        # 
        # ρ < 0 (음의 상관): 분산 효과로 위험 감소
        # ρ > 0 (양의 상관): 위험 집중으로 위험 증가
        # ========================================
        market_risk_sq = (risk_equity ** 2) + (risk_fx ** 2) + (2 * correlations * risk_equity * risk_fx)
        market_risk = np.sqrt(np.maximum(market_risk_sq, 0))  # 음수 방지
        
        # 총 리스크 = 시장 리스크 + 금리 리스크 (보수적 합산)
        total_risk = market_risk + risk_rate
        total_risk = np.maximum(total_risk, 1e-6)  # 0 방지
        
        # SCR 비율 계산 (가용자본 / 요구자본)
        # 높을수록 안전 (1.0 = 100%, 규제 최소선)
        scr_ratios = available_capital / total_risk
        
        # 0~1 범위로 정규화 (PPO 학습 호환)
        scr_ratios = np.clip(scr_ratios, 0, 10)  # 상한 1000%
        scr_ratios = scr_ratios / 10  # 0~1로 정규화
        
        return scr_ratios