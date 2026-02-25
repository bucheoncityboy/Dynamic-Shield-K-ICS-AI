"""
지연시간 측정 유틸리티 (Latency Monitor)
=========================================
시스템 실시간성 검증을 위한 지연시간 측정 도구

제안서 연관:
- "완전한 실시간성(Real-time Availability)"의 정량적 증거 제공
- Surrogate 추론 < 10ms, 전체 파이프라인 < 50ms 목표
"""

import time
import functools
import statistics
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LatencyStats:
    """지연시간 통계"""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    samples: List[float] = field(default_factory=list)
    max_samples: int = 1000  # 메모리 제한
    
    def add(self, latency_ms: float):
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        
        # 샘플 저장 (순환 버퍼)
        if len(self.samples) >= self.max_samples:
            self.samples.pop(0)
        self.samples.append(latency_ms)
    
    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0
    
    @property
    def p50_ms(self) -> float:
        """중앙값 (50th percentile)"""
        if not self.samples:
            return 0.0
        return statistics.median(self.samples)
    
    @property
    def p95_ms(self) -> float:
        """95th percentile"""
        if len(self.samples) < 2:
            return self.max_ms
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx]
    
    @property
    def p99_ms(self) -> float:
        """99th percentile"""
        if len(self.samples) < 2:
            return self.max_ms
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[idx]
    
    @property
    def std_ms(self) -> float:
        """표준편차"""
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples)


class LatencyMonitor:
    """
    지연시간 측정 모니터
    
    사용법:
    1. 데코레이터: @monitor.measure("function_name")
    2. 컨텍스트: with monitor.measure_context("block_name"):
    3. 수동: monitor.start("name") ... monitor.stop("name")
    """
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Args:
            thresholds: 경고 임계값 (ms). 예: {"surrogate": 10, "ppo": 50}
        """
        self.stats: Dict[str, LatencyStats] = defaultdict(LatencyStats)
        self.active_timers: Dict[str, float] = {}
        self.thresholds = thresholds or {
            "default": 50.0,
            "surrogate": 10.0,
            "ppo_predict": 10.0,
            "safety_layer": 5.0,
            "total_pipeline": 50.0
        }
        self.warnings: List[str] = []
    
    def measure(self, name: str = None):
        """
        함수 지연시간 측정 데코레이터
        
        Usage:
            @monitor.measure("my_function")
            def my_function():
                ...
        """
        def decorator(func: Callable) -> Callable:
            func_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self._record(func_name, elapsed_ms)
            
            return wrapper
        return decorator
    
    def measure_context(self, name: str):
        """
        컨텍스트 매니저로 지연시간 측정
        
        Usage:
            with monitor.measure_context("my_block"):
                ...
        """
        return _LatencyContext(self, name)
    
    def start(self, name: str):
        """수동 측정 시작"""
        self.active_timers[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """수동 측정 종료, 경과 시간(ms) 반환"""
        if name not in self.active_timers:
            return 0.0
        
        start = self.active_timers.pop(name)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._record(name, elapsed_ms)
        return elapsed_ms
    
    def _record(self, name: str, elapsed_ms: float):
        """측정값 기록 및 임계값 체크"""
        self.stats[name].add(elapsed_ms)
        
        # 임계값 체크
        threshold = self.thresholds.get(name, self.thresholds.get("default", 50.0))
        if elapsed_ms > threshold:
            warning = f"[LATENCY WARNING] {name}: {elapsed_ms:.2f}ms > {threshold:.0f}ms"
            self.warnings.append(warning)
            print(warning)
    
    def get_stats(self, name: str) -> Optional[LatencyStats]:
        """특정 항목의 통계 반환"""
        return self.stats.get(name)
    
    def report(self, show_all: bool = True) -> Dict[str, Dict[str, float]]:
        """
        전체 통계 리포트 생성
        
        Returns:
            {name: {mean, p50, p95, p99, min, max, count}}
        """
        report = {}
        
        for name, stats in self.stats.items():
            if stats.count == 0:
                continue
            
            threshold = self.thresholds.get(name, self.thresholds.get("default", 50.0))
            is_ok = stats.p95_ms <= threshold
            
            report[name] = {
                'count': stats.count,
                'mean_ms': round(stats.mean_ms, 3),
                'p50_ms': round(stats.p50_ms, 3),
                'p95_ms': round(stats.p95_ms, 3),
                'p99_ms': round(stats.p99_ms, 3),
                'min_ms': round(stats.min_ms, 3),
                'max_ms': round(stats.max_ms, 3),
                'threshold_ms': threshold,
                'status': '✓ PASS' if is_ok else '✗ FAIL'
            }
        
        return report
    
    def print_report(self):
        """리포트 출력"""
        print("\n" + "=" * 70)
        print("Latency Report")
        print("=" * 70)
        
        report = self.report()
        
        if not report:
            print("  (측정 데이터 없음)")
            return
        
        # 헤더
        print(f"{'Name':<25} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8} {'Status':>10}")
        print("-" * 70)
        
        for name, stats in report.items():
            print(f"{name:<25} {stats['mean_ms']:>7.2f}ms {stats['p50_ms']:>7.2f}ms "
                  f"{stats['p95_ms']:>7.2f}ms {stats['p99_ms']:>7.2f}ms "
                  f"{stats['max_ms']:>7.2f}ms {stats['status']:>10}")
        
        print("-" * 70)
        
        # 경고 요약
        if self.warnings:
            print(f"\n⚠️  총 {len(self.warnings)}회 임계값 초과 발생")
    
    def reset(self):
        """모든 통계 초기화"""
        self.stats.clear()
        self.active_timers.clear()
        self.warnings.clear()


class _LatencyContext:
    """컨텍스트 매니저 헬퍼"""
    
    def __init__(self, monitor: LatencyMonitor, name: str):
        self.monitor = monitor
        self.name = name
    
    def __enter__(self):
        self.monitor.start(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop(self.name)
        return False


# === 전역 모니터 인스턴스 (편의용) ===
_global_monitor = LatencyMonitor()

def get_monitor() -> LatencyMonitor:
    """전역 모니터 반환"""
    return _global_monitor


# === 테스트 코드 ===
if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("Latency Monitor 테스트")
    print("=" * 60)
    
    monitor = LatencyMonitor()
    
    # 1. 데코레이터 테스트
    @monitor.measure("test_function")
    def simulate_work(duration_ms: float):
        time.sleep(duration_ms / 1000)
        return duration_ms
    
    print("\n[데코레이터 테스트]")
    for _ in range(10):
        simulate_work(np.random.uniform(1, 5))
    
    # 2. 컨텍스트 테스트
    print("\n[컨텍스트 테스트]")
    for _ in range(10):
        with monitor.measure_context("context_block"):
            time.sleep(np.random.uniform(0.002, 0.008))
    
    # 3. 수동 측정 테스트
    print("\n[수동 측정 테스트]")
    for _ in range(10):
        monitor.start("manual_timer")
        time.sleep(np.random.uniform(0.001, 0.003))
        monitor.stop("manual_timer")
    
    # 4. 임계값 초과 테스트
    print("\n[임계값 초과 테스트]")
    with monitor.measure_context("slow_operation"):
        time.sleep(0.1)  # 100ms - 기본 50ms 초과
    
    # 5. 리포트 출력
    monitor.print_report()
    
    print("\n[테스트 완료] ✓")
