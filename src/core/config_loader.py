"""
Config Loader - YAML 설정 파일 로더
===================================
설정 파일을 로드하고 검증하는 유틸리티 모듈

사용 예:
    from config_loader import ConfigLoader
    
    loader = ConfigLoader()
    config = loader.load_base_config()
    timegan_epochs = config['timegan']['training']['epochs']
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    YAML 설정 파일 로더
    
    사용 예:
        loader = ConfigLoader()
        config = loader.load_base_config()
        timegan_epochs = config['timegan']['training']['epochs']
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: 설정 파일 디렉토리 경로 (기본값: 프로젝트 루트/config)
        """
        if config_dir is None:
            # 프로젝트 루트 자동 탐색
            script_dir = Path(__file__).parent  # src/core/
            project_root = script_dir.parent.parent  # 프로젝트 루트
            config_dir = project_root / 'config'
        
        self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Config 디렉토리가 없습니다: {self.config_dir}\n"
                f"다음 명령으로 생성하세요: mkdir -p {self.config_dir}"
            )
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        YAML 파일 로드
        
        Args:
            filename: YAML 파일명 (예: 'base_config.yaml')
        
        Returns:
            설정 딕셔너리
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"설정 파일이 없습니다: {filepath}\n"
                f"다음 파일을 생성하세요: {filepath}"
            )
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def load_base_config(self) -> Dict[str, Any]:
        """
        base_config.yaml 로드
        
        Returns:
            기본 설정 딕셔너리
        """
        return self.load_yaml('base_config.yaml')
    
    def load_scenarios(self) -> Dict[str, Any]:
        """
        scenarios.yaml 로드
        
        Returns:
            시나리오 설정 딕셔너리
        """
        return self.load_yaml('scenarios.yaml')
    
    def get_timegan_config(self) -> Dict[str, Any]:
        """
        TimeGAN 설정만 추출
        
        Returns:
            TimeGAN 설정 딕셔너리
        """
        config = self.load_base_config()
        return config.get('timegan', {})
    
    def get_ppo_config(self) -> Dict[str, Any]:
        """
        PPO 설정만 추출
        
        Returns:
            PPO 설정 딕셔너리
        """
        config = self.load_base_config()
        return config.get('ppo', {})
    
    def get_kics_config(self) -> Dict[str, Any]:
        """
        K-ICS 설정만 추출
        
        Returns:
            K-ICS 설정 딕셔너리
        """
        config = self.load_base_config()
        return config.get('kics', {})
    
    def get_gym_env_config(self) -> Dict[str, Any]:
        """
        Gym Environment 설정만 추출
        
        Returns:
            Gym Environment 설정 딕셔너리
        """
        config = self.load_base_config()
        return config.get('gym_env', {})
    
    def get_agent_config(self) -> Dict[str, Any]:
        """
        Agent 설정만 추출
        
        Returns:
            Agent 설정 딕셔너리
        """
        config = self.load_base_config()
        return config.get('agent', {})
    
    def get_paths(self) -> Dict[str, str]:
        """
        경로 설정만 추출 (상대 경로를 절대 경로로 변환)
        
        Returns:
            경로 설정 딕셔너리 (절대 경로)
        """
        config = self.load_base_config()
        paths = config.get('paths', {})
        
        # 상대 경로를 절대 경로로 변환
        script_dir = Path(__file__).parent  # src/core/
        project_root = script_dir.parent.parent  # 프로젝트 루트
        
        absolute_paths = {}
        for key, value in paths.items():
            if isinstance(value, str) and not os.path.isabs(value):
                # 상대 경로인 경우 프로젝트 루트 기준으로 변환
                absolute_paths[key] = str(project_root / value)
            else:
                absolute_paths[key] = value
        
        return absolute_paths


# 전역 인스턴스 (선택적)
_default_loader = None

def get_config_loader() -> ConfigLoader:
    """전역 ConfigLoader 인스턴스 반환"""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader


# 편의 함수
def load_base_config() -> Dict[str, Any]:
    """기본 설정 로드 (편의 함수)"""
    return get_config_loader().load_base_config()

def load_scenarios() -> Dict[str, Any]:
    """시나리오 설정 로드 (편의 함수)"""
    return get_config_loader().load_scenarios()

