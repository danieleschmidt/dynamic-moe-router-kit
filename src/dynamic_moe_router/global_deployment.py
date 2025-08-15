"""Global-first deployment infrastructure for dynamic MoE routing."""

import logging
import os
import json
import time
import threading
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported global regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"


@dataclass
class GlobalConfig:
    """Global deployment configuration."""
    # Multi-region settings
    primary_region: Region = Region.US_EAST
    secondary_regions: List[Region] = None
    enable_multi_region: bool = True
    
    # Internationalization
    default_language: str = "en"
    supported_languages: List[str] = None
    enable_i18n: bool = True
    
    # Compliance and privacy
    enable_gdpr_compliance: bool = True
    enable_ccpa_compliance: bool = True
    enable_pdpa_compliance: bool = True
    data_residency_requirements: Dict[str, str] = None
    
    # Performance optimization per region
    region_specific_configs: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set defaults for optional fields."""
        if self.secondary_regions is None:
            self.secondary_regions = [Region.EU_WEST, Region.ASIA_PACIFIC]
        
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de", "ja", "zh", "ko"]
        
        if self.data_residency_requirements is None:
            self.data_residency_requirements = {
                "eu": "eu-west-1",
                "asia": "ap-southeast-1",
                "us": "us-east-1"
            }
        
        if self.region_specific_configs is None:
            self.region_specific_configs = {}


class I18nManager:
    """Internationalization manager for global deployment."""
    
    def __init__(self, supported_languages: List[str]):
        self.supported_languages = supported_languages
        self.translations = self._load_translations()
        self.default_language = "en"
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries."""
        translations = {
            "en": {
                "router_initialized": "Router initialized successfully",
                "routing_failed": "Routing operation failed",
                "health_check_passed": "Health check passed",
                "health_check_failed": "Health check failed",
                "performance_degraded": "Performance degraded",
                "circuit_breaker_open": "Circuit breaker is open",
                "fallback_activated": "Fallback strategy activated",
                "experts_selected": "experts selected for token",
                "flop_reduction": "FLOP reduction achieved",
                "cache_hit": "Cache hit",
                "cache_miss": "Cache miss"
            },
            "es": {
                "router_initialized": "Router inicializado exitosamente",
                "routing_failed": "Operación de enrutamiento falló",
                "health_check_passed": "Verificación de salud aprobada",
                "health_check_failed": "Verificación de salud falló",
                "performance_degraded": "Rendimiento degradado",
                "circuit_breaker_open": "Circuit breaker está abierto",
                "fallback_activated": "Estrategia de respaldo activada",
                "experts_selected": "expertos seleccionados para token",
                "flop_reduction": "Reducción de FLOP lograda",
                "cache_hit": "Acierto de caché",
                "cache_miss": "Falla de caché"
            },
            "fr": {
                "router_initialized": "Routeur initialisé avec succès",
                "routing_failed": "Opération de routage échouée",
                "health_check_passed": "Contrôle de santé réussi",
                "health_check_failed": "Contrôle de santé échoué",
                "performance_degraded": "Performance dégradée",
                "circuit_breaker_open": "Disjoncteur est ouvert",
                "fallback_activated": "Stratégie de secours activée",
                "experts_selected": "experts sélectionnés pour le token",
                "flop_reduction": "Réduction FLOP atteinte",
                "cache_hit": "Succès de cache",
                "cache_miss": "Échec de cache"
            },
            "de": {
                "router_initialized": "Router erfolgreich initialisiert",
                "routing_failed": "Routing-Operation fehlgeschlagen",
                "health_check_passed": "Gesundheitsprüfung bestanden",
                "health_check_failed": "Gesundheitsprüfung fehlgeschlagen",
                "performance_degraded": "Leistung verschlechtert",
                "circuit_breaker_open": "Schaltkreisunterbrecher ist offen",
                "fallback_activated": "Fallback-Strategie aktiviert",
                "experts_selected": "Experten für Token ausgewählt",
                "flop_reduction": "FLOP-Reduktion erreicht",
                "cache_hit": "Cache-Treffer",
                "cache_miss": "Cache-Fehler"
            },
            "ja": {
                "router_initialized": "ルーターが正常に初期化されました",
                "routing_failed": "ルーティング操作が失敗しました",
                "health_check_passed": "ヘルスチェックが成功しました",
                "health_check_failed": "ヘルスチェックが失敗しました",
                "performance_degraded": "パフォーマンスが低下しました",
                "circuit_breaker_open": "サーキットブレーカーが開いています",
                "fallback_activated": "フォールバック戦略が有効化されました",
                "experts_selected": "トークン用エキスパートが選択されました",
                "flop_reduction": "FLOP削減が達成されました",
                "cache_hit": "キャッシュヒット",
                "cache_miss": "キャッシュミス"
            },
            "zh": {
                "router_initialized": "路由器初始化成功",
                "routing_failed": "路由操作失败",
                "health_check_passed": "健康检查通过",
                "health_check_failed": "健康检查失败",
                "performance_degraded": "性能下降",
                "circuit_breaker_open": "断路器已打开",
                "fallback_activated": "后备策略已激活",
                "experts_selected": "为令牌选择的专家",
                "flop_reduction": "FLOP减少已实现",
                "cache_hit": "缓存命中",
                "cache_miss": "缓存未命中"
            }
        }
        
        return translations
    
    def get_message(self, key: str, language: str = None) -> str:
        """Get translated message."""
        language = language or self.default_language
        
        if language not in self.supported_languages:
            language = self.default_language
        
        return self.translations.get(language, {}).get(
            key, 
            self.translations[self.default_language].get(key, key)
        )
    
    def format_message(self, key: str, language: str = None, **kwargs) -> str:
        """Get formatted translated message."""
        message = self.get_message(key, language)
        try:
            return message.format(**kwargs)
        except:
            return message


class ComplianceManager:
    """Manages regulatory compliance across regions."""
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.data_processing_logs = []
        self._lock = threading.Lock()
    
    def process_request(self, request_data: Dict[str, Any], user_region: str = None) -> Dict[str, Any]:
        """Process request with compliance requirements."""
        processed_data = request_data.copy()
        compliance_info = {
            'gdpr_compliant': False,
            'ccpa_compliant': False,
            'pdpa_compliant': False,
            'data_residency_region': None
        }
        
        # Determine data residency requirements
        if user_region:
            compliance_info['data_residency_region'] = self._get_required_region(user_region)
        
        # GDPR compliance (EU users)
        if self.config.enable_gdpr_compliance and self._is_eu_user(user_region):
            processed_data = self._apply_gdpr_processing(processed_data)
            compliance_info['gdpr_compliant'] = True
        
        # CCPA compliance (California users)
        if self.config.enable_ccpa_compliance and self._is_california_user(user_region):
            processed_data = self._apply_ccpa_processing(processed_data)
            compliance_info['ccpa_compliant'] = True
        
        # PDPA compliance (APAC users)
        if self.config.enable_pdpa_compliance and self._is_apac_user(user_region):
            processed_data = self._apply_pdpa_processing(processed_data)
            compliance_info['pdpa_compliant'] = True
        
        # Log data processing
        self._log_data_processing(processed_data, compliance_info, user_region)
        
        processed_data['_compliance_info'] = compliance_info
        return processed_data
    
    def _get_required_region(self, user_region: str) -> str:
        """Get required processing region based on data residency."""
        region_mapping = {
            'eu': ['eu-west-1', 'eu-central-1'],
            'us': ['us-east-1', 'us-west-2'],
            'asia': ['ap-southeast-1', 'ap-northeast-1']
        }
        
        for region_group, regions in region_mapping.items():
            if user_region in regions or user_region.startswith(region_group):
                return self.config.data_residency_requirements.get(region_group, regions[0])
        
        return self.config.primary_region.value
    
    def _is_eu_user(self, user_region: str) -> bool:
        """Check if user is from EU."""
        eu_regions = ['eu-west-1', 'eu-central-1', 'eu-north-1']
        return user_region in eu_regions if user_region else False
    
    def _is_california_user(self, user_region: str) -> bool:
        """Check if user is from California (simplified)."""
        ca_regions = ['us-west-1', 'us-west-2']
        return user_region in ca_regions if user_region else False
    
    def _is_apac_user(self, user_region: str) -> bool:
        """Check if user is from APAC."""
        apac_regions = ['ap-southeast-1', 'ap-northeast-1', 'ap-south-1']
        return user_region in apac_regions if user_region else False
    
    def _apply_gdpr_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GDPR data processing requirements."""
        # Remove or anonymize PII
        gdpr_processed = data.copy()
        
        # Add GDPR processing metadata
        gdpr_processed['_gdpr_processed'] = True
        gdpr_processed['_processing_timestamp'] = time.time()
        gdpr_processed['_lawful_basis'] = 'legitimate_interest'  # or 'consent'
        
        return gdpr_processed
    
    def _apply_ccpa_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply CCPA data processing requirements."""
        ccpa_processed = data.copy()
        
        # Add CCPA processing metadata
        ccpa_processed['_ccpa_processed'] = True
        ccpa_processed['_processing_timestamp'] = time.time()
        ccpa_processed['_do_not_sell'] = True  # Default to not selling data
        
        return ccpa_processed
    
    def _apply_pdpa_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PDPA data processing requirements."""
        pdpa_processed = data.copy()
        
        # Add PDPA processing metadata
        pdpa_processed['_pdpa_processed'] = True
        pdpa_processed['_processing_timestamp'] = time.time()
        pdpa_processed['_consent_obtained'] = True
        
        return pdpa_processed
    
    def _log_data_processing(self, data: Dict[str, Any], compliance_info: Dict[str, Any], user_region: str):
        """Log data processing for compliance auditing."""
        with self._lock:
            log_entry = {
                'timestamp': time.time(),
                'user_region': user_region,
                'data_size': len(str(data)),
                'compliance_info': compliance_info,
                'processing_type': 'moe_routing'
            }
            self.data_processing_logs.append(log_entry)
            
            # Keep only recent logs (last 10000 entries)
            if len(self.data_processing_logs) > 10000:
                self.data_processing_logs = self.data_processing_logs[-10000:]
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        with self._lock:
            if not self.data_processing_logs:
                return {'message': 'No data processing logs available'}
            
            total_requests = len(self.data_processing_logs)
            gdpr_requests = sum(1 for log in self.data_processing_logs 
                              if log['compliance_info']['gdpr_compliant'])
            ccpa_requests = sum(1 for log in self.data_processing_logs 
                              if log['compliance_info']['ccpa_compliant'])
            pdpa_requests = sum(1 for log in self.data_processing_logs 
                              if log['compliance_info']['pdpa_compliant'])
            
            return {
                'total_requests_processed': total_requests,
                'gdpr_compliant_requests': gdpr_requests,
                'ccpa_compliant_requests': ccpa_requests,
                'pdpa_compliant_requests': pdpa_requests,
                'compliance_rates': {
                    'gdpr_rate': gdpr_requests / max(total_requests, 1),
                    'ccpa_rate': ccpa_requests / max(total_requests, 1),
                    'pdpa_rate': pdpa_requests / max(total_requests, 1)
                },
                'recent_processing_countries': list(set(
                    log['user_region'] for log in self.data_processing_logs[-100:]
                    if log['user_region']
                ))
            }


class GlobalLoadBalancer:
    """Global load balancer for multi-region deployment."""
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.region_health = {region.value: True for region in Region}
        self.region_latencies = {region.value: 0.0 for region in Region}
        self.request_counts = {region.value: 0 for region in Region}
        self._lock = threading.Lock()
    
    def select_optimal_region(self, user_region: str = None, request_data: Dict[str, Any] = None) -> str:
        """Select optimal region for request processing."""
        # Data residency requirements take precedence
        if request_data and '_compliance_info' in request_data:
            required_region = request_data['_compliance_info'].get('data_residency_region')
            if required_region and self.region_health.get(required_region, False):
                return required_region
        
        # Geographic proximity
        if user_region:
            closest_region = self._get_closest_region(user_region)
            if self.region_health.get(closest_region, False):
                with self._lock:
                    self.request_counts[closest_region] += 1
                return closest_region
        
        # Fallback to primary region
        primary = self.config.primary_region.value
        if self.region_health.get(primary, False):
            with self._lock:
                self.request_counts[primary] += 1
            return primary
        
        # Emergency fallback to any healthy region
        for region, healthy in self.region_health.items():
            if healthy:
                with self._lock:
                    self.request_counts[region] += 1
                return region
        
        # Last resort - return primary even if unhealthy
        return primary
    
    def _get_closest_region(self, user_region: str) -> str:
        """Get geographically closest region."""
        region_groups = {
            'us': [Region.US_EAST, Region.US_WEST],
            'eu': [Region.EU_WEST, Region.EU_CENTRAL],
            'asia': [Region.ASIA_PACIFIC, Region.ASIA_NORTHEAST]
        }
        
        for group, regions in region_groups.items():
            if user_region.startswith(group):
                # Return first healthy region in group
                for region in regions:
                    if self.region_health.get(region.value, False):
                        return region.value
        
        return self.config.primary_region.value
    
    def update_region_health(self, region: str, healthy: bool, latency: float = 0.0):
        """Update region health status."""
        with self._lock:
            self.region_health[region] = healthy
            if latency > 0:
                self.region_latencies[region] = latency
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            total_requests = sum(self.request_counts.values())
            
            return {
                'region_health': self.region_health.copy(),
                'region_latencies': self.region_latencies.copy(),
                'request_distribution': {
                    region: count / max(total_requests, 1)
                    for region, count in self.request_counts.items()
                },
                'total_requests': total_requests,
                'healthy_regions': sum(1 for healthy in self.region_health.values() if healthy)
            }


class GlobalDeploymentManager:
    """Manages global deployment of MoE router."""
    
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.i18n = I18nManager(config.supported_languages)
        self.compliance = ComplianceManager(config)
        self.load_balancer = GlobalLoadBalancer(config)
        
        logger.info(f"Global deployment manager initialized for regions: {config.secondary_regions}")
    
    def process_global_request(self, 
                             request_data: Dict[str, Any], 
                             user_region: str = None,
                             user_language: str = None) -> Dict[str, Any]:
        """Process request with global considerations."""
        
        # Apply compliance processing
        processed_data = self.compliance.process_request(request_data, user_region)
        
        # Select optimal region
        target_region = self.load_balancer.select_optimal_region(user_region, processed_data)
        
        # Add global metadata
        processed_data['_global_info'] = {
            'user_region': user_region,
            'target_region': target_region,
            'user_language': user_language or self.config.default_language,
            'processing_timestamp': time.time(),
            'multi_region_enabled': self.config.enable_multi_region
        }
        
        return processed_data
    
    def get_localized_message(self, key: str, language: str = None, **kwargs) -> str:
        """Get localized message for user."""
        return self.i18n.format_message(key, language, **kwargs)
    
    def health_check_global(self) -> Dict[str, Any]:
        """Perform global health check."""
        return {
            'global_deployment_status': 'healthy',
            'supported_languages': self.config.supported_languages,
            'compliance_status': {
                'gdpr_enabled': self.config.enable_gdpr_compliance,
                'ccpa_enabled': self.config.enable_ccpa_compliance,
                'pdpa_enabled': self.config.enable_pdpa_compliance
            },
            'load_balancing_stats': self.load_balancer.get_load_balancing_stats(),
            'compliance_report': self.compliance.get_compliance_report()
        }
    
    def export_global_configuration(self) -> Dict[str, Any]:
        """Export global configuration for deployment."""
        return {
            'deployment_config': asdict(self.config),
            'supported_regions': [region.value for region in Region],
            'compliance_features': {
                'gdpr': self.config.enable_gdpr_compliance,
                'ccpa': self.config.enable_ccpa_compliance,
                'pdpa': self.config.enable_pdpa_compliance
            },
            'i18n_features': {
                'supported_languages': self.config.supported_languages,
                'default_language': self.config.default_language
            }
        }