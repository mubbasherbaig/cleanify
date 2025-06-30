"""
Cleanify v2-alpha Settings
Tunables, feature flags, and configuration parameters
"""

import os
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CorridorSettings:
    """Corridor analysis configuration"""
    CORRIDOR_HALF_M: float = 250.0
    WAY_OFFSET_THRESH_M: float = 400.0
    MAX_DETOUR_KM: float = 0.3
    MAX_DETOUR_RATIO: float = 0.05  # 5%
    CANDIDATE_SCAN_RADIUS: float = 1000.0


@dataclass
class WaitingSettings:
    """Traffic waiting configuration"""
    SAFETY_PAD_MIN: float = 1.0
    MAX_WAIT_MINUTES: float = 15.0
    RUSH_HOUR_BONUS: float = 5.0
    OVERFLOW_THRESHOLD: float = 95.0


@dataclass
class TilingSettings:
    """H3 tiling configuration"""
    TILE_RES: int = 9  # ~500m hex cells
    NEIGHBOR_RINGS: int = 2
    MAX_BINS_PER_TILE: int = 50


@dataclass
class OptimizationSettings:
    """OR-Tools optimization parameters"""
    MAX_VEHICLES: int = 20
    MAX_BINS_PER_ROUTE: int = 15
    VEHICLE_CAPACITY_BUFFER: float = 0.1  # 10% safety margin
    MAX_ROUTE_DISTANCE_KM: float = 100.0
    MAX_ROUTE_DURATION_MIN: float = 480  # 8 hours
    OPTIMIZATION_TIMEOUT_SEC: int = 30


@dataclass
class AgentSettings:
    """Agent system configuration"""
    HEARTBEAT_INTERVAL_SEC: float = 5.0
    MESSAGE_TTL_SEC: int = 300
    MAX_RETRY_ATTEMPTS: int = 3
    AGENT_STARTUP_DELAY_SEC: float = 1.0
    SUPERVISION_CHECK_INTERVAL_SEC: float = 10.0


@dataclass
class RedisSettings:
    """Redis configuration"""
    HOST: str = "localhost"
    PORT: int = 6379
    DB: int = 0
    PASSWORD: Optional[str] = None
    STREAM_MAXLEN: int = 1000
    CONSUMER_GROUP: str = "cleanify_agents"
    CONSUMER_TIMEOUT_MS: int = 1000


@dataclass
class APISettings:
    """FastAPI configuration"""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    WORKERS: int = 1
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list = None


@dataclass
class MLSettings:
    """Machine learning configuration"""
    MODEL_UPDATE_INTERVAL_MIN: int = 60
    PREDICTION_CACHE_TTL_SEC: int = 300
    FEATURE_WINDOW_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 100


@dataclass
class LLMSettings:
    """LLM advisor configuration"""
    ENABLE_LLM_ADVISOR: bool = False
    MODEL_NAME: str = "gpt-3.5-turbo"  # Changed from local model
    MAX_NEW_TOKENS: int = 300  # Reduced for API efficiency
    TEMPERATURE: float = 0.7
    REQUEST_TIMEOUT_SEC: int = 30


class Settings:
    """Main settings class with environment variable support"""
    
    def __init__(self):
        # Core settings
        self.corridor = CorridorSettings()
        self.waiting = WaitingSettings()
        self.tiling = TilingSettings()
        self.optimization = OptimizationSettings()
        self.agents = AgentSettings()
        self.redis = RedisSettings()
        self.api = APISettings()
        self.ml = MLSettings()
        self.llm = LLMSettings()
        
        # Feature flags
        self.ENABLE_TILING = True
        self.ENABLE_TRAFFIC_WAITING = True
        self.ENABLE_CORRIDOR_ANALYSIS = True
        self.ENABLE_PREDICTIVE_COLLECTION = True
        self.ENABLE_DYNAMIC_THRESHOLDS = True
        self.ENABLE_ROUTE_OPTIMIZATION = True
        self.ENABLE_EMERGENCY_MONITORING = True
        self.ENABLE_METRICS_COLLECTION = True
        self.ENABLE_REDIS_PERSISTENCE = True
        
        # System constraints
        self.MAX_BINS_TOTAL = 10000
        self.MAX_TRUCKS_TOTAL = 100
        self.MAX_ACTIVE_ROUTES = 50
        self.MAX_AGENT_INSTANCES = 20
        
        # Performance tuning
        self.DECISION_INTERVAL_SEC = 10.0
        self.STATE_UPDATE_INTERVAL_SEC = 1.0
        self.METRICS_UPDATE_INTERVAL_SEC = 30.0
        self.CLEANUP_INTERVAL_SEC = 300.0
        
        # Data retention
        self.MAX_ROUTE_HISTORY = 1000
        self.MAX_EVENT_HISTORY = 5000
        self.MAX_METRIC_POINTS = 10000
        self.LOG_RETENTION_DAYS = 7
        
        # Safety limits
        self.MAX_FILL_LEVEL = 150.0  # Allow overflow for testing
        self.MIN_TRUCK_FUEL = 10.0
        self.EMERGENCY_RESPONSE_TIME_MIN = 5.0
        self.CRITICAL_BIN_THRESHOLD = 98.0
        
        # Load from environment
        self._load_from_env()
        
        # System capability detection
        self._detect_capabilities()
    
    def _load_from_env(self):
        """Load settings from environment variables"""
        
        # Redis settings
        self.redis.HOST = os.getenv("REDIS_HOST", self.redis.HOST)
        self.redis.PORT = int(os.getenv("REDIS_PORT", self.redis.PORT))
        self.redis.PASSWORD = os.getenv("REDIS_PASSWORD", self.redis.PASSWORD)
        
        # API settings
        self.api.HOST = os.getenv("API_HOST", self.api.HOST)
        self.api.PORT = int(os.getenv("API_PORT", self.api.PORT))
        self.api.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        self.api.LOG_LEVEL = os.getenv("LOG_LEVEL", self.api.LOG_LEVEL)
        
        # Feature flags from environment
        self.ENABLE_LLM_ADVISOR = os.getenv("ENABLE_LLM_ADVISOR", "false").lower() == "true"
        self.ENABLE_TILING = os.getenv("ENABLE_TILING", "true").lower() == "true"
        self.ENABLE_TRAFFIC_WAITING = os.getenv("ENABLE_TRAFFIC_WAITING", "true").lower() == "true"
        
        # Optimization settings
        self.optimization.MAX_VEHICLES = int(os.getenv("MAX_VEHICLES", self.optimization.MAX_VEHICLES))
        self.optimization.MAX_BINS_PER_ROUTE = int(os.getenv("MAX_BINS_PER_ROUTE", self.optimization.MAX_BINS_PER_ROUTE))
        self.optimization.OPTIMIZATION_TIMEOUT_SEC = int(os.getenv("OPTIMIZATION_TIMEOUT_SEC", self.optimization.OPTIMIZATION_TIMEOUT_SEC))
        
        # Corridor settings
        self.corridor.CORRIDOR_HALF_M = float(os.getenv("CORRIDOR_HALF_M", self.corridor.CORRIDOR_HALF_M))
        self.corridor.MAX_DETOUR_KM = float(os.getenv("MAX_DETOUR_KM", self.corridor.MAX_DETOUR_KM))
        
        # Performance settings
        self.DECISION_INTERVAL_SEC = float(os.getenv("DECISION_INTERVAL_SEC", self.DECISION_INTERVAL_SEC))
        self.STATE_UPDATE_INTERVAL_SEC = float(os.getenv("STATE_UPDATE_INTERVAL_SEC", self.STATE_UPDATE_INTERVAL_SEC))
    
    def _detect_capabilities(self):
        """Detect system capabilities and adjust settings"""
        
        # CPU detection for optimization
        cpu_count = psutil.cpu_count()
        if cpu_count >= 12:
            self.optimization.OPTIMIZATION_TIMEOUT_SEC = 60  # Allow longer optimization
            self.api.WORKERS = min(4, cpu_count // 3)
        elif cpu_count >= 8:
            self.optimization.OPTIMIZATION_TIMEOUT_SEC = 45
            self.api.WORKERS = 2
        else:
            self.optimization.OPTIMIZATION_TIMEOUT_SEC = 30
            self.api.WORKERS = 1
        
        # Memory detection for general system performance
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Adjust batch sizes based on available memory
        if total_memory_gb >= 32:
            self.MAX_BINS_TOTAL = 10000
            self.tiling.MAX_BINS_PER_TILE = 100
        elif total_memory_gb >= 16:
            self.MAX_BINS_TOTAL = 5000
            self.tiling.MAX_BINS_PER_TILE = 50
        else:
            self.MAX_BINS_TOTAL = 1000
            self.tiling.MAX_BINS_PER_TILE = 25
        
        # LLM Advisor is now based on API availability, not local memory
        if self.ENABLE_LLM_ADVISOR:
            print(f"✅ LLM Advisor enabled via OpenAI API")
        else:
            print(f"⚠️ LLM Advisor disabled in settings")
        
        print(f"🔧 System capabilities: {cpu_count} CPUs, {total_memory_gb:.1f}GB RAM")
        print(f"   Max bins: {self.MAX_BINS_TOTAL}, Workers: {self.api.WORKERS}")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis.PASSWORD:
            return f"redis://:{self.redis.PASSWORD}@{self.redis.HOST}:{self.redis.PORT}/{self.redis.DB}"
        else:
            return f"redis://{self.redis.HOST}:{self.redis.PORT}/{self.redis.DB}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for API responses"""
        return {
            "corridor": {
                "corridor_half_m": self.corridor.CORRIDOR_HALF_M,
                "way_offset_thresh_m": self.corridor.WAY_OFFSET_THRESH_M,
                "max_detour_km": self.corridor.MAX_DETOUR_KM,
                "max_detour_ratio": self.corridor.MAX_DETOUR_RATIO,
                "candidate_scan_radius": self.corridor.CANDIDATE_SCAN_RADIUS
            },
            "optimization": {
                "max_vehicles": self.optimization.MAX_VEHICLES,
                "max_bins_per_route": self.optimization.MAX_BINS_PER_ROUTE,
                "max_route_distance_km": self.optimization.MAX_ROUTE_DISTANCE_KM,
                "optimization_timeout_sec": self.optimization.OPTIMIZATION_TIMEOUT_SEC
            },
            "tiling": {
                "tile_res": self.tiling.TILE_RES,
                "neighbor_rings": self.tiling.NEIGHBOR_RINGS,
                "max_bins_per_tile": self.tiling.MAX_BINS_PER_TILE
            },
            "features": {
                "enable_tiling": self.ENABLE_TILING,
                "enable_traffic_waiting": self.ENABLE_TRAFFIC_WAITING,
                "enable_corridor_analysis": self.ENABLE_CORRIDOR_ANALYSIS,
                "enable_llm_advisor": self.llm.ENABLE_LLM_ADVISOR,
                "enable_route_optimization": self.ENABLE_ROUTE_OPTIMIZATION
            },
            "limits": {
                "max_bins_total": self.MAX_BINS_TOTAL,
                "max_trucks_total": self.MAX_TRUCKS_TOTAL,
                "max_active_routes": self.MAX_ACTIVE_ROUTES
            }
        }
    
    def validate(self) -> bool:
        """Validate settings consistency"""
        errors = []
        
        # Check basic constraints
        if self.optimization.MAX_BINS_PER_ROUTE <= 0:
            errors.append("MAX_BINS_PER_ROUTE must be positive")
        
        if self.corridor.CORRIDOR_HALF_M <= 0:
            errors.append("CORRIDOR_HALF_M must be positive")
        
        if self.optimization.OPTIMIZATION_TIMEOUT_SEC <= 0:
            errors.append("OPTIMIZATION_TIMEOUT_SEC must be positive")
        
        if self.tiling.TILE_RES < 0 or self.tiling.TILE_RES > 15:
            errors.append("TILE_RES must be between 0 and 15")
        
        # Check Redis connection parameters
        if not (1 <= self.redis.PORT <= 65535):
            errors.append("Redis port must be between 1 and 65535")
        
        # Check API parameters
        if not (1 <= self.api.PORT <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        if errors:
            print("❌ Settings validation errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance"""
    return settings


def reload_settings():
    """Reload settings from environment"""
    global settings
    settings = Settings()
    return settings