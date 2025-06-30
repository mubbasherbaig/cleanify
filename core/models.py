"""
Cleanify v2-alpha Core Models
Dataclasses for bins, trucks, routes, and system state
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


class TruckStatus(Enum):
    """Truck operational status"""
    IDLE = "idle"
    EN_ROUTE = "en_route"
    COLLECTING = "collecting"
    RETURNING = "returning"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"


class BinType(Enum):
    """Bin type classification"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    ORGANIC = "organic"
    RECYCLING = "recycling"
    GENERAL = "general"
    MEDICAL = "medical"


class RouteStatus(Enum):
    """Route execution status"""
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority(Enum):
    """Collection priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Bin:
    """Core bin model with H3 tiling support"""
    id: str
    lat: float
    lon: float
    capacity_l: int
    fill_level: float          # % (0-100)
    fill_rate_lph: float       # litres/hour
    tile_id: str               # H3 index
    bin_type: BinType = BinType.GENERAL
    way_id: Optional[int] = None  # OSRM edge id (phantom node)
    snap_offset_m: float = 0.0    # along-edge offset
    threshold: float = 85.0       # collection threshold %
    priority: Priority = Priority.MEDIUM
    last_collected: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    assigned_truck: Optional[str] = None
    being_collected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def urgency_score(self) -> float:
        """Calculate urgency based on fill level and threshold"""
        base_urgency = self.fill_level / self.threshold
        
        # Priority multiplier
        priority_multipliers = {
            Priority.LOW: 0.8,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.3,
            Priority.CRITICAL: 2.0
        }
        
        return base_urgency * priority_multipliers[self.priority]

    def estimated_full_time(self) -> Optional[float]:
        """Estimate minutes until bin is full"""
        if self.fill_rate_lph <= 0:
            return None
        
        remaining_capacity = 100.0 - self.fill_level
        fill_rate_per_minute = self.fill_rate_lph / 60.0
        
        if fill_rate_per_minute <= 0:
            return None
            
        return remaining_capacity / fill_rate_per_minute

    def is_urgent(self) -> bool:
        """Check if bin needs immediate collection"""
        return self.fill_level >= self.threshold


@dataclass
class Truck:
    """Core truck model with capacity and status tracking"""
    id: str
    name: str
    capacity_l: int
    lat: float
    lon: float
    current_load_l: int = 0
    status: TruckStatus = TruckStatus.IDLE
    speed_kmh: float = 30.0
    fuel_level: float = 100.0  # %
    route_id: Optional[str] = None
    last_updated: Optional[datetime] = None
    driver_id: Optional[str] = None
    maintenance_due: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def capacity_utilization(self) -> float:
        """Calculate capacity utilization percentage"""
        return (self.current_load_l / self.capacity_l) * 100.0

    def available_capacity(self) -> int:
        """Get available capacity in litres"""
        return max(0, self.capacity_l - self.current_load_l)

    def can_collect_bin(self, bin_obj: Bin) -> bool:
        """Check if truck can collect this bin"""
        if self.status not in [TruckStatus.IDLE, TruckStatus.EN_ROUTE]:
            return False
            
        bin_waste = bin_obj.capacity_l * (bin_obj.fill_level / 100.0)
        return self.available_capacity() >= bin_waste

    def is_available(self) -> bool:
        """Check if truck is available for new routes"""
        return self.status == TruckStatus.IDLE


@dataclass
class RouteStop:
    """Individual stop in a route"""
    id: str
    lat: float
    lon: float
    stop_type: str  # 'bin', 'depot', 'waypoint'
    bin_id: Optional[str] = None
    estimated_arrival: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    estimated_duration_min: float = 5.0
    completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Route:
    """Complete route with stops and optimization data"""
    id: str
    truck_id: str
    stops: List[RouteStop]
    status: RouteStatus = RouteStatus.PLANNED
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_distance_km: float = 0.0
    estimated_duration_min: float = 0.0
    actual_duration_min: Optional[float] = None
    polyline_coords: List[Tuple[float, float]] = field(default_factory=list)
    route_wids: List[int] = field(default_factory=list)  # OSRM way IDs
    optimization_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def bin_count(self) -> int:
        """Count number of bin stops"""
        return len([stop for stop in self.stops if stop.stop_type == 'bin'])

    def completed_stops(self) -> int:
        """Count completed stops"""
        return len([stop for stop in self.stops if stop.completed])

    def progress_percentage(self) -> float:
        """Calculate route progress as percentage"""
        if not self.stops:
            return 0.0
        return (self.completed_stops() / len(self.stops)) * 100.0

    def get_bin_stops(self) -> List[RouteStop]:
        """Get only bin collection stops"""
        return [stop for stop in self.stops if stop.stop_type == 'bin']

    def is_active(self) -> bool:
        """Check if route is currently active"""
        return self.status == RouteStatus.ACTIVE


@dataclass
class TrafficCondition:
    """Traffic condition data"""
    timestamp: datetime
    level: str  # 'free', 'light', 'moderate', 'heavy'
    multiplier: float
    region: str = "default"
    source: str = "prediction"  # 'prediction', 'real-time', 'historical'


@dataclass
class SystemState:
    """Complete system state snapshot"""
    timestamp: datetime
    bins: List[Bin]
    trucks: List[Truck]
    active_routes: List[Route]
    traffic_conditions: List[TrafficCondition]
    simulation_running: bool = False
    simulation_speed: float = 1.0
    current_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def urgent_bins(self) -> List[Bin]:
        """Get all bins needing immediate collection"""
        return [bin_obj for bin_obj in self.bins if bin_obj.is_urgent()]

    def available_trucks(self) -> List[Truck]:
        """Get all available trucks"""
        return [truck for truck in self.trucks if truck.is_available()]

    def active_truck_count(self) -> int:
        """Count trucks currently on routes"""
        return len([truck for truck in self.trucks if truck.status == TruckStatus.EN_ROUTE])

    def system_capacity_utilization(self) -> float:
        """Calculate system-wide capacity utilization"""
        if not self.trucks:
            return 0.0
            
        total_capacity = sum(truck.capacity_l for truck in self.trucks)
        total_load = sum(truck.current_load_l for truck in self.trucks)
        
        return (total_load / total_capacity) * 100.0


@dataclass
class AgentMessage:
    """Message format for agent communication via Redis"""
    agent_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: int = 0  # Higher = more urgent
    ttl_seconds: Optional[int] = None


@dataclass
class CorridorConfig:
    """Configuration for corridor analysis"""
    corridor_half_m: float = 250.0
    way_offset_thresh_m: float = 400.0
    max_detour_km: float = 0.3
    max_detour_ratio: float = 0.05  # 5%
    candidate_scan_radius: float = 1000.0


@dataclass
class WaitingDecision:
    """Departure timing decision"""
    truck_id: str
    decision: str  # 'GO_NOW', 'WAIT_{minutes}_MIN'
    reason: str
    traffic_delay_min: float
    safety_pad_min: float = 1.0
    predicted_overflow_risk: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMRecommendation:
    """LLM advisor recommendation"""
    recommendation_id: str
    route_stats: Dict[str, Any]
    suggested_action: str
    confidence_score: float
    reasoning: str
    alternative_options: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# Type aliases for common data structures
BinMap = Dict[str, Bin]
TruckMap = Dict[str, Truck]
RouteMap = Dict[str, Route]
TileMap = Dict[str, List[Bin]]  # H3 tile_id -> bins in tile