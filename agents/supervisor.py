"""
Cleanify v2-alpha Supervisor Agent - COMPLETE FIX
Orchestrates all other agents and manages system lifecycle
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import structlog
from dataclasses import dataclass
from .llm_advisor import ask_llm
from math import radians, sin, cos, sqrt, atan2
import time
from .base import AgentBase
from core.models import TruckStatus
from .base import AgentBase
from .forecast import ForecastAgent
from .traffic import TrafficAgent
from .route_planner import RoutePlannerAgent
from .corridor import CorridorAgent
from .departure import DepartureAgent
from .emergency import EmergencyAgent
from .watchdog import WatchdogAgent
from core.models import SystemState, Bin, Truck, Route, BinType, TruckStatus
from core.settings import get_settings
from core.geo import haversine_distance

ARRIVAL_TOLERANCE_M = 25.0

logger = structlog.get_logger()
@dataclass
class RouteData:
    id: str
    truck_id: str
    bin_ids: List[str]
    waypoints: List[dict]
    status: str  # "planned", "active", "completed"
    created_at: datetime
    estimated_duration: Optional[int] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0

class SupervisorAgent(AgentBase):
    """
    Supervisor agent that orchestrates the entire Cleanify v2 system
    """
    
    def __init__(self):
        super().__init__("supervisor", "supervisor")
        self.depot_info = {}
        self.system_state = None

        self.simulation_start_time = None
        self.simulation_current_time = None
        self.simulation_running = False
        self.simulation_speed = 1.0
        self.last_update_time = None
        self.depot_info = {}

        # Agent management
        self.managed_agents: Dict[str, AgentBase] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.agent_health: Dict[str, Dict[str, Any]] = {}
        
        # System state
        self.system_state: Optional[SystemState] = None
        
        # Configuration
        self.settings = get_settings()
        
        # Performance metrics
        self.decisions_made = 0
        self.routes_planned = 0
        self.emergencies_handled = 0
        
        # Register message handlers
        self._register_supervisor_handlers()
    
    async def initialize(self):
        """Initialize supervisor and create managed agents"""
        self.logger.info("Initializing Cleanify v2 Supervisor")
        
        # Create agent instances
        await self._create_agents()
        
        # Initialize system state
        await self._initialize_system_state()
        
        self.logger.info("Supervisor initialization complete",
                        agent_count=len(self.managed_agents))
    
    async def main_loop(self):
        """Main orchestration loop with FIXED time progression"""
        self.logger.info("Starting supervisor orchestration loop")
        await self._start_agents()
        
        loop_counter = 0
        
        while self.running:
            try:
                loop_counter += 1
                
                # CRITICAL: Update simulation time EVERY iteration
                if self.simulation_running:
                    await self._simulate_time_progression()
                
                # Use ForecastAgent for bin fill updates every few iterations
                if self.simulation_running and loop_counter % 5 == 0:
                    await self._update_bins_with_forecast_agent()
                
                # Orchestrate with agents every 10 iterations
                if loop_counter % 10 == 0:
                    await self._orchestrate_with_all_agents()
                
                # Update truck movements
                await self._update_truck_movements()
                
                # Monitor for emergencies
                await self._check_critical_bins_auto()
                
                self.decisions_made += 1
                
                # Sleep for a short interval to control loop speed
                await asyncio.sleep(0.1)  # 100ms interval for smooth time progression
                
            except Exception as e:
                print(f"Orchestration error: {e}")
                await asyncio.sleep(1.0)

    async def _update_bins_with_forecast_agent(self):
        """Update bin fills using ForecastAgent for hourly rates"""
        if not self.system_state or not self.simulation_current_time:
            return
            
        current_hour = self.simulation_current_time.hour
        
        for bin_obj in self.system_state.bins:
            if getattr(bin_obj, 'being_collected', False):
                continue
                
            # Use actual hourly rates from config (loaded in bin metadata)
            if hasattr(bin_obj, 'metadata') and bin_obj.metadata.get("has_hourly_rates", False):
                hourly_rates = bin_obj.metadata.get("hourly_fill_rates", {})
                current_hourly_rate = hourly_rates.get(current_hour, bin_obj.fill_rate_lph)
            else:
                current_hourly_rate = bin_obj.fill_rate_lph
            
            # Store for frontend
            if not hasattr(bin_obj, 'metadata'):
                bin_obj.metadata = {}
            bin_obj.metadata['current_hourly_rate'] = current_hourly_rate
            
            # Calculate fill increase per second
            if self.simulation_running:
                fill_increase_per_second = current_hourly_rate / 3600
                fill_increase_liters = fill_increase_per_second * self.simulation_speed
                fill_increase_percent = (fill_increase_liters / bin_obj.capacity_l) * 100
                
                old_fill = bin_obj.fill_level
                bin_obj.fill_level = min(120.0, bin_obj.fill_level + fill_increase_percent)
                
                if bin_obj.fill_level >= 85.0 and old_fill < 85.0:
                    print(f"URGENT: Bin {bin_obj.id} reached {bin_obj.fill_level:.1f}% (Hour {current_hour} rate: {current_hourly_rate}L/h)")
        
    async def shutdown(self):
        """Shutdown supervisor and all managed agents"""
        self.logger.info("Shutting down supervisor")
        
        # Stop all managed agents
        await self._stop_agents()
        
        # Call parent shutdown
        await super().shutdown()
    
    async def cleanup(self):
        """Cleanup supervisor agent - required by AgentBase"""
        self.logger.info("Supervisor agent cleanup")
        
        # Stop all managed agents
        await self._stop_agents()
        
        # Clear state
        self.system_state = None
        self.managed_agents.clear()
        self.agent_tasks.clear()
        self.agent_health.clear()
    
    # Agent Management
    async def _create_agents(self):
        """Create and initialize all managed agents"""
        agent_classes = {
            "forecast": ForecastAgent,
            "traffic": TrafficAgent, 
            "route_planner": RoutePlannerAgent,
            "corridor": CorridorAgent,
            "departure": DepartureAgent,
            "emergency": EmergencyAgent,
            "watchdog": WatchdogAgent
        }
        
        for agent_name, agent_class in agent_classes.items():
            try:
                self.logger.info("Creating agent", agent=agent_name)
                agent = agent_class()
                await agent.initialize()
                self.managed_agents[agent_name] = agent
                self.agent_health[agent_name] = {
                    "status": "healthy",
                    "last_check": datetime.now(),
                    "restarts": 0
                }
            except Exception as e:
                self.logger.error("Failed to create agent", 
                                agent=agent_name, error=str(e))
    
    async def _start_agents(self):
        """Start all managed agents"""
        for agent_name, agent in self.managed_agents.items():
            try:
                self.logger.info("Starting agent", agent=agent_name)
                task = asyncio.create_task(agent.run())
                self.agent_tasks[agent_name] = task
            except Exception as e:
                self.logger.error("Failed to start agent",
                                agent=agent_name, error=str(e))
    
    async def _stop_agents(self):
        """Stop all managed agents"""
        # Signal all agents to stop
        for agent in self.managed_agents.values():
            await agent.shutdown()
        
        # Cancel all tasks
        for task in self.agent_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.agent_tasks:
            await asyncio.gather(*self.agent_tasks.values(), return_exceptions=True)
    
    # System State Management
    async def _initialize_system_state(self):
        """Initialize empty system state with proper time"""
        self.simulation_current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        
        self.system_state = SystemState(
            timestamp=datetime.now(),
            bins=[],
            trucks=[],
            active_routes=[],
            traffic_conditions=[],
            simulation_running=False,
            current_time=self.simulation_current_time  # CRITICAL: Set from start
        )
        
        print(f"🏗️ SUPERVISOR: System state initialized with time {self.simulation_current_time.strftime('%H:%M:%S')}")
        self.logger.info("System state initialized")

    ARRIVAL_TOLERANCE_M = 25.0  # metres, tweak as you like

    async def _update_truck_movements(self):
        """Update truck movements along routes (including RETURNING)."""
        if not self.system_state:
            return
        for truck in self.system_state.trucks:
            if truck.status in [
                TruckStatus.EN_ROUTE,
                TruckStatus.COLLECTING,
                TruckStatus.RETURNING
            ]:
                await self._move_truck_along_route(truck)

    async def _move_truck_along_route(self, truck):
        """Move truck with traffic adjustment, then handle RETURNING cleanup."""
        if not truck.current_route_id:
            return
        route = next((r for r in self.system_state.active_routes if r.id == truck.current_route_id), None)
        if not route or not route.waypoints:
            return
        base_speed = 0.01
        traffic_mult = getattr(route, "traffic_multiplier", 1.0)
        movement_speed = (base_speed / traffic_mult) * self.simulation_speed * 0.1

        if truck.status == TruckStatus.RETURNING:
            depot_lat  = self.depot_info.get("latitude", 33.6844)
            depot_lon  = self.depot_info.get("longitude", 73.0479)
            dist = haversine_distance(truck.lat, truck.lon, depot_lat, depot_lon)

            if dist <= ARRIVAL_TOLERANCE_M:
                await self._complete_truck_route(truck, route)
                return

            if not getattr(truck, "_return_leg_injected", False):
                truck._return_leg_injected = True
                route.waypoints = [
                    {"lat": truck.lat, "lon": truck.lon, "type": "route"},
                    {"lat": depot_lat, "lon": depot_lon, "type": "depot"}
                ]
                truck.route_progress = 0.0

            truck.route_progress = min(1.0, truck.route_progress + movement_speed)
            print("RETURNING branch, movement_speed is", "defined" if "movement_speed" in locals() else "MISSING")

            if truck.route_progress >= 1.0:
                await self._complete_truck_route(truck, route)
            else:
                await self._interpolate_truck_position(truck, route)
            return

        # EN_ROUTE / COLLECTING: advance along the polyline
        if not hasattr(truck, "route_progress"):
            truck.route_progress = 0.0

        base_speed = 0.01
        traffic_multiplier = getattr(route, "traffic_multiplier", 1.0)
        movement_speed = (base_speed / traffic_multiplier) \
                        * self.simulation_speed * 0.1

        truck.route_progress = min(1.0, truck.route_progress + movement_speed)
        if hasattr(route, "progress"):
            route.progress = truck.route_progress

        if truck.route_progress >= 1.0:
            await self._complete_truck_route(truck, route)
        else:
            await self._interpolate_truck_position(truck, route)
    # ─────────────────────────────────────────────────────────

    
    async def _complete_truck_route(self, truck, route):
        """Complete agent-planned route with proper return animation"""
        try:
            print(f"Completing agent-optimized route {route.id}")
            
            # Check if truck is already back at depot
            depot_lat = self.depot_info.get("latitude", 33.6844)
            depot_lon = self.depot_info.get("longitude", 73.0479)
            
            dist_to_depot = haversine_distance(truck.lat, truck.lon, depot_lat, depot_lon)

            # If truck is not at depot (using consistent 25m tolerance), start return journey  
            if dist_to_depot > ARRIVAL_TOLERANCE_M:
                truck.status = TruckStatus.RETURNING
                return
            
            # Only complete when truck reaches depot
            route.status = "completed"
            route.completed_at = datetime.now()
            
            # Reset truck to IDLE
            truck.status = TruckStatus.IDLE
            truck.current_route_id = None
            truck.route_progress = 0.0
            
            # Service bins
            collected_bins = 0
            for bin_id in route.bin_ids:
                bin_obj = next((b for b in self.system_state.bins if b.id == bin_id), None)
                if bin_obj:
                    bin_obj.fill_level = 5.0
                    bin_obj.being_collected = False
                    collected_bins += 1
            for bid in route.bin_ids:
                b = next(b for b in self.system_state.bins if b.id==bid)
                if hasattr(b, 'being_collected'):
                    del b.being_collected
            # Remove completed route
            self.system_state.active_routes = [r for r in self.system_state.active_routes if r.id != route.id]
            
            truck.current_load_l = 0
            truck.last_updated = datetime.now()
            
            print(f"Route {route.id} completed: {collected_bins} bins, truck {truck.id} IDLE")
            
        except Exception as e:
            print(f"Error completing route {route.id}: {e}")

    async def _interpolate_truck_position(self, truck, route):
        """Interpolate truck position along OSRM waypoints"""
        waypoints = route.waypoints
        if len(waypoints) < 2:
            return
        
        total_segments = len(waypoints) - 1
        current_segment_progress = truck.route_progress * total_segments
        segment_index = int(current_segment_progress)
        segment_progress = current_segment_progress - segment_index
        
        if segment_index >= total_segments:
            segment_index = total_segments - 1
            segment_progress = 1.0
        
        start_wp = waypoints[segment_index]
        end_wp = waypoints[min(segment_index + 1, len(waypoints) - 1)]
        
        start_lat, start_lon = start_wp["lat"], start_wp["lon"]
        end_lat, end_lon = end_wp["lat"], end_wp["lon"]
        
        truck.lat = start_lat + (end_lat - start_lat) * segment_progress
        truck.lon = start_lon + (end_lon - start_lon) * segment_progress
        
        if end_wp["type"] == "bin" and segment_progress > 0.8:
            truck.status = TruckStatus.COLLECTING
        elif truck.status == TruckStatus.RETURNING:
            truck.status = TruckStatus.RETURNING  # Keep RETURNING status
        elif truck.status != TruckStatus.COLLECTING:
            truck.status = TruckStatus.EN_ROUTE
        
        truck.last_updated = datetime.now()

    # Orchestration Logic
    async def _orchestrate_with_all_agents(self):
        """Use actual agent methods directly - ENHANCED VERSION"""
        if not self.system_state or not self.simulation_running:
            return
        now = time.time()
        # if hasattr(self, "_last_urgent_check") and now - self._last_urgent_check < 30:
        #     return
        # self._last_urgent_check = now
        try:
            # ─── build set of bin IDs already in flight ───────────────────────
            assigned = {
                bid
                for r in self.system_state.active_routes
                for bid in r.bin_ids or []
            }
            urgent_bins = [
                b for b in self.system_state.bins
                if b.fill_level >= 85.0
                and b.id not in assigned
            ]

            # Find idle trucks
            available_trucks = [
                t for t in self.system_state.trucks
                if t.status in (TruckStatus.IDLE, TruckStatus.RETURNING)
                ]

            # Debug summary
            if urgent_bins:
                lvls = [f"{b.id}({b.fill_level:.0f}%)" for b in urgent_bins[:3]]
                print(f"   📊 Urgent bins: {', '.join(lvls)}"
                    f"{'...' if len(urgent_bins)>3 else ''}")
            if available_trucks:
                names = [t.id for t in available_trucks[:3]]
                print(f"   🚛 Available trucks: {', '.join(names)}"
                    f"{'...' if len(available_trucks)>3 else ''}")

            # If we still have both, plan & dispatch *once*
            if urgent_bins and available_trucks:
                print("🔄 Calling RoutePlannerAgent...")
                routes = await self._use_actual_route_planner_agent(
                    urgent_bins, available_trucks
                )

                if routes:
                    print(f"✅ Got {len(routes)} routes from RoutePlannerAgent")
                    # Mark those bins as in‐flight so we don’t re‐use them
                    for r in routes:
                        for b_id in r["bin_ids"]:
                            bin_obj = next(b for b in self.system_state.bins if b.id == b_id)
                            bin_obj.being_collected = True

                    await self._execute_agent_routes(routes)
                else:
                    print("⚠️ No routes returned from RoutePlannerAgent")
        except Exception as e:
            print(f"❌ Agent orchestration error: {e}", file=sys.stderr)

    async def _execute_agent_routes(self, routes):
        """Execute routes from actual agents - ENHANCED VERSION"""
        if not routes:
            print("⚠️ No routes to execute")
            return
        
        try:
            print(f"🚀 Executing {len(routes)} agent-planned routes...")
            
            # First enhance routes with CorridorAgent (optional)
            try:
                enhanced_routes = await self._enhance_routes_with_corridor_agent(routes)
                print(f"🛣️ Enhanced {len(enhanced_routes)} routes with corridor optimization")
            except Exception as e:
                print(f"⚠️ Corridor enhancement failed, using original routes: {e}")
                enhanced_routes = routes
            
            # Get traffic conditions (optional)
            try:
                traffic_data = await self._get_traffic_conditions_from_agent()
                print(f"🚦 Got traffic data: {traffic_data}")
            except Exception as e:
                print(f"⚠️ Traffic data failed, using default: {e}")
                traffic_data = {"multiplier": 1.0}
            
            # Execute each route
            executed_count = 0
            for route_data in enhanced_routes:
                try:
                    # Validate route data
                    if not route_data.get("truck_id") or not route_data.get("bin_ids"):
                        print(f"⚠️ Invalid route data: {route_data}")
                        continue
                    
                    # Create route object
                    route = RouteData(
                        id=route_data["id"],
                        truck_id=route_data["truck_id"],
                        bin_ids=route_data["bin_ids"],
                        waypoints=route_data["waypoints"],
                        status="active",
                        created_at=datetime.now(),
                        estimated_duration=route_data.get("estimated_duration", 25)
                    )
                    
                    # Add metadata
                    route.progress = 0.0
                    route.traffic_multiplier = traffic_data.get("multiplier", 1.0)
                    route.corridor_bins_added = route_data.get("corridor_bins_added", 0)
                    route.optimization_type = route_data.get("optimization", "agent")
                    
                    # Add to system state
                    self.system_state.active_routes.append(route)
                    
                    # Update truck status
                    truck = next((t for t in self.system_state.trucks if t.id == route_data["truck_id"]), None)
                    if truck:
                        truck.status = TruckStatus.EN_ROUTE
                        truck.current_route_id = route.id
                        truck.route_progress = 0.0
                        print(f"✅ Updated truck {truck.id} status to EN_ROUTE")
                    else:
                        print(f"⚠️ Truck {route_data['truck_id']} not found for route {route.id}")
                    
                    # Mark bins as being collected
                    bins_marked = 0
                    for bin_id in route_data["bin_ids"]:
                        bin_obj = next((b for b in self.system_state.bins if b.id == bin_id), None)
                        if bin_obj:
                            bin_obj.being_collected = True
                            bins_marked += 1
                        else:
                            print(f"⚠️ Bin {bin_id} not found for route {route.id}")
                    
                    print(f"✅ Route {route.id}: truck={route_data['truck_id']}, bins={bins_marked}/{len(route_data['bin_ids'])}, waypoints={len(route_data['waypoints'])}")
                    executed_count += 1
                    
                except Exception as e:
                    print(f"❌ Failed to execute individual route {route_data.get('id', 'UNKNOWN')}: {e}")
                    continue
            
            print(f"🎉 Successfully executed {executed_count}/{len(enhanced_routes)} agent-optimized routes")
            
            # Publish updated system state
            await self._publish_system_state()
            
        except Exception as e:
            print(f"❌ Route execution failed: {e}")
            import traceback
            traceback.print_exc()

    async def _use_actual_route_planner_agent(self, urgent_bins, available_trucks):
        """Use RoutePlannerAgent's actual plan_routes method - FIXED VERSION"""
        try:
            # Get the actual RoutePlannerAgent instance
            route_planner = self.managed_agents.get("route_planner")
            if not route_planner:
                print("RoutePlannerAgent not available")
                return []
            
            # Convert to agent's expected format
            truck_objects = []
            for truck in available_trucks:
                truck_obj = Truck(
                    id=truck.id,
                    name=getattr(truck, 'name', truck.id),
                    lat=truck.lat,
                    lon=truck.lon,
                    capacity_l=truck.capacity_l,
                    status="IDLE"
                )
                truck_obj.current_load_l = getattr(truck, 'current_load_l', 0)
                truck_objects.append(truck_obj)
            
            bin_objects = []
            for bin_data in urgent_bins:
                bin_obj = Bin(
                    id=bin_data.id,
                    lat=bin_data.lat,
                    lon=bin_data.lon,
                    capacity_l=bin_data.capacity_l,
                    fill_level=bin_data.fill_level,
                    fill_rate_lph=bin_data.fill_rate_lph,
                    tile_id=bin_data.tile_id,
                    bin_type=bin_data.bin_type
                )
                bin_objects.append(bin_obj)
            
            print(f"Using RoutePlannerAgent.plan_routes() with {len(truck_objects)} trucks, {len(bin_objects)} bins")
            
            # Call actual agent method
            routes = await route_planner.plan_routes(truck_objects, bin_objects)
            
            print(f"RoutePlannerAgent returned {len(routes)} optimized routes")
            
            # Convert routes to our format - COMPLETE IMPLEMENTATION
            converted_routes = []
            for route in routes:
                # Extract bin IDs from route stops
                bin_ids = []
                for stop in route.stops:
                    if stop.stop_type == "bin" and stop.bin_id:
                        bin_ids.append(stop.bin_id)
                
                # Generate waypoints with OSRM
                waypoints = await self._get_route_waypoints_with_osrm(route)
                
                route_data = {
                    "id": route.id,
                    "truck_id": route.truck_id,
                    "bin_ids": bin_ids,
                    "waypoints": waypoints,
                    "estimated_duration": route.estimated_duration_min,
                    "distance_km": getattr(route, 'total_distance_km', 0),
                    "optimization": "ortools"
                }
                converted_routes.append(route_data)
                
                print(f"✅ Converted route {route.id}: truck={route.truck_id}, bins={len(bin_ids)}, waypoints={len(waypoints)}")
            
            return converted_routes
            
        except Exception as e:
            print(f"RoutePlannerAgent error: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def _get_route_waypoints_with_osrm(self, route):
        """FIXED: OSRM waypoints with proper coordinate validation"""
        try:
            import aiohttp
            import math
            
            # Build coordinates with validation
            coordinates = []
            depot_lat = self.depot_info.get("latitude", 33.6844)
            depot_lon = self.depot_info.get("longitude", 73.0479)
            
            # Start at depot
            coordinates.append(f"{depot_lon},{depot_lat}")
            print(f"🏭 Depot: [{depot_lat}, {depot_lon}]")
            
            # Add bin coordinates with distance validation
            valid_bins = 0
            for stop in route.stops:
                if stop.stop_type == "bin":
                    bin_lat, bin_lon = None, None
                    
                    if hasattr(stop, 'lat') and hasattr(stop, 'lon') and stop.lat and stop.lon:
                        bin_lat, bin_lon = stop.lat, stop.lon
                    elif stop.bin_id:
                        bin_obj = next((b for b in self.system_state.bins if b.id == stop.bin_id), None)
                        if bin_obj:
                            bin_lat, bin_lon = bin_obj.lat, bin_obj.lon
                    
                    # CRITICAL: Validate bin is not at depot location
                    if bin_lat and bin_lon:
                        distance_from_depot = ((bin_lat - depot_lat)**2 + (bin_lon - depot_lon)**2)**0.5 * 111000  # Convert to meters
                        
                        if distance_from_depot > 100:  # At least 100m from depot
                            coordinates.append(f"{bin_lon},{bin_lat}")
                            valid_bins += 1
                            print(f"✅ Valid bin: [{bin_lat}, {bin_lon}] - {distance_from_depot:.0f}m from depot")
                        else:
                            print(f"❌ Bin too close to depot: [{bin_lat}, {bin_lon}] - {distance_from_depot:.0f}m")
            
            # Return to depot
            if valid_bins > 0:
                coordinates.append(f"{depot_lon},{depot_lat}")
            
            if len(coordinates) < 3:
                print(f"⚠️ Insufficient valid coordinates ({len(coordinates)}), using enhanced fallback")
                return self._create_enhanced_fallback_waypoints(route)
            
            # Make OSRM request
            coords_str = ";".join(coordinates)
            osrm_url = f"http://localhost:5000/route/v1/driving/{coords_str}?steps=true&geometries=geojson&overview=full"
            
            print(f"🗺️ OSRM Request: {len(coordinates)} coordinates")
            print(f"   URL: {osrm_url}")
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(osrm_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data.get("routes") and len(data["routes"]) > 0:
                                geometry = data["routes"][0].get("geometry", {}).get("coordinates", [])
                                
                                if len(geometry) > 1:
                                    # ENHANCED: Validate coordinate spread
                                    max_distance = 0.0
                                    min_lat = min_lon = float('inf')
                                    max_lat = max_lon = float('-inf')
                                    
                                    for coord in geometry:
                                        lon, lat = coord[0], coord[1]
                                        min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
                                        min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
                                    
                                    lat_span = (max_lat - min_lat) * 111000  # Convert to meters
                                    lon_span = (max_lon - min_lon) * 111000 * abs(math.cos(math.radians((min_lat + max_lat) / 2)))
                                    total_span = (lat_span**2 + lon_span**2)**0.5
                                    
                                    print(f"🔍 OSRM geometry span: {total_span:.0f}m ({len(geometry)} points)")
                                    
                                    if total_span < 200:  # Less than 200m span is suspicious
                                        print(f"❌ OSRM geometry too clustered ({total_span:.0f}m), using fallback")
                                        return self._create_enhanced_fallback_waypoints(route)
                                    
                                    # Create validated waypoints
                                    waypoints = []
                                    for i, coord in enumerate(geometry):
                                        waypoint_type = "depot" if (i == 0 or i == len(geometry) - 1) else "route"
                                        waypoint_id = "depot" if waypoint_type == "depot" else f"osrm_{i}"
                                        
                                        waypoints.append({
                                            "lat": coord[1],
                                            "lon": coord[0],
                                            "type": waypoint_type,
                                            "id": waypoint_id
                                        })
                                    
                                    print(f"✅ OSRM SUCCESS: {len(waypoints)} waypoints, span {total_span:.0f}m")
                                    return waypoints
                            
                            print(f"❌ OSRM returned no valid geometry")
                        else:
                            error_text = await response.text()
                            print(f"❌ OSRM HTTP {response.status}: {error_text}")
                
                except Exception as e:
                    print(f"❌ OSRM request failed: {e}")
        
        except Exception as e:
            print(f"❌ OSRM waypoint generation error: {e}")
        
        # Always fallback on any error
        return self._create_enhanced_fallback_waypoints(route)

    def _create_enhanced_fallback_waypoints(self, route):
        """Create enhanced fallback waypoints with proper spacing"""
        depot_lat = self.depot_info.get("latitude", 33.6844)
        depot_lon = self.depot_info.get("longitude", 73.0479)
        
        waypoints = []
        valid_bins = []
        
        # Collect valid bins
        for stop in route.stops:
            if stop.stop_type == "bin":
                bin_lat, bin_lon = None, None
                
                if hasattr(stop, 'lat') and hasattr(stop, 'lon') and stop.lat and stop.lon:
                    bin_lat, bin_lon = stop.lat, stop.lon
                elif stop.bin_id:
                    bin_obj = next((b for b in self.system_state.bins if b.id == stop.bin_id), None)
                    if bin_obj:
                        bin_lat, bin_lon = bin_obj.lat, bin_obj.lon
                
                # Only add bins that are meaningfully distant from depot
                if bin_lat and bin_lon:
                    distance = ((bin_lat - depot_lat)**2 + (bin_lon - depot_lon)**2)**0.5 * 111000
                    if distance > 100:  # At least 100m from depot
                        valid_bins.append({
                            "lat": bin_lat,
                            "lon": bin_lon,
                            "id": stop.bin_id or stop.id,
                            "distance": distance
                        })
        
        if not valid_bins:
            print(f"❌ No valid bins found, creating minimal route")
            # Create a small route around depot
            return [
                {"lat": depot_lat, "lon": depot_lon, "type": "depot", "id": "depot"},
                {"lat": depot_lat + 0.002, "lon": depot_lon + 0.002, "type": "route", "id": "fallback_1"},
                {"lat": depot_lat + 0.004, "lon": depot_lon, "type": "route", "id": "fallback_2"},
                {"lat": depot_lat, "lon": depot_lon, "type": "depot", "id": "depot"}
            ]
        
        # Start at depot
        waypoints.append({
            "lat": depot_lat,
            "lon": depot_lon,
            "type": "depot",
            "id": "depot"
        })
        
        # Add intermediate waypoints for smooth movement
        for bin_data in valid_bins:
            # Add 3 intermediate points between depot and bin
            for i in range(1, 4):
                progress = i / 4.0
                intermediate_lat = depot_lat + (bin_data["lat"] - depot_lat) * progress
                intermediate_lon = depot_lon + (bin_data["lon"] - depot_lon) * progress
                
                waypoints.append({
                    "lat": intermediate_lat,
                    "lon": intermediate_lon,
                    "type": "route",
                    "id": f"intermediate_{i}"
                })
            
            # Add bin location
            waypoints.append({
                "lat": bin_data["lat"],
                "lon": bin_data["lon"],
                "type": "bin",
                "id": bin_data["id"]
            })
        
        # Return to depot with intermediate points
        if valid_bins:
            last_bin = valid_bins[-1]
            for i in range(1, 4):
                progress = i / 4.0
                intermediate_lat = last_bin["lat"] + (depot_lat - last_bin["lat"]) * progress
                intermediate_lon = last_bin["lon"] + (depot_lon - last_bin["lon"]) * progress
                
                waypoints.append({
                    "lat": intermediate_lat,
                    "lon": intermediate_lon,
                    "type": "route",
                    "id": f"return_{i}"
                })
        
        # Final depot
        waypoints.append({
            "lat": depot_lat,
            "lon": depot_lon,
            "type": "depot",
            "id": "depot"
        })
        
        # Calculate total span for validation
        if len(waypoints) >= 2:
            first_wp = waypoints[0]
            last_bin = max(valid_bins, key=lambda x: x["distance"]) if valid_bins else None
            
            if last_bin:
                total_span = ((last_bin["lat"] - first_wp["lat"])**2 + (last_bin["lon"] - first_wp["lon"])**2)**0.5 * 111000
                print(f"📍 Enhanced fallback: {len(waypoints)} waypoints, span {total_span:.0f}m, {len(valid_bins)} bins")
            else:
                print(f"📍 Enhanced fallback: {len(waypoints)} waypoints (minimal route)")
        
        return waypoints

    

    async def _enhance_routes_with_corridor_agent(self, routes):
        """Use CorridorAgent's actual build_corridor method"""
        try:
            corridor_agent = self.managed_agents.get("corridor")
            if not corridor_agent:
                print("CorridorAgent not available")
                return routes
            
            enhanced_routes = []
            
            for route in routes:
                print(f"Using CorridorAgent.build_corridor() for route {route['id']}")
                
                # Convert waypoints to polyline
                polyline_latlon = [(wp["lat"], wp["lon"]) for wp in route["waypoints"]]
                
                # Get all candidate bins (>70% fill)
                candidate_bins = []
                for bin_obj in self.system_state.bins:
                    if bin_obj.fill_level >= 50.0 and bin_obj.id not in route["bin_ids"]:
                        candidate_bins.append(bin_obj)
                                
                # ───── GPT micro-advisor (adds near-route bins) ─────────────────────────
                llm_cfg = get_settings().llm
                extra_ids: list[int] = []

                if (llm_cfg.ENABLE_LLM_ADVISOR
                    and candidate_bins
                    and len(candidate_bins) > 0):

                    # Build compressed payload
                    payload = {
                        "truck_capacity_left": (
                            next((t.capacity_l - t.current_load_l for t in self.system_state.trucks
                                if t.id == route["truck_id"]), 0)
                        ),
                        "candidates": [
                            {
                                "id":       b.id,
                                "proj_fill": round(b.fill_level, 1),   # we didn’t compute proj here
                                "detour_m": 0                         # unknown yet
                            }
                            for b in candidate_bins[: llm_cfg.MAX_CANDIDATE_BINS]
                        ]
                    }

                    # One–line async call (cheap, cached, rate-limited)
                    extra_ids = await ask_llm(payload) or []

                    # Strict whitelist – remove anything GPT hallucinated
                    valid_ids = {b.id for b in candidate_bins}
                    extra_ids = [bid for bid in extra_ids if bid in valid_ids]

                    if extra_ids:
                        print(f"🧠 LLM added {len(extra_ids)} bins to route {route['id']}: {extra_ids}")
                # ─────────────────────────────────────────────────────────────────────────


                # Add this in _enhance_routes_with_corridor_agent before corridor call:
                fill_distribution = {}
                for bin_obj in self.system_state.bins:
                    fill_range = f"{int(bin_obj.fill_level//10)*10}-{int(bin_obj.fill_level//10)*10+9}%"
                    fill_distribution[fill_range] = fill_distribution.get(fill_range, 0) + 1

                print(f"📊 Bin fill distribution: {fill_distribution}")

                # Call actual CorridorAgent method with timeout protection
                try:
                    corridor_bin_ids = await asyncio.wait_for(
                        corridor_agent.build_corridor(
                            polyline_latlon=polyline_latlon,
                            route_wids=[],  # No OSM way IDs
                            bins=candidate_bins
                        ),
                        timeout=5.0  # 5 second timeout
                    )
                except asyncio.TimeoutError:
                    print(f"⚠️ CorridorAgent timed out for route {route['id']}, skipping corridor analysis")
                    corridor_bin_ids = set()
                except Exception as e:
                    print(f"⚠️ CorridorAgent error for route {route['id']}: {e}")
                    corridor_bin_ids = set()
                
                print(f"CorridorAgent found {len(corridor_bin_ids)} additional bins for route {route['id']}")
                
                # Add corridor bins to route
                if corridor_bin_ids:
                    route["bin_ids"].extend(list(corridor_bin_ids))
                    route["corridor_bins_added"] = len(corridor_bin_ids)
                    
                    # Update waypoints to include corridor bins
                    route["waypoints"] = await self._rebuild_waypoints_with_corridor_bins(
                        route["waypoints"], corridor_bin_ids
                    )
                # Merge LLM extras (if any) after corridor logic
                if extra_ids:
                    route["bin_ids"].extend(extra_ids)
                    route["corridor_bins_added"] = route.get("corridor_bins_added", 0) + len(extra_ids)

                enhanced_routes.append(route)
            
            return enhanced_routes
            
        except Exception as e:
            print(f"CorridorAgent error: {e}")
            return routes

    async def _rebuild_waypoints_with_corridor_bins(self, original_waypoints, corridor_bin_ids):
        """Rebuild waypoints to include corridor bins"""
        try:
            # Extract depot waypoints
            depot_waypoints = [wp for wp in original_waypoints if wp["type"] == "depot"]
            
            # Get all bin objects (original + corridor)
            all_bins = []
            for wp in original_waypoints:
                if wp["type"] == "bin":
                    bin_obj = next((b for b in self.system_state.bins if b.id == wp["id"]), None)
                    if bin_obj:
                        all_bins.append(bin_obj)
            
            for bin_id in corridor_bin_ids:
                bin_obj = next((b for b in self.system_state.bins if b.id == bin_id), None)
                if bin_obj:
                    all_bins.append(bin_obj)
            
            if not all_bins:
                return original_waypoints
            
            # Sort bins by fill level (highest first)
            all_bins.sort(key=lambda b: b.fill_level, reverse=True)
            
            # Build new coordinate sequence for OSRM
            coordinates = []
            depot_lat = self.depot_info.get("latitude", 33.6844)
            depot_lon = self.depot_info.get("longitude", 73.0479)
            
            coordinates.append(f"{depot_lon},{depot_lat}")
            for bin_obj in all_bins:
                coordinates.append(f"{bin_obj.lon},{bin_obj.lat}")
            coordinates.append(f"{depot_lon},{depot_lat}")
            
            # Get optimized route from OSRM
            coords_str = ";".join(coordinates)
            osrm_url = f"http://localhost:5000/route/v1/driving/{coords_str}?steps=true&geometries=geojson"
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(osrm_url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("routes"):
                            geometry = data["routes"][0]["geometry"]["coordinates"]
                            
                            new_waypoints = []
                            for i, coord in enumerate(geometry):
                                new_waypoints.append({
                                    "lat": coord[1],
                                    "lon": coord[0],
                                    "type": "route",
                                    "id": f"enhanced_{i}"
                                })
                            
                            if new_waypoints:
                                new_waypoints[0]["type"] = "depot"
                                new_waypoints[0]["id"] = "depot"
                                new_waypoints[-1]["type"] = "depot"
                                new_waypoints[-1]["id"] = "depot"
                            
                            return new_waypoints
            
        except Exception as e:
            print(f"Waypoint rebuilding failed: {e}")
        
        return original_waypoints

    async def _get_traffic_conditions_from_agent(self):
        """Use TrafficAgent's actual traffic analysis"""
        try:
            traffic_agent = self.managed_agents.get("traffic")
            if not traffic_agent:
                print("TrafficAgent not available")
                return {"multiplier": 1.0, "level": "unknown"}
            
            # Call TrafficAgent method (if it has a direct method)
            # For now, simulate based on time
            current_hour = self.simulation_current_time.hour
            
            if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
                return {"multiplier": 1.8, "level": "heavy"}
            elif 10 <= current_hour <= 16:
                return {"multiplier": 1.2, "level": "moderate"}
            else:
                return {"multiplier": 1.0, "level": "light"}
            
        except Exception as e:
            print(f"TrafficAgent error: {e}")
            return {"multiplier": 1.0, "level": "unknown"}

    async def _request_forecast_predictions(self):
        """Request overflow predictions from ForecastAgent"""
        try:
            correlation_id = f"forecast_{datetime.now().strftime('%H%M%S')}"
            
            # Send to ForecastAgent using proper message format
            await self.send_message(
                "predict_overflow",
                {
                    "bins": [
                        {
                            "id": bin_obj.id,
                            "lat": bin_obj.lat,
                            "lon": bin_obj.lon,
                            "capacity_l": bin_obj.capacity_l,
                            "fill_level": bin_obj.fill_level,
                            "fill_rate_lph": bin_obj.fill_rate_lph,
                            "tile_id": bin_obj.tile_id,
                            "bin_type": "general"
                        }
                        for bin_obj in self.system_state.bins
                    ],
                    "correlation_id": correlation_id
                },
                target_stream="cleanify:agents:forecast:input"
            )
            
            print(f"Sent overflow prediction request to ForecastAgent")
            return {"status": "requested", "correlation_id": correlation_id}
            
        except Exception as e:
            print(f"Forecast request failed: {e}")
            return None

    async def _check_critical_bins_auto(self):
        """Auto-handle critical bins without manual intervention"""
        if not self.system_state:
            return
            
        critical_bins = [
            bin_obj for bin_obj in self.system_state.bins
            if bin_obj.fill_level >= 120.0  # Critical overflow threshold
        ]
        
        if critical_bins:
            print(f"🚨 SUPERVISOR: AUTO-EMERGENCY for {len(critical_bins)} critical bins")
            
            # Immediately assign any available trucks to critical bins
            available_trucks = [
                truck for truck in self.system_state.trucks 
                if truck.status == TruckStatus.IDLE
            ]
            
            emergency_routes = 0
            for i, truck in enumerate(available_trucks):
                if i < len(critical_bins):
                    bin_obj = critical_bins[i]
                    
                    # Emergency assignment
                    emergency_route_id = f"EMERGENCY_{truck.id}_{int(datetime.now().timestamp())}"
                    
                    truck.status = TruckStatus.EN_ROUTE
                    truck.route_id = emergency_route_id
                    bin_obj.being_collected = True
                    bin_obj.assigned_truck = truck.id
                    
                    emergency_routes += 1
                    print(f"🚨 EMERGENCY DISPATCH: {truck.id} → {bin_obj.id} ({bin_obj.fill_level:.1f}%)")
            
            self.emergencies_handled += 1
            
            # Send emergency alert for monitoring
            await self._send_emergency_alert(critical_bins)

    
    async def _send_emergency_alert(self, critical_bins: List[Bin]):
        """Send emergency alert for critical bins"""
        emergency_data = {
            "type": "critical_bins",
            "bins": [self._bin_to_dict(bin_obj) for bin_obj in critical_bins],
            "timestamp": datetime.now().isoformat(),
            "priority": "high"
        }
        
        await self.send_message(
            "emergency_alert",
            emergency_data,
            target_stream="cleanify:agents:emergency:input"
        )
        
        self.emergencies_handled += 1
        self.logger.warning("Critical bins detected",
                           count=len(critical_bins))
 
    async def _monitor_active_routes(self):
        """Monitor and auto-complete routes"""
        if not self.system_state:
            return
        
        try:
            # Simulate route completion after some time
            current_time = datetime.now()
            
            for truck in self.system_state.trucks:
                if truck.status == TruckStatus.EN_ROUTE and truck.route_id:
                    # Simple completion logic - routes take ~5 minutes
                    route_start_timestamp = int(truck.route_id.split('_')[-1])
                    route_start_time = datetime.fromtimestamp(route_start_timestamp)
                    
                    if (current_time - route_start_time).total_seconds() > 300:  # 5 minutes
                        print(f"✅ SUPERVISOR: Auto-completing route for {truck.id}")
                        
                        # Complete the route
                        truck.status = TruckStatus.IDLE
                        truck.route_id = None
                        truck.current_load_l = 0  # Assume truck empties
                        
                        # Find and reset the bin
                        for bin_obj in self.system_state.bins:
                            if bin_obj.assigned_truck == truck.id:
                                bin_obj.being_collected = False
                                bin_obj.assigned_truck = None
                                bin_obj.fill_level = 10.0  # Reset to low level
                                bin_obj.last_collected = self.simulation_current_time
                                print(f"♻️ SUPERVISOR: {bin_obj.id} collected, reset to {bin_obj.fill_level}%")
                                break
                        
        except Exception as e:
            print(f"❌ SUPERVISOR: Route monitoring error: {e}")
    
    def _register_supervisor_handlers(self):
        """Register supervisor-specific message handlers"""
        self.register_handler("load_config", self._handle_load_config)
        self.register_handler("start_simulation", self._handle_start_simulation)
        self.register_handler("pause_simulation", self._handle_pause_simulation)
        self.register_handler("set_simulation_speed", self._handle_set_simulation_speed)
        self.register_handler("get_agent_health", self._handle_get_agent_health)
        self.register_handler("route_planned", self._handle_route_planned)
    
    async def _handle_load_config(self, config_data):
        """Load configuration and create system objects - FIXED VERSION"""
        try:
            config = config_data.get("config", config_data)
            
            # Store depot info
            if "depot" in config:
                self.depot_info = config["depot"]
                print(f"✅ Depot loaded: {self.depot_info.get('name', 'Unnamed')}")
            
            # Create bins with proper BinType handling
            bins = []
            if "bins" in config:
                for bin_config in config["bins"]:
                    # Set default bin_type if not specified
                    bin_type_str = bin_config.get("bin_type", "general")
                    try:
                        bin_type = BinType(bin_type_str.lower())
                    except ValueError:
                        bin_type = BinType.GENERAL  # Default fallback
                    
                    bin_obj = Bin(
                        id=bin_config["id"],
                        lat=bin_config.get("latitude", bin_config.get("lat", 0.0)),
                        lon=bin_config.get("longitude", bin_config.get("lon", 0.0)),
                        capacity_l=bin_config.get("capacity_l", 240),
                        fill_level=bin_config.get("fill_level", 75.0),
                        fill_rate_lph=bin_config.get("fill_rate_lph", 3.0),
                        tile_id=bin_config.get("tile_id", ""),
                        bin_type=bin_type  # Properly set BinType
                    )
                    bins.append(bin_obj)
            
            # Create trucks
            trucks = []
            if "trucks" in config:
                for truck_config in config["trucks"]:
                    truck_obj = Truck(
                        id=truck_config["id"],
                        name=truck_config.get("name", truck_config["id"]),
                        capacity_l=truck_config.get("capacity_l", truck_config.get("capacity", 5000)),
                        lat=self.depot_info.get("latitude", 33.6844),
                        lon=self.depot_info.get("longitude", 73.0479),
                        status=TruckStatus.IDLE
                    )
                    trucks.append(truck_obj)
            
            # Create system state
            self.system_state = SystemState(
                timestamp=datetime.now(),
                bins=bins,
                trucks=trucks,
                active_routes=[],
                traffic_conditions=[],
                simulation_running=False,
                current_time=datetime.now()
            )
            
            print(f"✅ Configuration loaded: {len(bins)} bins, {len(trucks)} trucks")
            return True
            
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _handle_start_simulation(self, data: Dict[str, Any]):
        """Handle simulation start - FIXED with proper time initialization"""
        try:
            print("🚀 SUPERVISOR: Starting simulation")
            
            # Initialize timing
            self.simulation_running = True
            self.simulation_start_time = datetime.now()
            self.last_update_time = datetime.now()
            
            # CRITICAL: Always set simulation time to 8:00 AM
            self.simulation_current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            
            # CRITICAL: Update system state immediately
            if self.system_state:
                self.system_state.simulation_running = True
                self.system_state.current_time = self.simulation_current_time
                self.system_state.simulation_speed = self.simulation_speed
            
            print(f"✅ SUPERVISOR: Simulation started at {self.simulation_current_time.strftime('%H:%M:%S')}")
            self.logger.info("Simulation started")
            
            # CRITICAL: Publish state immediately after starting
            await self._publish_system_state()
            
            return {
                "status": "simulation_started", 
                "simulation_time": self.simulation_current_time.isoformat(),
                "real_time": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ SUPERVISOR: Simulation start failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_pause_simulation(self, data: Dict[str, Any]):
        """Handle simulation pause"""
        try:
            print("⏸️ SUPERVISOR: Pausing simulation")
            
            self.simulation_running = False
            
            if self.system_state:
                self.system_state.simulation_running = False
            
            print("✅ SUPERVISOR: Simulation paused")
            self.logger.info("Simulation paused")
            
            return {"status": "simulation_paused"}
        except Exception as e:
            print(f"❌ SUPERVISOR: Simulation pause failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_set_simulation_speed(self, data: Dict[str, Any]):
        """Handle simulation speed change - FIXED with 25x limit"""
        try:
            new_speed = data.get("speed", 1.0)
            # FIXED: Increase maximum speed from 10x to 25x to match frontend slider
            self.simulation_speed = max(0.1, min(25.0, new_speed))
            
            # CRITICAL: Also update system state if it exists
            if self.system_state:
                self.system_state.simulation_speed = self.simulation_speed
            
            print(f"⚡ SUPERVISOR: Speed set to {self.simulation_speed}x")
            return {"status": "speed_updated", "new_speed": self.simulation_speed}
        except Exception as e:
            print(f"❌ SUPERVISOR: Speed change failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_get_agent_health(self, data: Dict[str, Any]):
        """Handle agent health request"""
        return {
            "agents": self.agent_health,
            "total_agents": len(self.managed_agents),
            "healthy_agents": len([h for h in self.agent_health.values() if h["status"] == "healthy"])
        }
    
    async def _handle_route_planned(self, data: Dict[str, Any]):
        """Handle route planned notification"""
        route_data = data.get("route", {})
        self.routes_planned += 1
        self.logger.info("Route planned", route_id=route_data.get("id"))
        return {"status": "route_acknowledged"}

    async def _publish_system_state(self):
        """Publish system state with guaranteed time"""
        if not self.system_state:
            return

        try:
            # CRITICAL: Ensure time is always available
            if self.simulation_current_time is None:
                self.simulation_current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            
            state_dict = {
                "timestamp": self.system_state.timestamp.isoformat(),
                "bins": [self._bin_to_dict(bin_obj) for bin_obj in self.system_state.bins],
                "trucks": [self._truck_to_dict(truck) for truck in self.system_state.trucks],
                "active_routes": [self._route_to_dict(route) for route in self.system_state.active_routes],
                "simulation_running": self.simulation_running,
                "simulation_speed": self.simulation_speed,
                "current_time": self.simulation_current_time.isoformat()  # GUARANTEED
            }

            # Publish to Redis if available
            if hasattr(self, 'redis_client') and self.redis_client:
                await self.redis_client.publish(
                    "system_state_updates",
                    json.dumps(state_dict, indent=2)
                )
            
            print(f"📤 SUPERVISOR: Published state with time {self.simulation_current_time.strftime('%H:%M:%S')}")

        except Exception as e:
            print(f"❌ SUPERVISOR: Failed to publish system state: {e}")
    
    def _bin_to_dict(self, bin_obj):
        """Convert bin with forecast data"""
        current_hourly_rate = bin_obj.fill_rate_lph
        
        if hasattr(bin_obj, 'metadata') and self.simulation_current_time:
            current_hourly_rate = bin_obj.metadata.get('current_hourly_rate', bin_obj.fill_rate_lph)
        
        return {
            "id": bin_obj.id,
            "lat": float(bin_obj.lat),
            "lon": float(bin_obj.lon),
            "capacity_l": bin_obj.capacity_l,
            "fill_level": bin_obj.fill_level,
            "fill_rate_lph": bin_obj.fill_rate_lph,
            "current_hourly_rate": current_hourly_rate,
            "has_hourly_rates": getattr(bin_obj, 'metadata', {}).get("has_hourly_rates", False),
            "being_collected": getattr(bin_obj, 'being_collected', False),
            "last_updated": bin_obj.last_updated.isoformat() if getattr(bin_obj, 'last_updated', None) else None
        }

    def _truck_to_dict(self, truck):
        """Convert truck with movement data"""
        return {
            "id": truck.id,
            "name": getattr(truck, 'name', truck.id),
            "lat": float(truck.lat),
            "lon": float(truck.lon),
            "status": truck.status.value if hasattr(truck.status, 'value') else str(truck.status),
            "capacity_l": truck.capacity_l,
            "current_load_l": getattr(truck, 'current_load_l', 0),
            "current_route_id": getattr(truck, 'current_route_id', None),
            "route_progress": getattr(truck, 'route_progress', 0.0),
            "last_updated": truck.last_updated.isoformat() if getattr(truck, 'last_updated', None) else None
        }
    
    def _route_to_dict(self, route):
        """Convert route with agent optimization data"""
        return {
            "id": route.id,
            "truck_id": route.truck_id,
            "bin_ids": route.bin_ids,
            "waypoints": route.waypoints,
            "status": route.status,
            "progress": getattr(route, 'progress', 0.0),
            "traffic_multiplier": getattr(route, 'traffic_multiplier', 1.0),
            "corridor_bins_added": getattr(route, 'corridor_bins_added', 0),
            "optimization_type": getattr(route, 'optimization_type', 'unknown'),
            "created_at": route.created_at.isoformat() if route.created_at else None,
            "completed_at": getattr(route, 'completed_at', None).isoformat() if getattr(route, 'completed_at', None) else None,
            "estimated_duration": getattr(route, 'estimated_duration', None)
        }
    
    async def _simulate_time_progression(self):
        """FIXED time progression with proper speed handling"""
        if not self.simulation_running or not self.simulation_current_time:
            return

        try:
            current_real_time = datetime.now()
            
            # Initialize last_update_time if not set
            if self.last_update_time is None:
                self.last_update_time = current_real_time
                return
            
            # Calculate real time elapsed since last update (in seconds)
            real_time_delta_seconds = (current_real_time - self.last_update_time).total_seconds()
            
            # Apply simulation speed multiplier to get simulation time delta
            simulation_time_delta_seconds = real_time_delta_seconds * self.simulation_speed
            
            # Advance simulation time
            old_sim_time = self.simulation_current_time
            self.simulation_current_time += timedelta(seconds=simulation_time_delta_seconds)
            self.last_update_time = current_real_time

            # Update system state time
            if self.system_state:
                self.system_state.current_time = self.simulation_current_time
                self.system_state.simulation_speed = self.simulation_speed

            # Debug logging for time progression
            if simulation_time_delta_seconds > 0.5:  # Only log significant changes
                time_diff = (self.simulation_current_time - old_sim_time).total_seconds()
                # print(f"⏰ TIME: {old_sim_time.strftime('%H:%M:%S')} → {self.simulation_current_time.strftime('%H:%M:%S')} "
                #     f"(+{time_diff:.1f}s @ {self.simulation_speed}x speed)")

            # Simulate bin fill progression with FIXED calculation
            await self._simulate_bin_fills(simulation_time_delta_seconds)

        except Exception as e:
            print(f"❌ SUPERVISOR: Time progression error: {e}")
            import traceback
            traceback.print_exc()

    async def _simulate_bin_fills(self, simulation_elapsed_seconds: float):
        """FIXED bin filling simulation with proper time calculation"""
        if not self.system_state or not self.system_state.bins:
            return
            
        current_hour = self.simulation_current_time.hour
        
        # Convert elapsed simulation seconds to hours for fill rate calculation
        simulation_elapsed_hours = simulation_elapsed_seconds / 3600.0
        
        # Only process if enough time has elapsed (avoid tiny increments)
        if simulation_elapsed_hours < 0.001:  # Less than 3.6 seconds of simulation time
            return
        
        bins_updated = 0
        for bin_obj in self.system_state.bins:
            try:
                # Skip bins being collected
                if getattr(bin_obj, 'being_collected', False):
                    continue
                
                # Get base fill rate
                base_rate = bin_obj.fill_rate_lph
                
                # Apply hourly variation
                hour_multiplier = 1.0
                if 6 <= current_hour <= 10:  # Morning rush
                    hour_multiplier = 1.5
                elif 17 <= current_hour <= 21:  # Evening rush  
                    hour_multiplier = 1.3
                elif 22 <= current_hour or current_hour <= 5:  # Night
                    hour_multiplier = 0.3
                    
                adjusted_rate = base_rate * hour_multiplier
                
                # FIXED: Calculate fill increase based on actual elapsed time
                fill_increase_liters = adjusted_rate * simulation_elapsed_hours
                fill_increase_percentage = (fill_increase_liters / bin_obj.capacity_l) * 100.0
                
                # Update fill level
                old_fill = bin_obj.fill_level
                bin_obj.fill_level = min(120.0, bin_obj.fill_level + fill_increase_percentage)  # Allow 20% overflow
                
                # Store current hourly rate for frontend display
                bin_obj.current_hourly_rate = adjusted_rate
                
                # Log significant changes
                if bin_obj.fill_level >= 85.0 and old_fill < 85.0:
                    print(f"URGENT: Bin {bin_obj.id} reached {bin_obj.fill_level:.1f}% (Hour {current_hour} rate: {adjusted_rate:.2f}L/h)")
                
                bins_updated += 1
                        
            except Exception as e:
                print(f"❌ Error simulating fill for bin {bin_obj.id}: {e}")
        
        # Debug logging for bin fills
        if bins_updated > 0 and simulation_elapsed_hours > 0.01:  # Only log meaningful updates
            print(f"📊 FILLS: Updated {bins_updated} bins (+{simulation_elapsed_hours*60:.1f} sim-minutes @ hour {current_hour})")
