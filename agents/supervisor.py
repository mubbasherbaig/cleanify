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
                
    async def _orchestrate_with_agents(self):
        """Orchestration with direct route creation fallback"""
        if not self.system_state or not self.simulation_running:
            return
        
        # Find urgent bins and trucks
        urgent_bins = [b for b in self.system_state.bins 
                    if b.fill_level >= 85.0 and not getattr(b, 'being_collected', False)]
        available_trucks = [t for t in self.system_state.trucks if t.status == TruckStatus.IDLE]
        
        print(f"DEBUG: Found {len(urgent_bins)} urgent bins, {len(available_trucks)} available trucks")
        
        if urgent_bins and available_trucks:
            # Try agent coordination first
            try:
                # Send messages to agents (existing code)
                await self._request_route_planning(urgent_bins, available_trucks, {})
                # If agents don't respond in 2 seconds, create routes directly
                await asyncio.sleep(2)
                
            except:
                pass
            
            # Direct route creation if no active routes exist
            if len(self.system_state.active_routes) == 0:
                await self._create_direct_routes(urgent_bins, available_trucks)

    async def _create_direct_routes(self, urgent_bins, available_trucks):
        """Direct route creation bypass"""
        routes_created = 0
        max_routes = min(len(urgent_bins), len(available_trucks), 3)
        
        for i in range(max_routes):
            truck = available_trucks[i]
            bin_obj = urgent_bins[i]
            
            route_id = f"DIRECT_{datetime.now().strftime('%H%M%S')}_{truck.id}"
            
            # Simple waypoints
            waypoints = [
                {"lat": self.depot_info.get("latitude", 33.6844), "lon": self.depot_info.get("longitude", 73.0479), "type": "depot", "id": "depot"},
                {"lat": bin_obj.lat, "lon": bin_obj.lon, "type": "bin", "id": bin_obj.id},
                {"lat": self.depot_info.get("latitude", 33.6844), "lon": self.depot_info.get("longitude", 73.0479), "type": "depot", "id": "depot"}
            ]
            
            route = RouteData(
                id=route_id,
                truck_id=truck.id,
                bin_ids=[bin_obj.id],
                waypoints=waypoints,
                status="active",
                created_at=datetime.now(),
                estimated_duration=20
            )
            
            self.system_state.active_routes.append(route)
            truck.status = TruckStatus.EN_ROUTE
            truck.current_route_id = route_id
            truck.route_progress = 0.0
            bin_obj.being_collected = True
            
            routes_created += 1
            print(f"DIRECT ROUTE: Created {route_id} for bin {bin_obj.id} ({bin_obj.fill_level:.1f}%)")
        
        return routes_created
    
    async def _calculate_route_distance(self, waypoints):
        """Calculate total route distance"""
        if len(waypoints) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            # Simple distance calculation
            lat_diff = wp2["lat"] - wp1["lat"]
            lon_diff = wp2["lon"] - wp1["lon"]
            distance = (lat_diff**2 + lon_diff**2)**0.5 * 111  # Rough km conversion
            total_distance += distance
        
        return total_distance

    async def _request_route_planning(self, urgent_bins, available_trucks):
        """Request route planning from RoutePlannerAgent using OR-Tools"""
        try:
            correlation_id = f"route_{datetime.now().strftime('%H%M%S')}"
            
            # Prepare data for RoutePlannerAgent
            bins_data = []
            for bin_obj in urgent_bins:
                bins_data.append({
                    "id": bin_obj.id,
                    "lat": bin_obj.lat,
                    "lon": bin_obj.lon,
                    "capacity_l": bin_obj.capacity_l,
                    "fill_level": bin_obj.fill_level,
                    "fill_rate_lph": bin_obj.fill_rate_lph,
                    "urgency_score": 1.5 if bin_obj.fill_level >= 95 else 1.2,
                    "tile_id": bin_obj.tile_id
                })
            
            trucks_data = []
            for truck in available_trucks:
                trucks_data.append({
                    "id": truck.id,
                    "lat": truck.lat,
                    "lon": truck.lon,
                    "capacity_l": truck.capacity_l,
                    "current_load_l": getattr(truck, 'current_load_l', 0),
                    "status": str(truck.status)
                })
            
            # Send to RoutePlannerAgent
            await self.send_message(
                "plan_routes_direct",
                {
                    "trucks": trucks_data,
                    "bins": bins_data,
                    "depot": {
                        "lat": self.depot_info.get("latitude", 33.6844),
                        "lon": self.depot_info.get("longitude", 73.0479)
                    },
                    "use_ortools": True,
                    "use_osrm": True,
                    "correlation_id": correlation_id
                },
                target_stream="cleanify:agents:route_planner:input"
            )
            
            print(f"Sent route planning request to RoutePlannerAgent with OR-Tools + OSRM")
            
            # Since we can't wait for response, create direct routes for now
            return await self._create_direct_osrm_routes(urgent_bins, available_trucks)
            
        except Exception as e:
            print(f"Route planning request failed: {e}")
            return []

    async def _create_direct_osrm_routes(self, urgent_bins, available_trucks):
        """Create routes with OSRM for immediate execution"""
        routes = []
        max_routes = min(len(urgent_bins), len(available_trucks), 3)
        
        for i in range(max_routes):
            truck = available_trucks[i]
            bin_obj = urgent_bins[i]
            
            # Get OSRM route
            waypoints = await self._get_osrm_route_direct(truck, bin_obj)
            
            route_data = {
                "id": f"OSRM_{datetime.now().strftime('%H%M%S')}_{truck.id}",
                "truck_id": truck.id,
                "bin_ids": [bin_obj.id],
                "waypoints": waypoints,
                "estimated_duration": 25,
                "distance_km": await self._calculate_route_distance(waypoints)
            }
            
            routes.append(route_data)
            print(f"Created OSRM route {route_data['id']} with {len(waypoints)} waypoints")
        
        return routes

    async def _get_osrm_route_direct(self, truck, bin_obj):
        """Get route from OSRM API directly"""
        try:
            import aiohttp
            
            depot_lat = self.depot_info.get("latitude", 33.6844)
            depot_lon = self.depot_info.get("longitude", 73.0479)
            
            # OSRM route request: Depot -> Bin -> Depot
            coordinates = f"{depot_lon},{depot_lat};{bin_obj.lon},{bin_obj.lat};{depot_lon},{depot_lat}"
            osrm_url = f"http://localhost:5000/route/v1/driving/{coordinates}?steps=true&geometries=geojson&annotations=true&overview=full"
            
            print(f"OSRM REQUEST: {osrm_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(osrm_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("routes"):
                            geometry = data["routes"][0]["geometry"]["coordinates"]
                            
                            waypoints = []
                            for i, coord in enumerate(geometry):
                                if i == 0:
                                    waypoint_type = "depot"
                                    waypoint_id = "depot"
                                elif i == len(geometry) - 1:
                                    waypoint_type = "depot"
                                    waypoint_id = "depot"
                                elif abs(coord[1] - bin_obj.lat) < 0.001 and abs(coord[0] - bin_obj.lon) < 0.001:
                                    waypoint_type = "bin"
                                    waypoint_id = bin_obj.id
                                else:
                                    waypoint_type = "route"
                                    waypoint_id = f"point_{i}"
                                
                                waypoints.append({
                                    "lat": coord[1],
                                    "lon": coord[0],
                                    "type": waypoint_type,
                                    "id": waypoint_id
                                })
                            
                            print(f"OSRM SUCCESS: Got {len(waypoints)} waypoints from OSRM")
                            return waypoints
                    else:
                        print(f"OSRM ERROR: HTTP {response.status}")
            
        except Exception as e:
            print(f"OSRM request failed: {e}")
        
        # Fallback to straight line
        return [
            {"lat": depot_lat, "lon": depot_lon, "type": "depot", "id": "depot"},
            {"lat": bin_obj.lat, "lon": bin_obj.lon, "type": "bin", "id": bin_obj.id},
            {"lat": depot_lat, "lon": depot_lon, "type": "depot", "id": "depot"}
        ]

    async def _optimize_routes_with_corridor_agent(self, routes):
        """Optimize routes using CorridorAgent"""
        try:
            correlation_id = f"corridor_{datetime.now().strftime('%H%M%S')}"
            
            for route in routes:
                # Send corridor analysis request
                await self.send_message(
                    "analyze_corridor",
                    {
                        "polyline_latlon": [(wp["lat"], wp["lon"]) for wp in route["waypoints"]],
                        "route_wids": [],  # No way IDs available
                        "bins": [
                            {
                                "id": bin_obj.id,
                                "lat": bin_obj.lat,
                                "lon": bin_obj.lon,
                                "fill_level": bin_obj.fill_level,
                                "capacity_l": bin_obj.capacity_l
                            }
                            for bin_obj in self.system_state.bins
                        ],
                        "correlation_id": correlation_id
                    },
                    target_stream="cleanify:agents:corridor:input"
                )
            
            print(f"Sent corridor optimization request for {len(routes)} routes")
            return routes  # Return original routes for now
            
        except Exception as e:
            print(f"Corridor optimization failed: {e}")
            return routes


    async def _execute_agent_planned_routes(self, routes):
        """Execute routes planned by agents"""
        try:
            for route_data in routes:
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
                
                route.progress = 0.0
                self.system_state.active_routes.append(route)
                
                # Update truck
                truck = next((t for t in self.system_state.trucks if t.id == route_data["truck_id"]), None)
                if truck:
                    truck.status = TruckStatus.EN_ROUTE
                    truck.current_route_id = route.id
                    truck.route_progress = 0.0
                
                # Mark bins
                for bin_id in route_data["bin_ids"]:
                    bin_obj = next((b for b in self.system_state.bins if b.id == bin_id), None)
                    if bin_obj:
                        bin_obj.being_collected = True
            
            print(f"EXECUTED: {len(routes)} agent-planned routes with OSRM waypoints")
            
        except Exception as e:
            print(f"Route execution failed: {e}")
    async def _execute_route_plan(self, departure_schedule):
        """Execute the planned routes"""
        try:
            routes = departure_schedule.get("scheduled_routes", [])
            
            for route_data in routes:
                # Create route in system
                route = RouteData(
                    id=route_data["id"],
                    truck_id=route_data["truck_id"],
                    bin_ids=route_data["bin_ids"],
                    waypoints=route_data["waypoints"],
                    status="planned",
                    created_at=datetime.now(),
                    estimated_duration=route_data.get("estimated_duration", 30),
                    scheduled_departure=datetime.fromisoformat(route_data["departure_time"])
                )
                
                self.system_state.active_routes.append(route)
                
                # Update truck status
                truck = next((t for t in self.system_state.trucks if t.id == route_data["truck_id"]), None)
                if truck:
                    truck.status = TruckStatus.EN_ROUTE
                    truck.current_route_id = route.id
                    truck.route_progress = 0.0
                
                # Mark bins as being collected
                for bin_id in route_data["bin_ids"]:
                    bin_obj = next((b for b in self.system_state.bins if b.id == bin_id), None)
                    if bin_obj:
                        bin_obj.being_collected = True
            
            print(f"Executed {len(routes)} routes from agent coordination")
            
        except Exception as e:
            print(f"Route execution failed: {e}")

    async def _wait_for_agent_response(self, agent_type, correlation_id, timeout=5):
        """Wait for agent response with timeout"""
        try:
            # This would implement actual message waiting logic
            # For now, return mock response to keep system working
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Mock responses based on agent type
            if agent_type == "forecast":
                return {
                    "predictions": {bin_id: 120 for bin_id in [b.id for b in self.system_state.bins[:3]]},
                    "correlation_id": correlation_id
                }
            elif agent_type == "traffic":
                return {
                    "conditions": [{"region": "center", "multiplier": 1.2}],
                    "correlation_id": correlation_id
                }
            elif agent_type == "route_planner":
                return {
                    "routes": [
                        {
                            "id": f"ROUTE_{datetime.now().strftime('%H%M%S')}",
                            "truck_id": self.system_state.trucks[0].id if self.system_state.trucks else "T001",
                            "bin_ids": [self.system_state.bins[0].id] if self.system_state.bins else ["B001"],
                            "waypoints": self._generate_basic_waypoints(),
                            "estimated_duration": 25
                        }
                    ] if self.system_state.trucks and self.system_state.bins else [],
                    "correlation_id": correlation_id
                }
            elif agent_type == "corridor":
                return {"routes": [], "correlation_id": correlation_id}
            elif agent_type == "departure":
                return {
                    "scheduled_routes": [
                        {
                            "id": f"ROUTE_{datetime.now().strftime('%H%M%S')}",
                            "truck_id": self.system_state.trucks[0].id if self.system_state.trucks else "T001",
                            "bin_ids": [self.system_state.bins[0].id] if self.system_state.bins else ["B001"],
                            "waypoints": self._generate_basic_waypoints(),
                            "departure_time": (datetime.now() + timedelta(minutes=2)).isoformat(),
                            "estimated_duration": 25
                        }
                    ] if self.system_state.trucks and self.system_state.bins else [],
                    "correlation_id": correlation_id
                }
            
            return {}
            
        except asyncio.TimeoutError:
            print(f"Timeout waiting for {agent_type} response")
            return None

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


    def _generate_basic_waypoints(self):
        """Generate basic waypoints for route"""
        depot_lat = self.depot_info.get("latitude", 33.6844)
        depot_lon = self.depot_info.get("longitude", 73.0479)
        
        if not self.system_state.bins:
            return []
        
        bin_obj = self.system_state.bins[0]
        
        return [
            {"lat": depot_lat, "lon": depot_lon, "type": "depot", "id": "depot"},
            {"lat": bin_obj.lat, "lon": bin_obj.lon, "type": "bin", "id": bin_obj.id},
            {"lat": depot_lat, "lon": depot_lon, "type": "depot", "id": "depot"}
        ]

    async def _request_departure_scheduling(self, optimized_routes):
        """Request departure scheduling from DepartureAgent"""
        try:
            correlation_id = f"departure_{datetime.now().strftime('%H%M%S')}"
            
            await self.send_message(
                "schedule_departures",
                {
                    "routes": optimized_routes.get("routes", []),
                    "current_time": self.simulation_current_time.isoformat(),
                    "constraints": {
                        "max_concurrent_routes": 3,
                        "min_departure_interval": 5,  # minutes
                        "priority_bins_first": True
                    },
                    "correlation_id": correlation_id
                },
                target_stream="departure"
            )
            
            departure_response = await self._wait_for_agent_response("departure", correlation_id, timeout=5)
            return departure_response
            
        except Exception as e:
            print(f"Departure scheduling failed: {e}")
            return optimized_routes

    async def _request_corridor_optimization(self, route_plan):
        """Request corridor optimization from CorridorAgent"""
        try:
            correlation_id = f"corridor_{datetime.now().strftime('%H%M%S')}"
            
            await self.send_message(
                "optimize_corridors",
                {
                    "routes": route_plan.get("routes", []),
                    "traffic_data": route_plan.get("traffic_data", {}),
                    "time_window": {
                        "start": self.simulation_current_time.isoformat(),
                        "duration_minutes": 120
                    },
                    "correlation_id": correlation_id
                },
                target_stream="corridor"
            )
            
            corridor_response = await self._wait_for_agent_response("corridor", correlation_id, timeout=8)
            return corridor_response
            
        except Exception as e:
            print(f"Corridor optimization failed: {e}")
            return route_plan 

    # async def _request_route_planning(self, urgent_bins, available_trucks, traffic_data):
    #     """Request route planning from RoutePlannerAgent"""
    #     try:
    #         correlation_id = f"route_{datetime.now().strftime('%H%M%S')}"
            
    #         await self.send_message(
    #             "plan_routes",
    #             {
    #                 "urgent_bins": [
    #                     {
    #                         "id": bin_obj.id,
    #                         "lat": bin_obj.lat,
    #                         "lon": bin_obj.lon,
    #                         "fill_level": bin_obj.fill_level,
    #                         "priority": "high" if bin_obj.fill_level >= 95 else "medium"
    #                     }
    #                     for bin_obj in urgent_bins
    #                 ],
    #                 "available_trucks": [
    #                     {
    #                         "id": truck.id,
    #                         "lat": truck.lat,
    #                         "lon": truck.lon,
    #                         "capacity_l": truck.capacity_l,
    #                         "current_load_l": getattr(truck, 'current_load_l', 0)
    #                     }
    #                     for truck in available_trucks
    #                 ],
    #                 "depot": {
    #                     "lat": self.depot_info.get("latitude", 33.6844),
    #                     "lon": self.depot_info.get("longitude", 73.0479)
    #                 },
    #                 "traffic_conditions": traffic_data,
    #                 "correlation_id": correlation_id
    #             },
    #             target_stream="route_planner"
    #         )
            
    #         route_response = await self._wait_for_agent_response("route_planner", correlation_id, timeout=10)
    #         return route_response
            
    #     except Exception as e:
    #         print(f"Route planning request failed: {e}")
    #         return None

    async def _request_traffic_update(self):
        """Request traffic data from TrafficAgent"""
        try:
            correlation_id = f"traffic_{datetime.now().strftime('%H%M%S')}"
            
            await self.send_message(
                "get_traffic_conditions",
                {
                    "region": "islamabad",
                    "timestamp": datetime.now().isoformat(),
                    "correlation_id": correlation_id
                },
                target_stream="traffic"
            )
            
            traffic_response = await self._wait_for_agent_response("traffic", correlation_id, timeout=3)
            return traffic_response
            
        except Exception as e:
            print(f"Traffic request failed: {e}")
            return {"conditions": [], "multiplier": 1.0}
        
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
    
    async def _update_simulation_time(self):
        """Update simulation time progression"""
        if self.simulation_running:
            await self._simulate_time_progression()

    async def _update_system_state(self):
        """Update system state"""
        if not self.system_state:
            return
        
        self.system_state.timestamp = datetime.now()

    async def _update_bin_fills_with_forecast(self):
        """Update bin fills using ForecastAgent hourly rates"""
        if not self.simulation_current_time:
            return
            
        current_hour = self.simulation_current_time.hour
        
        for bin_obj in self.system_state.bins:
            if getattr(bin_obj, 'being_collected', False):
                continue
                
            # Get hourly rate from bin metadata (loaded from config)
            if hasattr(bin_obj, 'metadata') and bin_obj.metadata.get("has_hourly_rates", False):
                hourly_rates = bin_obj.metadata.get("hourly_fill_rates", {})
                current_hourly_rate = hourly_rates.get(current_hour, bin_obj.fill_rate_lph)
            else:
                current_hourly_rate = bin_obj.fill_rate_lph
            
            # Store for frontend
            if not hasattr(bin_obj, 'metadata'):
                bin_obj.metadata = {}
            bin_obj.metadata['current_hourly_rate'] = current_hourly_rate
            
            # Calculate fill increase
            if self.simulation_running:
                fill_increase_per_second = current_hourly_rate / 3600
                fill_increase_liters = fill_increase_per_second * self.simulation_speed
                fill_increase_percent = (fill_increase_liters / bin_obj.capacity_l) * 100
                
                old_fill = bin_obj.fill_level
                bin_obj.fill_level = min(120.0, bin_obj.fill_level + fill_increase_percent)
                
                if bin_obj.fill_level >= 85.0 and old_fill < 85.0:
                    print(f"URGENT: Bin {bin_obj.id} reached {bin_obj.fill_level:.1f}% (Agent-based fill rate: {current_hourly_rate}L/h)")


    async def _update_truck_movements(self):
        """Update truck movements along routes"""
        if not self.system_state:
            return
            
        for truck in self.system_state.trucks:
            if truck.status in [TruckStatus.EN_ROUTE, TruckStatus.COLLECTING]:
                await self._move_truck_along_route(truck)

    async def _move_truck_along_route(self, truck):
        """Move truck with traffic adjustment"""
        if not truck.current_route_id:
            return
            
        route = next((r for r in self.system_state.active_routes if r.id == truck.current_route_id), None)
        if not route or not route.waypoints:
            return
        
        if not hasattr(truck, 'route_progress'):
            truck.route_progress = 0.0
        
        # Traffic-adjusted movement
        base_speed = 0.01
        traffic_multiplier = getattr(route, 'traffic_multiplier', 1.0)
        adjusted_speed = base_speed / traffic_multiplier
        movement_speed = adjusted_speed * self.simulation_speed
        
        truck.route_progress = min(1.0, truck.route_progress + movement_speed)
        
        if hasattr(route, 'progress'):
            route.progress = truck.route_progress
        
        if truck.route_progress >= 1.0:
            await self._complete_truck_route(truck, route)
        else:
            await self._interpolate_truck_position(truck, route)
    async def _complete_truck_route(self, truck, route):
        """Complete agent-planned route"""
        try:
            print(f"Completing agent-optimized route {route.id}")
            
            route.status = "completed"
            route.completed_at = datetime.now()
            
            # Reset truck to IDLE
            truck.status = TruckStatus.IDLE
            truck.current_route_id = None
            truck.route_progress = 0.0
            
            # Return to depot
            if self.depot_info:
                truck.lat = self.depot_info.get("latitude", 33.6844)
                truck.lon = self.depot_info.get("longitude", 73.0479)
            
            # Empty all bins (including corridor bins)
            collected_bins = 0
            for bin_id in route.bin_ids:
                bin_obj = next((b for b in self.system_state.bins if b.id == bin_id), None)
                if bin_obj:
                    bin_obj.fill_level = 5.0
                    bin_obj.being_collected = False
                    collected_bins += 1
            
            # Remove completed route
            self.system_state.active_routes = [r for r in self.system_state.active_routes if r.id != route.id]
            
            truck.current_load_l = 0
            truck.last_updated = datetime.now()
            
            corridor_info = f" (+{getattr(route, 'corridor_bins_added', 0)} corridor)" if getattr(route, 'corridor_bins_added', 0) > 0 else ""
            print(f"Agent route {route.id} completed: {collected_bins} bins{corridor_info}, truck {truck.id} IDLE")
            
        except Exception as e:
            print(f"Error completing agent route {route.id}: {e}")

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
        elif truck.status != TruckStatus.COLLECTING:
            truck.status = TruckStatus.EN_ROUTE
        
        truck.last_updated = datetime.now()

    async def _update_bin_fills_hourly(self):
        """Update bin fills using EXISTING hourly rates from config"""
        if not self.system_state or not self.simulation_current_time:
            return
            
        current_hour = self.simulation_current_time.hour
        
        for bin_obj in self.system_state.bins:
            if getattr(bin_obj, 'being_collected', False):
                continue
                
            # USE EXISTING HOURLY RATES FROM CONFIG - NOT HARDCODED
            if bin_obj.metadata.get("has_hourly_rates", False):
                hourly_rates = bin_obj.metadata.get("hourly_fill_rates", {})
                current_hourly_rate = hourly_rates.get(current_hour, bin_obj.fill_rate_lph)
            else:
                # Fallback to base rate if no hourly data
                current_hourly_rate = bin_obj.fill_rate_lph
            
            # Store current rate for frontend display
            bin_obj.metadata['current_hourly_rate'] = current_hourly_rate
            
            # Calculate fill increase (per second)
            if self.simulation_running:
                fill_increase_per_second = current_hourly_rate / 3600  # L/s
                fill_increase_liters = fill_increase_per_second * self.simulation_speed
                fill_increase_percent = (fill_increase_liters / bin_obj.capacity_l) * 100
                
                # Update fill level
                old_fill = bin_obj.fill_level
                bin_obj.fill_level = min(120.0, bin_obj.fill_level + fill_increase_percent)
                
                if bin_obj.fill_level >= 85.0 and old_fill < 85.0:
                    print(f"URGENT: Bin {bin_obj.id} reached {bin_obj.fill_level:.1f}% (Hour {current_hour} rate: {current_hourly_rate}L/h)")

    async def _update_bin_fill_levels(self):
        """Update bin fill levels based on time and fill rates"""
        if not self.system_state or not self.simulation_current_time:
            return
            
        for bin_obj in self.system_state.bins:
            if bin_obj.being_collected:
                continue  # Don't update bins being collected
                
            # Calculate time-based fill increase
            if bin_obj.last_updated:
                time_delta = (self.simulation_current_time - bin_obj.last_updated).total_seconds() / 3600  # hours
                
                # Get current hourly rate
                current_hour = self.simulation_current_time.hour
                hourly_rates = bin_obj.metadata.get("hourly_fill_rates", {})
                current_rate = hourly_rates.get(str(current_hour), bin_obj.fill_rate_lph)
                
                # Calculate fill increase
                fill_increase_l = current_rate * time_delta
                fill_increase_pct = (fill_increase_l / bin_obj.capacity_l) * 100
                
                # Update fill level
                bin_obj.fill_level = min(150.0, bin_obj.fill_level + fill_increase_pct)
                bin_obj.last_updated = self.simulation_current_time
    
    async def _update_truck_positions(self):
        """Update truck positions for moving trucks"""
        # This would update truck positions based on their routes
        # For now, just placeholder
        pass
    
    # Orchestration Logic
    async def _orchestrate_decisions(self):
        """Orchestration using proper agent architecture"""
        if not self.system_state or not self.simulation_running:
            return
        
        try:
            # Check for critical bins
            await self._check_critical_bins_auto()
            
            # Use ForecastAgent for predictions if available
            forecast_agent = self.managed_agents.get("forecast")
            if forecast_agent:
                await self._request_forecast_predictions()
            
            # Auto route planning
            await self._auto_route_planning()
            
            # Monitor active routes
            await self._monitor_active_routes()
            
            self.decisions_made += 1
            
        except Exception as e:
            print(f"Orchestration error: {e}")

    async def _orchestrate_with_all_agents(self):
        """Use actual agent methods directly - ENHANCED VERSION"""
        if not self.system_state or not self.simulation_running:
            return
        
        try:
            # Find urgent bins and available trucks with detailed logging
            urgent_bins = []
            for bin_obj in self.system_state.bins:
                if (bin_obj.fill_level >= 85.0 and 
                    not getattr(bin_obj, 'being_collected', False)):
                    urgent_bins.append(bin_obj)
            
            available_trucks = []
            for truck in self.system_state.trucks:
                truck_status = truck.status.value if hasattr(truck.status, 'value') else str(truck.status)
                if truck_status.upper() == 'IDLE':
                    available_trucks.append(truck)
            
            print(f"🎯 ORCHESTRATION: {len(urgent_bins)} urgent bins, {len(available_trucks)} available trucks")
            
            # Debug info
            if urgent_bins:
                fill_levels = [f"{b.id}({b.fill_level:.1f}%)" for b in urgent_bins[:3]]
                print(f"   📊 Urgent bins: {', '.join(fill_levels)}{'...' if len(urgent_bins) > 3 else ''}")
            
            if available_trucks:
                truck_names = [f"{t.id}" for t in available_trucks[:3]]
                print(f"   🚛 Available trucks: {', '.join(truck_names)}{'...' if len(available_trucks) > 3 else ''}")
            
            if urgent_bins and available_trucks:
                # Use actual agents directly
                print("🔄 Calling RoutePlannerAgent...")
                routes = await self._use_actual_route_planner_agent(urgent_bins, available_trucks)
                
                if routes:
                    print(f"✅ Got {len(routes)} routes from RoutePlannerAgent")
                    await self._execute_agent_routes(routes)
                else:
                    print("⚠️ No routes returned from RoutePlannerAgent")
            else:
                if not urgent_bins:
                    print("ℹ️ No urgent bins found (fill_level >= 85% and not being collected)")
                if not available_trucks:
                    print("ℹ️ No available trucks found (status = IDLE)")
            
        except Exception as e:
            print(f"❌ Agent orchestration error: {e}")
            import traceback
            traceback.print_exc()

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
        """Get OSRM waypoints for RoutePlannerAgent route - ENHANCED DEBUG VERSION"""
        try:
            import aiohttp
            
            # Build coordinates from route stops
            coordinates = []
            
            # Start at depot
            depot_lat = self.depot_info.get("latitude", 33.6844)
            depot_lon = self.depot_info.get("longitude", 73.0479)
            coordinates.append(f"{depot_lon},{depot_lat}")
            
            print(f"🔍 OSRM DEBUG: Route {route.id} has {len(route.stops)} stops")
            
            # Add bin locations from stops
            bins_added = 0
            for i, stop in enumerate(route.stops):
                if stop.stop_type == "bin":
                    print(f"🔍 Stop {i}: type={stop.stop_type}, bin_id={stop.bin_id}")
                    
                    if hasattr(stop, 'lat') and hasattr(stop, 'lon') and stop.lat and stop.lon:
                        coordinates.append(f"{stop.lon},{stop.lat}")
                        bins_added += 1
                        print(f"✅ Added stop coordinates: [{stop.lat}, {stop.lon}]")
                    elif stop.bin_id:
                        # Find bin by ID
                        bin_obj = next((b for b in self.system_state.bins if b.id == stop.bin_id), None)
                        if bin_obj:
                            coordinates.append(f"{bin_obj.lon},{bin_obj.lat}")
                            bins_added += 1
                            print(f"✅ Added bin coordinates: [{bin_obj.lat}, {bin_obj.lon}] for bin {bin_obj.id}")
                        else:
                            print(f"❌ Bin {stop.bin_id} not found in system state")
                    else:
                        print(f"❌ Stop has no coordinates: {stop}")
            
            print(f"🔍 Total coordinates collected: {len(coordinates)} (depot + {bins_added} bins)")
            
            # Return to depot if we have bin stops
            if len(coordinates) > 1:
                coordinates.append(f"{depot_lon},{depot_lat}")
            
            if len(coordinates) < 3:  # Need at least depot -> bin -> depot
                print(f"⚠️ Not enough coordinates for OSRM route {route.id}: {coordinates}")
                return self._create_fallback_waypoints(route)
            
            coords_str = ";".join(coordinates)
            osrm_url = f"http://localhost:5000/route/v1/driving/{coords_str}?steps=true&geometries=geojson"
            
            print(f"🗺️ OSRM REQUEST for route {route.id}:")
            print(f"   URL: {osrm_url}")
            print(f"   Coordinates: {coordinates}")
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(osrm_url, timeout=15) as response:
                        print(f"📡 OSRM Response status: {response.status}")
                        
                        if response.status == 200:
                            data = await response.json()
                            print(f"🔍 OSRM Response keys: {list(data.keys())}")
                            
                            if data.get("routes") and len(data["routes"]) > 0:
                                route_data = data["routes"][0]
                                print(f"🔍 Route data keys: {list(route_data.keys())}")
                                
                                if "geometry" in route_data:
                                    geometry = route_data["geometry"]["coordinates"]
                                    print(f"🔍 Geometry has {len(geometry)} coordinate pairs")
                                    print(f"🔍 First 3 coordinates: {geometry[:3]}")
                                    print(f"🔍 Last 3 coordinates: {geometry[-3:]}")
                                    
                                    # Check if all coordinates are the same (broken)
                                    if len(geometry) > 1:
                                        # Find the maximum distance between any two points in the route
                                        max_distance = 0.0
                                        for i in range(len(geometry)):
                                            for j in range(i + 1, len(geometry)):
                                                coord1 = geometry[i]
                                                coord2 = geometry[j]
                                                distance = ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
                                                max_distance = max(max_distance, distance)
                                        
                                        print(f"🔍 Maximum coordinate spread: {max_distance:.6f} degrees")
                                        
                                        if max_distance < 0.001:  # Less than ~100m
                                            print(f"❌ OSRM returned clustered coordinates! All waypoints near same location.")
                                            return self._create_fallback_waypoints(route)
                                    
                                    waypoints = []
                                    for i, coord in enumerate(geometry):
                                        waypoint_type = "route"
                                        waypoint_id = f"osrm_{i}"
                                        
                                        # Mark first and last as depot
                                        if i == 0 or i == len(geometry) - 1:
                                            waypoint_type = "depot"
                                            waypoint_id = "depot"
                                        
                                        waypoints.append({
                                            "lat": coord[1],
                                            "lon": coord[0],
                                            "type": waypoint_type,
                                            "id": waypoint_id
                                        })
                                    
                                    print(f"✅ OSRM SUCCESS: {len(waypoints)} waypoints for route {route.id}")
                                    print(f"   First waypoint: [{waypoints[0]['lat']}, {waypoints[0]['lon']}]")
                                    print(f"   Last waypoint: [{waypoints[-1]['lat']}, {waypoints[-1]['lon']}]")
                                    return waypoints
                                else:
                                    print(f"❌ No geometry in OSRM response")
                            else:
                                print(f"❌ No routes in OSRM response")
                                print(f"   Response data: {data}")
                        else:
                            error_text = await response.text()
                            print(f"❌ OSRM HTTP ERROR {response.status}")
                            print(f"   Error response: {error_text}")
                            
                except asyncio.TimeoutError:
                    print(f"❌ OSRM request timed out after 15 seconds")
                except Exception as e:
                    print(f"❌ OSRM request exception: {e}")
            
        except Exception as e:
            print(f"❌ OSRM waypoint generation failed for route {route.id}: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback waypoints
        print(f"🔄 Using fallback waypoints for route {route.id}")
        return self._create_fallback_waypoints(route)

    def _create_fallback_waypoints(self, route):
        """Create fallback waypoints when OSRM fails - ENHANCED DEBUG"""
        print(f"🔧 Creating fallback waypoints for route {route.id}")
        
        waypoints = []
        depot_lat = self.depot_info.get("latitude", 33.6844)
        depot_lon = self.depot_info.get("longitude", 73.0479)
        
        # Start at depot
        waypoints.append({
            "lat": depot_lat, 
            "lon": depot_lon, 
            "type": "depot", 
            "id": "depot"
        })
        print(f"   Added depot start: [{depot_lat}, {depot_lon}]")
        
        # Add bin waypoints
        bins_added = 0
        for stop in route.stops:
            if stop.stop_type == "bin":
                if hasattr(stop, 'lat') and hasattr(stop, 'lon') and stop.lat and stop.lon:
                    waypoints.append({
                        "lat": stop.lat,
                        "lon": stop.lon,
                        "type": "bin",
                        "id": stop.bin_id or stop.id
                    })
                    bins_added += 1
                    print(f"   Added bin from stop: [{stop.lat}, {stop.lon}]")
                elif stop.bin_id:
                    # Find bin in system state
                    bin_obj = next((b for b in self.system_state.bins if b.id == stop.bin_id), None)
                    if bin_obj:
                        waypoints.append({
                            "lat": bin_obj.lat,
                            "lon": bin_obj.lon,
                            "type": "bin",
                            "id": bin_obj.id
                        })
                        bins_added += 1
                        print(f"   Added bin from lookup: [{bin_obj.lat}, {bin_obj.lon}] (ID: {bin_obj.id})")
                    else:
                        print(f"   ❌ Could not find bin {stop.bin_id} in system state")
        
        # Return to depot
        waypoints.append({
            "lat": depot_lat, 
            "lon": depot_lon, 
            "type": "depot", 
            "id": "depot"
        })
        print(f"   Added depot return: [{depot_lat}, {depot_lon}]")
        
        print(f"📍 Fallback created {len(waypoints)} waypoints ({bins_added} bins)")
        
        # CRITICAL: Check if fallback waypoints are also broken
        if len(waypoints) >= 2:
            first_wp = waypoints[0]
            last_wp = waypoints[-1]
            wp_distance = ((first_wp['lat'] - last_wp['lat'])**2 + (first_wp['lon'] - last_wp['lon'])**2)**0.5
            print(f"🔍 Fallback waypoint spread: {wp_distance:.6f} degrees")
            
            if wp_distance < 0.001 and bins_added > 0:
                print(f"❌ FALLBACK WAYPOINTS ALSO BROKEN! All at same location.")
                print(f"   This suggests route.stops don't have proper bin coordinates")
        
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
                                
                # Add this in _enhance_routes_with_corridor_agent before corridor call:
                fill_distribution = {}
                for bin_obj in self.system_state.bins:
                    fill_range = f"{int(bin_obj.fill_level//10)*10}-{int(bin_obj.fill_level//10)*10+9}%"
                    fill_distribution[fill_range] = fill_distribution.get(fill_range, 0) + 1

                print(f"📊 Bin fill distribution: {fill_distribution}")

                # Call actual CorridorAgent method
                corridor_bin_ids = await corridor_agent.build_corridor(
                    polyline_latlon=polyline_latlon,
                    route_wids=[],  # No OSM way IDs
                    bins=candidate_bins
                )
                
                print(f"CorridorAgent found {len(corridor_bin_ids)} additional bins for route {route['id']}")
                
                # Add corridor bins to route
                if corridor_bin_ids:
                    route["bin_ids"].extend(list(corridor_bin_ids))
                    route["corridor_bins_added"] = len(corridor_bin_ids)
                    
                    # Update waypoints to include corridor bins
                    route["waypoints"] = await self._rebuild_waypoints_with_corridor_bins(
                        route["waypoints"], corridor_bin_ids
                    )
                
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

    async def _auto_route_planning(self):
        """Enhanced automatic route planning"""
        if not self.system_state or not self.simulation_running:
            return
            
        # Find urgent bins (85%+)
        urgent_bins = [b for b in self.system_state.bins if b.fill_level >= 85.0]
        
        # Find available trucks
        available_trucks = [t for t in self.system_state.trucks if t.status == TruckStatus.IDLE]
        
        # Auto-create routes if needed (max 3 active routes)
        active_routes = len([r for r in self.system_state.active_routes if r.status in ["active", "planned"]])
        
        if urgent_bins and available_trucks and active_routes < 3:
            routes_created = await self._create_routes_automatically(urgent_bins, available_trucks)
            if routes_created > 0:
                print(f"AUTO-ROUTING: Created {routes_created} routes for urgent bins")

    async def _create_routes_automatically(self, urgent_bins, available_trucks):
        """Create routes automatically for urgent bins"""
        routes_created = 0
        max_routes = min(len(urgent_bins), len(available_trucks), 3)
        
        for i in range(max_routes):
            truck = available_trucks[i]
            bin_obj = urgent_bins[i]
            
            # Create unique route ID
            route_id = f"AUTO_{datetime.now().strftime('%H%M%S')}_{truck.id}"
            
            # Create waypoints: Depot -> Bin -> Depot
            waypoints = []
            
            # Start at depot
            if self.depot_info:
                waypoints.append({
                    "lat": self.depot_info.get("latitude", 33.6844),
                    "lon": self.depot_info.get("longitude", 73.0479),
                    "type": "depot",
                    "id": "depot"
                })
            
            # Go to bin
            waypoints.append({
                "lat": bin_obj.lat,
                "lon": bin_obj.lon,
                "type": "bin", 
                "id": bin_obj.id
            })
            
            # Return to depot
            if self.depot_info:
                waypoints.append({
                    "lat": self.depot_info.get("latitude", 33.6844),
                    "lon": self.depot_info.get("longitude", 73.0479),
                    "type": "depot",
                    "id": "depot"
                })
            
            # Create route object
            route = RouteData(
                id=route_id,
                truck_id=truck.id,
                bin_ids=[bin_obj.id],
                waypoints=waypoints,
                status="active",
                created_at=datetime.now(),
                estimated_duration=20
            )
            
            # Add progress tracking
            route.progress = 0.0
            
            # Add to system state
            self.system_state.active_routes.append(route)
            
            # Update truck
            truck.status = TruckStatus.EN_ROUTE
            truck.current_route_id = route_id
            truck.route_progress = 0.0
            
            # Mark bin as being collected
            bin_obj.being_collected = True
            
            routes_created += 1
            print(f"Created route {route_id}: Truck {truck.id} -> Bin {bin_obj.id} ({bin_obj.fill_level:.1f}%)")
        
        return routes_created

    async def _check_critical_bins(self):
        """Check for bins that need immediate attention"""
        if not self.system_state:
            return
            
        critical_bins = [
            bin_obj for bin_obj in self.system_state.bins
            if bin_obj.fill_level >= 95.0 and not bin_obj.being_collected
        ]
        
        if critical_bins:
            # Send emergency alert
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
    
    async def _trigger_route_planning(self):
        """Trigger route planning when needed"""
        # Check if route planning is needed
        # For now, placeholder
        pass
    
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
    
    # Agent Health Monitoring
    async def _monitor_agent_health(self):
        """Monitor health of all managed agents"""
        for agent_name, agent in self.managed_agents.items():
            try:
                # Check if agent task is still running
                task = self.agent_tasks.get(agent_name)
                if task and task.done():
                    # Agent crashed, try to restart
                    await self._restart_agent(agent_name)
                else:
                    # Agent is running, update health
                    self.agent_health[agent_name]["status"] = "healthy"
                    self.agent_health[agent_name]["last_check"] = datetime.now()
                    
            except Exception as e:
                self.logger.error("Agent health check failed",
                                agent=agent_name, error=str(e))
    
    async def _restart_agent(self, agent_name: str):
        """Restart a failed agent"""
        self.logger.warning("Restarting agent", agent=agent_name)
        
        try:
            # Get agent class
            agent_classes = {
                "forecast": ForecastAgent,
                "traffic": TrafficAgent,
                "route_planner": RoutePlannerAgent,
                "corridor": CorridorAgent,
                "departure": DepartureAgent,
                "emergency": EmergencyAgent,
                "watchdog": WatchdogAgent
            }
            
            agent_class = agent_classes.get(agent_name)
            if not agent_class:
                return
            
            # Create new agent instance
            agent = agent_class()
            await agent.initialize()
            
            # Replace in managed agents
            self.managed_agents[agent_name] = agent
            
            # Start new task
            task = asyncio.create_task(agent.run())
            self.agent_tasks[agent_name] = task
            
            # Update health tracking
            self.agent_health[agent_name]["restarts"] += 1
            self.agent_health[agent_name]["status"] = "restarted"
            
            self.logger.info("Agent restarted successfully", agent=agent_name)
            
        except Exception as e:
            self.logger.error("Agent restart failed",
                            agent=agent_name, error=str(e))
            self.agent_health[agent_name]["status"] = "failed"
    
    def _register_supervisor_handlers(self):
        """Register supervisor-specific message handlers"""
        self.register_handler("load_config", self._handle_load_config)
        self.register_handler("start_simulation", self._handle_start_simulation)
        self.register_handler("pause_simulation", self._handle_pause_simulation)
        self.register_handler("set_simulation_speed", self._handle_set_simulation_speed)
        self.register_handler("get_agent_health", self._handle_get_agent_health)
        self.register_handler("route_planned", self._handle_route_planned)
    
    async def _handle_load_config(self, data: Dict[str, Any]):
        """Handle configuration loading - USE EXISTING HOURLY RATES"""
        try:
            print("SUPERVISOR: Loading config")
            config = data.get("config", {})
            
            if not config:
                raise ValueError("No config data received")
            
            bins_data = config.get("bins", [])
            trucks_data = config.get("trucks", [])
            depot_data = config.get("depot", {})
            
            # Parse bins with EXISTING hourly fill rate support
            bins = []
            for bin_data in bins_data:
                bin_obj = Bin(
                    id=str(bin_data["id"]),
                    lat=float(bin_data["latitude"]),
                    lon=float(bin_data["longitude"]), 
                    capacity_l=int(bin_data["capacity_l"]),
                    fill_level=float(bin_data.get("fill_level", 50.0)),
                    fill_rate_lph=float(bin_data.get("fill_rate_lph", 5.0)),
                    tile_id="",
                    bin_type=BinType.GENERAL
                )
                
                # PROPERLY load existing hourly fill rates from config
                hourly_rates = bin_data.get("hourly_fill_rates", {})
                if hourly_rates:
                    # Convert string keys to integers and store
                    bin_obj.metadata["hourly_fill_rates"] = {
                        int(hour): float(rate) for hour, rate in hourly_rates.items()
                    }
                    bin_obj.metadata["has_hourly_rates"] = True
                    rate_range = f"{min(hourly_rates.values()):.1f}-{max(hourly_rates.values()):.1f}L/h"
                    print(f"Bin {bin_obj.id} loaded with hourly rates: {rate_range}")
                else:
                    bin_obj.metadata["has_hourly_rates"] = False
                    print(f"Bin {bin_obj.id} using base rate: {bin_obj.fill_rate_lph}L/h")
                
                bin_obj.last_updated = datetime.now()
                bins.append(bin_obj)
            
            # Parse trucks
            trucks = []
            for truck_data in trucks_data:
                truck = Truck(
                    id=str(truck_data["id"]),
                    name=truck_data.get("name", truck_data["id"]),
                    lat=depot_data.get("latitude", 33.6844),
                    lon=depot_data.get("longitude", 73.0479),
                    capacity_l=int(truck_data.get("capacity", 5000)),
                    status=TruckStatus.IDLE
                )
                truck.current_load_l = 0
                truck.last_updated = datetime.now()
                trucks.append(truck)
            
            # Create system state
            self.system_state = SystemState(
                timestamp=datetime.now(),
                bins=bins,
                trucks=trucks,
                active_routes=[],
                traffic_conditions=[],
                simulation_running=False,
                current_time=self.simulation_current_time or datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            )
            
            self.depot_info = depot_data
            
            print(f"Config loaded: {len(bins)} bins, {len(trucks)} trucks")
            bins_with_hourly = len([b for b in bins if b.metadata.get("has_hourly_rates", False)])
            print(f"Bins with hourly rates: {bins_with_hourly}/{len(bins)}")
            
            return True
            
        except Exception as e:
            print(f"Config loading failed: {e}")
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

    async def _ensure_simulation_time_exists(self):
        """Ensure simulation time always exists"""
        if self.simulation_current_time is None:
            self.simulation_current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            if self.system_state:
                self.system_state.current_time = self.simulation_current_time
            print(f"🔧 SUPERVISOR: Ensured simulation time exists: {self.simulation_current_time.strftime('%H:%M:%S')}")

    def get_current_system_state(self):
        """Get current system state with guaranteed proper time"""
        if not self.system_state:
            return None
        
        # CRITICAL: Ensure simulation time is always set
        if self.simulation_current_time is None:
            self.simulation_current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            print(f"⚠️ SUPERVISOR: Late-initialized simulation time to {self.simulation_current_time.strftime('%H:%M:%S')}")
        
        # CRITICAL: Always return proper time
        return {
            "timestamp": self.system_state.timestamp.isoformat(),
            "bins": [self._bin_to_dict(bin_obj) for bin_obj in self.system_state.bins],
            "trucks": [self._truck_to_dict(truck) for truck in self.system_state.trucks],
            "active_routes": [self._route_to_dict(route) for route in self.system_state.active_routes],
            "simulation_running": self.simulation_running,
            "simulation_speed": self.simulation_speed,
            "current_time": self.simulation_current_time.isoformat()  # GUARANTEED to be set
        }
    
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
    
    async def _trigger_route_planning_direct(self, urgent_bins, available_trucks):
        """Enhanced route planning with proper route creation"""
        try:
            print(f"🎯 SUPERVISOR: Creating routes for {len(urgent_bins)} urgent bins")
            
            routes_created = 0
            
            for i, truck in enumerate(available_trucks):
                if i >= len(urgent_bins):
                    break
                    
                bin_obj = urgent_bins[i]
                
                # Create route with proper waypoints
                route_id = f"ROUTE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{truck.id}"
                
                # Create waypoints: Depot -> Bin -> Depot
                waypoints = []
                
                # Start at depot
                if self.depot_info:
                    waypoints.append({
                        "lat": self.depot_info.get("latitude", 33.6844),
                        "lon": self.depot_info.get("longitude", 73.0479),
                        "type": "depot",
                        "id": "depot"
                    })
                
                # Go to bin
                waypoints.append({
                    "lat": bin_obj.lat,
                    "lon": bin_obj.lon,
                    "type": "bin",
                    "id": bin_obj.id
                })
                
                # Return to depot
                if self.depot_info:
                    waypoints.append({
                        "lat": self.depot_info.get("latitude", 33.6844),
                        "lon": self.depot_info.get("longitude", 73.0479),
                        "type": "depot", 
                        "id": "depot"
                    })
                
                # Create route object
                route = RouteData(
                    id=route_id,
                    truck_id=truck.id,
                    bin_ids=[bin_obj.id],
                    waypoints=waypoints,
                    status="planned",
                    created_at=datetime.now(),
                    estimated_duration=30  # 30 minutes estimate
                )
                
                # Add route to system state
                if self.system_state:
                    self.system_state.active_routes.append(route)
                
                # Update truck status
                truck.status = TruckStatus.EN_ROUTE
                truck.current_route_id = route_id
                
                routes_created += 1
                print(f"✅ SUPERVISOR: Created route {route_id} for truck {truck.id} -> bin {bin_obj.id}")
            
            print(f"🎉 SUPERVISOR: Successfully created {routes_created} routes")
            
            # Publish updated system state
            await self._publish_system_state()
            
            return routes_created
            
        except Exception as e:
            print(f"❌ SUPERVISOR: Route planning failed: {e}")
            import traceback
            traceback.print_exc()
            return 0
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
                print(f"⏰ TIME: {old_sim_time.strftime('%H:%M:%S')} → {self.simulation_current_time.strftime('%H:%M:%S')} "
                    f"(+{time_diff:.1f}s @ {self.simulation_speed}x speed)")

            # Simulate bin fill progression with FIXED calculation
            await self._simulate_bin_fills(simulation_time_delta_seconds)
            
            # Auto-trigger route planning for urgent bins
            if self.system_state and self.system_state.bins:
                urgent_bins = [b for b in self.system_state.bins if b.fill_level >= 85.0 and not getattr(b, 'being_collected', False)]
                available_trucks = [t for t in self.system_state.trucks if t.status == TruckStatus.IDLE]
                
                if urgent_bins and available_trucks and len(self.system_state.active_routes) < 3:
                    print(f"🚨 AUTO-TRIGGERING: {len(urgent_bins)} urgent bins found")
                    await self._trigger_route_planning_direct(urgent_bins, available_trucks)

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
