# """
# Cleanify v2-alpha Supervisor Agent
# Orchestrates all other agents and manages system lifecycle
# """

# import asyncio
# import json
# from datetime import datetime, timedelta
# from typing import Dict, List, Any, Optional, Set
# import structlog

# from .base import AgentBase
# from .forecast import ForecastAgent
# from .traffic import TrafficAgent
# from .route_planner import RoutePlannerAgent
# from .corridor import CorridorAgent
# from .departure import DepartureAgent
# from .emergency import EmergencyAgent
# from .watchdog import WatchdogAgent
# from core.models import SystemState, Bin, Truck, Route
# from core.settings import get_settings


# logger = structlog.get_logger()


# class SupervisorAgent(AgentBase):
#     """
#     Supervisor agent that orchestrates the entire Cleanify v2 system
#     """
    
#     def __init__(self):
#         super().__init__("supervisor", "supervisor")
        
#         self.simulation_start_time = None  # When simulation started (real time)
#         self.simulation_current_time = None  # Current simulation time (starts at 8:00 AM)
#         self.simulation_running = False
#         self.simulation_speed = 1.0
#         self.last_update_time = None

#         # Agent management
#         self.managed_agents: Dict[str, AgentBase] = {}
#         self.agent_tasks: Dict[str, asyncio.Task] = {}
#         self.agent_health: Dict[str, Dict[str, Any]] = {}
        
#         # System state
#         self.system_state: Optional[SystemState] = None
#         # self.last_state_update = datetime.now()
#         # self.simulation_running = False
#         # self.simulation_speed = 1.0
        
#         # Configuration
#         self.settings = get_settings()
        
#         # Performance metrics
#         self.decisions_made = 0
#         self.routes_planned = 0
#         self.emergencies_handled = 0
        
#         # Register message handlers
#         self._register_supervisor_handlers()
    
#     async def initialize(self):
#         """Initialize supervisor and create managed agents"""
#         self.logger.info("Initializing Cleanify v2 Supervisor")
        
#         # Create agent instances
#         await self._create_agents()
        
#         # Initialize system state
#         await self._initialize_system_state()
        
#         self.logger.info("Supervisor initialization complete",
#                         agent_count=len(self.managed_agents))
    
#     async def main_loop(self):
#         """Main supervisor orchestration loop"""
#         self.logger.info("Starting supervisor main loop")
        
#         # Start all managed agents
#         await self._start_agents()
        
#         try:
#             while self.running:
#                 # Update system state
#                 await self._update_system_state()
                
#                 # Check agent health
#                 await self._check_agent_health()
                
#                 # Perform system orchestration
#                 await self._orchestrate_system()
                
#                 # Sleep based on decision interval
#                 await asyncio.sleep(self.settings.DECISION_INTERVAL_SEC)
                
#         except Exception as e:
#             self.logger.error("Error in supervisor main loop", error=str(e))
#             raise
    
#     async def cleanup(self):
#         """Cleanup supervisor and stop all agents"""
#         self.logger.info("Stopping all managed agents")
        
#         # Stop all agent tasks
#         for agent_id, task in self.agent_tasks.items():
#             self.logger.info(f"Stopping agent {agent_id}")
#             task.cancel()
        
#         # Wait for all tasks to complete
#         if self.agent_tasks:
#             await asyncio.gather(*self.agent_tasks.values(), return_exceptions=True)
        
#         # Cleanup managed agents
#         for agent in self.managed_agents.values():
#             try:
#                 await agent.shutdown()
#             except Exception as e:
#                 self.logger.warning(f"Error shutting down agent {agent.agent_id}", error=str(e))
        
#         self.logger.info("Supervisor cleanup complete")
    
#     async def _create_agents(self):
#         """Create all managed agent instances"""
#         self.logger.info("Creating managed agents")
        
#         # Core operational agents
#         self.managed_agents = {
#             "forecast": ForecastAgent(),
#             "traffic": TrafficAgent(), 
#             "route_planner": RoutePlannerAgent(),
#             "corridor": CorridorAgent(),
#             "departure": DepartureAgent(),
#             "emergency": EmergencyAgent(),
#             "watchdog": WatchdogAgent()
#         }
        
#         # Optional LLM advisor
#         if self.settings.llm.ENABLE_LLM_ADVISOR:
#             from .llm_advisor import LLMAdvisorAgent
#             self.managed_agents["llm_advisor"] = LLMAdvisorAgent()
#             self.logger.info("LLM Advisor agent included")
        
#         self.logger.info("Agent creation complete", 
#                         agent_count=len(self.managed_agents))
    
#     async def _start_agents(self):
#         """Start all managed agents"""
#         self.logger.info("Starting managed agents")
        
#         for agent_id, agent in self.managed_agents.items():
#             try:
#                 # Add startup delay to prevent resource conflicts
#                 await asyncio.sleep(self.settings.agents.AGENT_STARTUP_DELAY_SEC)
                
#                 # Start agent in background task
#                 task = asyncio.create_task(agent.run())
#                 self.agent_tasks[agent_id] = task
                
#                 self.logger.info(f"Started agent {agent_id}")
                
#             except Exception as e:
#                 self.logger.error(f"Failed to start agent {agent_id}", error=str(e))
#                 raise
        
#         # Wait for all agents to be ready
#         await asyncio.sleep(5.0)
#         self.logger.info("All agents started")
    
#     async def _initialize_system_state(self):
#         """Initialize empty system state with simulation time"""
#         # Set simulation start time to 8:00 AM today
#         from datetime import datetime, time
#         today = datetime.now().date()
#         simulation_start = datetime.combine(today, time(8, 0, 0))  # 8:00 AM
        
#         self.simulation_current_time = simulation_start
        
#         self.system_state = SystemState(
#             timestamp=datetime.now(),
#             bins=[],
#             trucks=[],
#             active_routes=[],
#             traffic_conditions=[],
#             simulation_running=False,
#             current_time=self.simulation_current_time,
#             simulation_speed=1.0
#         )
        
#         await self._publish_system_state()
    
#     async def _update_system_state(self):
#         """Update system state with simulation time and bin filling"""
#         if not self.system_state:
#             return
        
#         real_now = datetime.now()
#         self.system_state.timestamp = real_now
        
#         # Update simulation time based on speed
#         if self.simulation_running:
#             if self.last_update_time is None:
#                 self.last_update_time = real_now
            
#             # Calculate elapsed real time since last update
#             real_elapsed = (real_now - self.last_update_time).total_seconds()
            
#             # Apply simulation speed multiplier
#             sim_elapsed = real_elapsed * self.simulation_speed
            
#             # Update simulation time
#             if self.simulation_current_time:
#                 self.simulation_current_time += timedelta(seconds=sim_elapsed)
#                 self.system_state.current_time = self.simulation_current_time
            
#             # Update bin fill levels based on simulation time progression
#             await self._update_bin_fill_levels(sim_elapsed)
            
#             self.last_update_time = real_now
        
#         self.system_state.simulation_running = self.simulation_running
#         self.system_state.simulation_speed = self.simulation_speed
        
#         await self._publish_system_state()

#     async def _update_bin_fill_levels(self, sim_elapsed_seconds: float):
#         """Enhanced bin fill level update using hourly fill rates"""
#         if not self.system_state or not self.system_state.bins:
#             return
        
#         sim_elapsed_hours = sim_elapsed_seconds / 3600.0
#         current_hour = self.simulation_current_time.hour
        
#         print(f"🔧 FILL UPDATE: Simulation hour {current_hour}, elapsed {sim_elapsed_hours:.4f}h")
        
#         for bin_obj in self.system_state.bins:
#             if getattr(bin_obj, 'being_collected', False):
#                 continue
            
#             old_fill = bin_obj.fill_level
            
#             # Use hourly fill rates if available
#             if bin_obj.metadata.get("has_hourly_rates", False):
#                 hourly_rates = bin_obj.metadata.get("hourly_fill_rates", {})
#                 current_hour_rate = hourly_rates.get(current_hour, bin_obj.fill_rate_lph)
                
#                 # Calculate fill increase directly from hourly rate
#                 fill_increase_liters = current_hour_rate * sim_elapsed_hours
                
#             else:
#                 # Fallback to generic time multiplier
#                 hourly_rate = bin_obj.fill_rate_lph
#                 time_multiplier = self._get_generic_time_multiplier(current_hour, bin_obj.bin_type)
#                 effective_rate = hourly_rate * time_multiplier
#                 fill_increase_liters = effective_rate * sim_elapsed_hours
            
#             # Convert to percentage increase
#             if bin_obj.capacity_l > 0:
#                 fill_increase_percent = (fill_increase_liters / bin_obj.capacity_l) * 100
#                 new_fill_level = bin_obj.fill_level + fill_increase_percent
#                 bin_obj.fill_level = min(150.0, new_fill_level)  # Cap at 150% for overflow
                
#                 # Update timestamp
#                 bin_obj.last_updated = self.simulation_current_time
                
#                 # Debug logging for significant changes
#                 if fill_increase_percent > 0.1:  # Only log significant changes
#                     rate_used = current_hour_rate if bin_obj.metadata.get("has_hourly_rates") else f"{effective_rate:.2f} (calculated)"
#                     print(f"📈 {bin_obj.id}: {old_fill:.1f}% → {bin_obj.fill_level:.1f}% (+{fill_increase_percent:.2f}%) @ {rate_used}L/h")

#     def _get_generic_time_multiplier(self, hour: int, bin_type) -> float:
#         """Fallback time multiplier for bins without hourly rates"""
#         # Commercial/street bins
#         if 8 <= hour <= 18:  # Business hours
#             return 1.5
#         elif 19 <= hour <= 22:  # Evening
#             return 0.8
#         else:  # Night/early morning
#             return 0.3
#     async def _publish_system_state(self):
#         """Publish system state to all agents"""
#         if not self.system_state:
#             return
        
#         # Convert system state to dict for JSON serialization
#         state_dict = {
#             "timestamp": self.system_state.timestamp.isoformat(),
#             "bins": [self._bin_to_dict(bin_obj) for bin_obj in self.system_state.bins],
#             "trucks": [self._truck_to_dict(truck) for truck in self.system_state.trucks],
#             "active_routes": [self._route_to_dict(route) for route in self.system_state.active_routes],
#             "simulation_running": self.system_state.simulation_running,
#             "simulation_speed": self.system_state.simulation_speed,
#             "current_time": self.system_state.current_time.isoformat() if self.system_state.current_time else None
#         }
        
#         # Broadcast to all agents
#         await self.send_message(
#             "system_state_update",
#             state_dict,
#             target_stream="cleanify:system:state"
#         )
    
#     async def _check_agent_health(self):
#         """Check health of all managed agents"""
#         current_time = datetime.now()
        
#         for agent_id, agent in self.managed_agents.items():
#             health = agent.get_health_status()
#             self.agent_health[agent_id] = health
            
#             # Check if agent needs restart
#             if not health["healthy"]:
#                 self.logger.warning(f"Agent {agent_id} unhealthy", 
#                                   time_since_heartbeat=health["time_since_heartbeat_sec"])
                
#                 # Restart unhealthy agents
#                 if health["time_since_heartbeat_sec"] > 120:  # 2 minutes
#                     await self._restart_agent(agent_id)
    
#     async def _restart_agent(self, agent_id: str):
#         """Restart a failed agent"""
#         self.logger.info(f"Restarting agent {agent_id}")
        
#         try:
#             # Cancel existing task
#             if agent_id in self.agent_tasks:
#                 self.agent_tasks[agent_id].cancel()
#                 del self.agent_tasks[agent_id]
            
#             # Get agent reference
#             agent = self.managed_agents[agent_id]
            
#             # Shutdown and restart
#             await agent.shutdown()
#             await asyncio.sleep(2.0)
            
#             # Start new task
#             task = asyncio.create_task(agent.run())
#             self.agent_tasks[agent_id] = task
            
#             self.logger.info(f"Agent {agent_id} restarted successfully")
            
#         except Exception as e:
#             self.logger.error(f"Failed to restart agent {agent_id}", error=str(e))
    
#     async def _orchestrate_system(self):
#         """Perform high-level system orchestration"""
#         if not self.system_state:
#             return
        
#         # Get urgent bins that need collection
#         urgent_bins = self.system_state.urgent_bins()
#         available_trucks = self.system_state.available_trucks()
        
#         # If we have urgent bins and available trucks, request route planning
#         if urgent_bins and available_trucks:
#             await self._request_route_planning(urgent_bins, available_trucks)
        
#         # Check for emergency conditions
#         critical_bins = [b for b in urgent_bins if b.fill_level >= self.settings.CRITICAL_BIN_THRESHOLD]
#         if critical_bins:
#             await self._handle_critical_bins(critical_bins)
        
#         # Update metrics
#         self.decisions_made += 1
    
#     async def _request_route_planning(self, urgent_bins: List[Bin], available_trucks: List[Truck]):
#         """Request route planning from RoutePlannerAgent"""
#         request_data = {
#             "urgent_bins": [self._bin_to_dict(bin_obj) for bin_obj in urgent_bins],
#             "available_trucks": [self._truck_to_dict(truck) for truck in available_trucks],
#             "timestamp": datetime.now().isoformat()
#         }
        
#         await self.send_message(
#             "plan_routes_request",
#             request_data,
#             target_stream="cleanify:agents:route_planner:input"
#         )
        
#         self.logger.debug("Route planning requested",
#                          urgent_bins=len(urgent_bins),
#                          available_trucks=len(available_trucks))
    
#     async def _handle_critical_bins(self, critical_bins: List[Bin]):
#         """Handle critically full bins"""
#         emergency_data = {
#             "critical_bins": [self._bin_to_dict(bin_obj) for bin_obj in critical_bins],
#             "timestamp": datetime.now().isoformat(),
#             "urgency": "critical"
#         }
        
#         await self.send_message(
#             "emergency_alert",
#             emergency_data,
#             target_stream="cleanify:agents:emergency:input"
#         )
        
#         self.emergencies_handled += 1
#         self.logger.warning("Critical bins detected",
#                            count=len(critical_bins))
    
#     def _register_supervisor_handlers(self):
#         """Register supervisor-specific message handlers"""
#         self.register_handler("load_config", self._handle_load_config)
#         self.register_handler("start_simulation", self._handle_start_simulation)
#         self.register_handler("pause_simulation", self._handle_pause_simulation)
#         self.register_handler("set_simulation_speed", self._handle_set_simulation_speed)
#         self.register_handler("get_agent_health", self._handle_get_agent_health)
#         self.register_handler("route_planned", self._handle_route_planned)
    
#     async def _handle_load_config(self, data: Dict[str, Any]):
#         """Handle configuration loading - ENHANCED WITH HOURLY RATES"""
#         try:
#             print("🔧 SUPERVISOR: _handle_load_config called (Enhanced Version)")
#             config = data.get("config", {})
#             print(f"🔧 SUPERVISOR: Processing config with {len(config.get('bins', []))} bins, {len(config.get('trucks', []))} trucks")
            
#             if not config:
#                 raise ValueError("No config data received")
            
#             bins_data = config.get("bins", [])
#             trucks_data = config.get("trucks", [])
#             depot_data = config.get("depot", {})
            
#             # Parse bins with enhanced hourly fill rate support
#             bins = []
#             for i, bin_data in enumerate(bins_data):
#                 bin_obj = Bin(
#                     id=str(bin_data["id"]),
#                     lat=float(bin_data["latitude"]),
#                     lon=float(bin_data["longitude"]), 
#                     capacity_l=int(bin_data["capacity_l"]),
#                     fill_level=float(bin_data.get("fill_level", 50.0)),
#                     fill_rate_lph=float(bin_data.get("fill_rate_lph", 5.0)),
#                     tile_id="",
#                     bin_type=BinType.GENERAL  # Default, can be enhanced
#                 )
                
#                 # NEW: Store hourly fill rates in metadata
#                 hourly_rates = bin_data.get("hourly_fill_rates", {})
#                 if hourly_rates:
#                     # Convert string keys to integers and store
#                     bin_obj.metadata["hourly_fill_rates"] = {
#                         int(hour): float(rate) for hour, rate in hourly_rates.items()
#                     }
#                     bin_obj.metadata["has_hourly_rates"] = True
#                     print(f"✅ SUPERVISOR: Bin {bin_obj.id} loaded with hourly rates (range: {min(hourly_rates.values()):.1f}-{max(hourly_rates.values()):.1f}L/h)")
#                 else:
#                     bin_obj.metadata["has_hourly_rates"] = False
#                     print(f"⚠️ SUPERVISOR: Bin {bin_obj.id} using fallback rate: {bin_obj.fill_rate_lph}L/h")
                
#                 # Store additional metadata
#                 bin_obj.metadata["daily_fill_total"] = bin_data.get("daily_fill_total", 0)
#                 bin_obj.metadata["notes"] = bin_data.get("notes", "")
                
#                 bins.append(bin_obj)
            
#             # Parse trucks (unchanged)
#             trucks = []
#             depot_lat = float(depot_data["latitude"])
#             depot_lon = float(depot_data["longitude"])
            
#             for i, truck_data in enumerate(trucks_data):
#                 truck_obj = Truck(
#                     id=str(truck_data["id"]),
#                     name=str(truck_data["name"]),
#                     capacity_l=int(truck_data["capacity"]),
#                     lat=depot_lat,
#                     lon=depot_lon
#                 )
#                 trucks.append(truck_obj)
#                 print(f"✅ SUPERVISOR: Created truck {truck_obj.id}")
            
#             # Update system state
#             print(f"🔧 SUPERVISOR: Updating system state with {len(bins)} bins and {len(trucks)} trucks")
            
#             if self.system_state is None:
#                 print("🔧 SUPERVISOR: Creating new SystemState")
#                 from core.models import SystemState
#                 self.system_state = SystemState(
#                     timestamp=datetime.now(),
#                     bins=bins,
#                     trucks=trucks,
#                     active_routes=[],
#                     traffic_conditions=[],
#                     simulation_running=False,
#                     current_time=datetime.now()
#                 )
#             else:
#                 print("🔧 SUPERVISOR: Updating existing SystemState")
#                 self.system_state.bins = bins
#                 self.system_state.trucks = trucks
#                 self.system_state.timestamp = datetime.now()
            
#             # Count bins with hourly rates
#             hourly_bins = len([b for b in bins if b.metadata.get("has_hourly_rates", False)])
#             print(f"✅ SUPERVISOR: System state updated! Bins: {len(self.system_state.bins)} ({hourly_bins} with hourly rates), Trucks: {len(self.system_state.trucks)}")
            
#             # Safe message sending
#             if hasattr(self, 'redis_client') and self.redis_client is not None:
#                 print("🔧 SUPERVISOR: Publishing system state...")
#                 try:
#                     await self._publish_system_state()
#                     print("✅ SUPERVISOR: System state published successfully")
#                 except Exception as e:
#                     print(f"⚠️ SUPERVISOR: Failed to publish system state: {e}")
                    
#                 try:
#                     response_data = {
#                         "status": "success",
#                         "bins": len(bins),
#                         "trucks": len(trucks),
#                         "hourly_rate_bins": hourly_bins,
#                         "timestamp": datetime.now().isoformat()
#                     }
#                     await self.send_message("config_loaded", response_data)
#                     print("✅ SUPERVISOR: Response message sent")
#                 except Exception as e:
#                     print(f"⚠️ SUPERVISOR: Failed to send response message: {e}")
#             else:
#                 print("⚠️ SUPERVISOR: Redis client not available, skipping message publishing")
            
#             print(f"🎉 SUPERVISOR: Enhanced configuration loaded successfully!")
#             return True
            
#         except Exception as e:
#             print(f"❌ SUPERVISOR: Configuration loading failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return False

#     async def _publish_system_state(self):
#         """Publish system state to all agents - REDIS-SAFE VERSION"""
#         if not self.system_state:
#             print("⚠️ SUPERVISOR: No system state to publish")
#             return

#         if not hasattr(self, 'redis_client') or self.redis_client is None:
#             print("⚠️ SUPERVISOR: Redis client not available, cannot publish system state")
#             return

#         try:
#             # Convert system state to dict for JSON serialization
#             state_dict = {
#                 "timestamp": self.system_state.timestamp.isoformat(),
#                 "bins": [self._bin_to_dict(bin_obj) for bin_obj in self.system_state.bins],
#                 "trucks": [self._truck_to_dict(truck) for truck in self.system_state.trucks],
#                 "active_routes": [self._route_to_dict(route) for route in self.system_state.active_routes],
#                 "simulation_running": self.system_state.simulation_running,
#                 "simulation_speed": getattr(self.system_state, 'simulation_speed', 1.0),
#                 "current_time": self.system_state.current_time.isoformat() if self.system_state.current_time else None
#             }
            
#             print(f"🔧 SUPERVISOR: Publishing state with {len(state_dict['bins'])} bins, {len(state_dict['trucks'])} trucks")
            
#             # Broadcast to all agents
#             await self.send_message(
#                 "system_state_update",
#                 state_dict,
#                 target_stream="cleanify:system:state"
#             )
            
#             print("✅ SUPERVISOR: System state published successfully")
            
#         except Exception as e:
#             print(f"❌ SUPERVISOR: Failed to publish system state: {e}")
#             raise

#     # Add this method to provide direct access to system state for API
#     def get_current_system_state(self):
#         """Get current system state directly (for API access)"""
#         return self.system_state
    
#     def _get_time_multiplier(self, hour: int, bin_type) -> float:
#         """Get time-of-day multiplier for fill rate"""
#         # Business hours pattern for commercial bins
#         if str(bin_type).lower() in ['commercial', 'general']:
#             if 8 <= hour <= 18:  # Business hours
#                 return 1.5
#             elif 19 <= hour <= 22:  # Evening
#                 return 0.8
#             else:  # Night/early morning
#                 return 0.3
        
#         # Residential pattern
#         elif str(bin_type).lower() == 'residential':
#             if hour in [7, 8, 18, 19, 20]:  # Peak times
#                 return 2.0
#             elif 9 <= hour <= 17:  # Work hours (people away)
#                 return 0.4
#             else:  # Night
#                 return 0.2
        
#         # Default pattern
#         return 1.0

#     async def _handle_start_simulation(self, data: Dict[str, Any]):
#         """Handle simulation start with proper time initialization"""
#         # Initialize simulation timing
#         self.simulation_start_time = datetime.now()
#         self.last_update_time = self.simulation_start_time
        
#         # Set simulation time to 8:00 AM if not already set
#         if not self.simulation_current_time:
#             from datetime import time
#             today = datetime.now().date()
#             self.simulation_current_time = datetime.combine(today, time(8, 0, 0))
        
#         self.simulation_running = True
        
#         print(f"🚀 SIMULATION: Started at {self.simulation_current_time.strftime('%H:%M:%S')}")
#         print(f"   Real start time: {self.simulation_start_time.strftime('%H:%M:%S')}")
#         print(f"   Speed: {self.simulation_speed}x")
        
#         await self.send_message(
#             "simulation_started",
#             {
#                 "status": "started",
#                 "simulation_time": self.simulation_current_time.isoformat(),
#                 "real_time": self.simulation_start_time.isoformat(),
#                 "speed": self.simulation_speed
#             }
#         )
        
#         self.logger.info("Simulation started", 
#                         sim_time=self.simulation_current_time.strftime('%H:%M:%S'))
    
#     async def _handle_pause_simulation(self, data: Dict[str, Any]):
#         """Handle simulation pause"""
#         self.simulation_running = False
        
#         await self.send_message(
#             "simulation_paused",
#             {
#                 "status": "paused",
#                 "timestamp": datetime.now().isoformat()
#             }
#         )
        
#         self.logger.info("Simulation paused")
    
#     async def _handle_set_simulation_speed(self, data: Dict[str, Any]):
#         """Handle simulation speed change"""
#         new_speed = data.get("speed", 1.0)
#         old_speed = self.simulation_speed
        
#         self.simulation_speed = max(0.1, min(10.0, new_speed))
        
#         print(f"⚡ SIMULATION: Speed changed from {old_speed}x to {self.simulation_speed}x")
        
#         await self.send_message(
#             "simulation_speed_set",
#             {
#                 "speed": self.simulation_speed,
#                 "old_speed": old_speed,
#                 "simulation_time": self.simulation_current_time.isoformat() if self.simulation_current_time else None
#             }
#         )
        
#         self.logger.info("Simulation speed changed", 
#                         old_speed=old_speed, 
#                         new_speed=self.simulation_speed)
    
#     def get_simulation_status(self) -> Dict[str, Any]:
#         """Enhanced simulation status with hourly rate info"""
#         status = {
#             "running": self.simulation_running,
#             "speed": self.simulation_speed,
#             "current_time": self.simulation_current_time.isoformat() if self.simulation_current_time else None,
#             "current_hour": self.simulation_current_time.hour if self.simulation_current_time else None,
#             "start_time": self.simulation_start_time.isoformat() if self.simulation_start_time else None
#         }
        
#         # Add fill rate analysis
#         if self.system_state and self.system_state.bins:
#             bins_with_hourly = len([b for b in self.system_state.bins if b.metadata.get("has_hourly_rates", False)])
#             total_bins = len(self.system_state.bins)
            
#             # Current hour rates
#             if self.simulation_current_time:
#                 current_hour = self.simulation_current_time.hour
#                 current_rates = []
                
#                 for bin_obj in self.system_state.bins:
#                     if bin_obj.metadata.get("has_hourly_rates", False):
#                         hourly_rates = bin_obj.metadata.get("hourly_fill_rates", {})
#                         current_rates.append(hourly_rates.get(current_hour, 0))
                
#                 if current_rates:
#                     status.update({
#                         "current_hour_avg_rate": sum(current_rates) / len(current_rates),
#                         "current_hour_rate_range": [min(current_rates), max(current_rates)]
#                     })
            
#             status.update({
#                 "bins_with_hourly_rates": bins_with_hourly,
#                 "total_bins": total_bins,
#                 "hourly_rate_coverage": f"{bins_with_hourly}/{total_bins}"
#             })
        
#         return status

#     async def _handle_get_agent_health(self, data: Dict[str, Any]):
#         """Handle agent health request"""
#         health_data = {
#             "supervisor": self.get_health_status(),
#             "agents": self.agent_health,
#             "timestamp": datetime.now().isoformat(),
#             "correlation_id": data.get("correlation_id")
#         }
        
#         await self.send_message("agent_health_response", health_data)
    
#     async def _handle_route_planned(self, data: Dict[str, Any]):
#         """Handle route planning completion"""
#         self.routes_planned += 1
        
#         # Update system state with new route
#         route_data = data.get("route", {})
#         if route_data and self.system_state:
#             # Convert to Route object and add to active routes
#             # Implementation depends on route structure
#             pass
        
#         self.logger.info("Route planned", route_id=route_data.get("id"))
    
#     def _bin_to_dict(self, bin_obj: Bin) -> Dict[str, Any]:
#         """Enhanced bin conversion with hourly rate metadata"""
#         try:
#             result = {
#                 "id": bin_obj.id,
#                 "lat": bin_obj.lat,
#                 "lon": bin_obj.lon,
#                 "capacity_l": bin_obj.capacity_l,
#                 "fill_level": bin_obj.fill_level,
#                 "fill_rate_lph": bin_obj.fill_rate_lph,
#                 "tile_id": bin_obj.tile_id,
#                 "threshold": getattr(bin_obj, 'threshold', 85.0),
#                 "assigned_truck": getattr(bin_obj, 'assigned_truck', None),
#                 "being_collected": getattr(bin_obj, 'being_collected', False),
#                 "last_updated": bin_obj.last_updated.isoformat() if getattr(bin_obj, 'last_updated', None) else None
#             }
            
#             # Add hourly rate info if available
#             if bin_obj.metadata.get("has_hourly_rates", False):
#                 current_hour = self.simulation_current_time.hour if self.simulation_current_time else 8
#                 hourly_rates = bin_obj.metadata.get("hourly_fill_rates", {})
#                 current_rate = hourly_rates.get(current_hour, bin_obj.fill_rate_lph)
                
#                 result.update({
#                     "current_hourly_rate": current_rate,
#                     "has_hourly_rates": True,
#                     "daily_fill_total": bin_obj.metadata.get("daily_fill_total", 0),
#                     "notes": bin_obj.metadata.get("notes", "")
#                 })
#             else:
#                 result["has_hourly_rates"] = False
            
#             return result
#         except Exception as e:
#             print(f"❌ SUPERVISOR: _bin_to_dict failed: {e}")
#             raise
    
#     def _truck_to_dict(self, truck: Truck) -> Dict[str, Any]:
#         """Convert Truck object to dictionary - DEBUG VERSION"""
#         try:
#             result = {
#                 "id": truck.id,
#                 "name": truck.name,
#                 "capacity_l": truck.capacity_l,
#                 "lat": truck.lat,
#                 "lon": truck.lon,
#                 "current_load_l": getattr(truck, 'current_load_l', 0),
#                 "status": getattr(truck, 'status', 'idle').value if hasattr(getattr(truck, 'status', 'idle'), 'value') else str(getattr(truck, 'status', 'idle')),
#                 "speed_kmh": getattr(truck, 'speed_kmh', 30.0),
#                 "route_id": getattr(truck, 'route_id', None)
#             }
#             print(f"🔧 SUPERVISOR: _truck_to_dict result: {result}")
#             return result
#         except Exception as e:
#             print(f"❌ SUPERVISOR: _truck_to_dict failed: {e}")
#             raise
    
#     def _route_to_dict(self, route: Route) -> Dict[str, Any]:
#         """Convert Route object to dictionary"""
#         return {
#             "id": route.id,
#             "truck_id": route.truck_id,
#             "status": route.status.value,
#             "total_distance_km": route.total_distance_km,
#             "estimated_duration_min": route.estimated_duration_min,
#             "stops": [
#                 {
#                     "id": stop.id,
#                     "lat": stop.lat,
#                     "lon": stop.lon,
#                     "stop_type": stop.stop_type,
#                     "bin_id": stop.bin_id,
#                     "completed": stop.completed
#                 }
#                 for stop in route.stops
#             ]
#         }
    
#     async def get_system_metrics(self) -> Dict[str, Any]:
#         """Get system performance metrics"""
#         return {
#             "supervisor": {
#                 "decisions_made": self.decisions_made,
#                 "routes_planned": self.routes_planned,
#                 "emergencies_handled": self.emergencies_handled,
#                 "uptime_seconds": (datetime.now() - self.startup_time).total_seconds()
#             },
#             "system": {
#                 "simulation_running": self.simulation_running,
#                 "simulation_speed": self.simulation_speed,
#                 "agent_count": len(self.managed_agents),
#                 "healthy_agents": len([h for h in self.agent_health.values() if h.get("healthy", False)])
#             },
#             "state": {
#                 "bins": len(self.system_state.bins) if self.system_state else 0,
#                 "trucks": len(self.system_state.trucks) if self.system_state else 0,
#                 "active_routes": len(self.system_state.active_routes) if self.system_state else 0
#             }
#         }

"""
Cleanify v2-alpha Supervisor Agent - COMPLETE FIX
Orchestrates all other agents and manages system lifecycle
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import structlog

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
        """Enhanced main supervisor orchestration loop"""
        self.logger.info("Starting supervisor main loop")
        
        # Start all managed agents
        await self._start_agents()
        
        # Main loop with automatic decision making
        loop_counter = 0
        
        while self.running:
            try:
                loop_counter += 1
                
                # Update simulation time
                await self._update_simulation_time()
                
                # Auto orchestrate decisions every 5 seconds
                if loop_counter % 5 == 0:
                    await self._orchestrate_decisions()
                
                # Update system state
                await self._update_system_state()
                
                # Publish state to agents
                await self._publish_system_state()
                
                # Monitor agent health every 30 seconds
                if loop_counter % 30 == 0:
                    await self._monitor_agent_health()
                
                # Sleep based on simulation speed
                await asyncio.sleep(1.0 / self.simulation_speed)
                
            except Exception as e:
                self.logger.error("Supervisor main loop error", error=str(e))
                await asyncio.sleep(1.0)
    
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
        """Initialize empty system state"""
        self.system_state = SystemState(
            timestamp=datetime.now(),
            bins=[],
            trucks=[],
            active_routes=[],
            traffic_conditions=[],
            simulation_running=False,
            current_time=datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        )
        self.logger.info("System state initialized")
    
    async def _update_simulation_time(self):
        """Enhanced simulation time update - FIXED VERSION"""
        if not self.simulation_running:
            return
            
        current_real_time = datetime.now()
        
        # Initialize simulation time if not set
        if self.simulation_current_time is None:
            self.simulation_current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            print(f"🕐 SUPERVISOR: Initialized simulation time to {self.simulation_current_time.strftime('%H:%M:%S')}")
        
        if self.last_update_time is None:
            self.last_update_time = current_real_time
            return
        
        # Calculate time delta
        real_delta = (current_real_time - self.last_update_time).total_seconds()
        sim_delta = real_delta * self.simulation_speed
        
        # Update simulation time
        self.simulation_current_time += timedelta(seconds=sim_delta)
        
        # Update system state current time
        if self.system_state:
            self.system_state.current_time = self.simulation_current_time
            self.system_state.simulation_running = self.simulation_running
        
        self.last_update_time = current_real_time

    async def _update_system_state(self):
        """Update system state with latest data"""
        if not self.system_state:
            return
        
        # Update bin fill levels based on simulation time
        await self._update_bin_fill_levels()
        
        # Update truck positions if they're moving
        await self._update_truck_positions()
        
        # Update system timestamp
        self.system_state.timestamp = datetime.now()
    
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
        """Main orchestration logic - AUTOMATIC VERSION"""
        if not self.system_state or not self.simulation_running:
            return
        
        try:
            # Check for critical bins (auto emergency response)
            await self._check_critical_bins_auto()
            
            # Auto route planning for urgent bins
            await self._auto_route_planning()
            
            # Monitor active routes
            await self._monitor_active_routes()
            
            self.decisions_made += 1
            
        except Exception as e:
            print(f"❌ SUPERVISOR: Orchestration error: {e}")
    
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
        """Automatic route planning when conditions are met"""
        try:
            # Find urgent bins (85%+) not being collected
            urgent_bins = [
                bin_obj for bin_obj in self.system_state.bins 
                if bin_obj.fill_level >= 85.0 and not bin_obj.being_collected
            ]
            
            # Find available trucks
            available_trucks = [
                truck for truck in self.system_state.trucks 
                if truck.status == TruckStatus.IDLE
            ]
            
            if urgent_bins and available_trucks:
                print(f"🤖 SUPERVISOR: AUTO-PLANNING routes for {len(urgent_bins)} urgent bins")
                
                # Auto-assign trucks to bins (greedy assignment)
                routes_created = 0
                for i, truck in enumerate(available_trucks):
                    if i < len(urgent_bins):
                        bin_obj = urgent_bins[i]
                        
                        # Create route automatically
                        route_id = f"AUTO_ROUTE_{truck.id}_{int(datetime.now().timestamp())}"
                        
                        # Update truck state
                        truck.status = TruckStatus.EN_ROUTE
                        truck.route_id = route_id
                        
                        # Update bin state
                        bin_obj.being_collected = True
                        bin_obj.assigned_truck = truck.id
                        
                        routes_created += 1
                        print(f"✅ AUTO-ASSIGNED: {truck.id} → {bin_obj.id} ({bin_obj.fill_level:.1f}%)")
                
                self.routes_planned += routes_created
                print(f"🎯 SUPERVISOR: Auto-created {routes_created} routes")
                
        except Exception as e:
            print(f"❌ SUPERVISOR: Auto route planning failed: {e}")

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
        """Handle configuration loading - COMPLETE FIX"""
        try:
            print("🔧 SUPERVISOR: _handle_load_config called")
            config = data.get("config", {})
            print(f"🔧 SUPERVISOR: Processing config with {len(config.get('bins', []))} bins, {len(config.get('trucks', []))} trucks")
            
            if not config:
                raise ValueError("No config data received")
            
            bins_data = config.get("bins", [])
            trucks_data = config.get("trucks", [])
            depot_data = config.get("depot", {})
            
            print(f"🔧 SUPERVISOR: Raw data - bins: {len(bins_data)}, trucks: {len(trucks_data)}")
            
            # Parse bins with enhanced validation
            bins = []
            for i, bin_data in enumerate(bins_data):
                try:
                    print(f"🔧 Processing bin {i+1}: {bin_data}")
                    
                    # Validate required fields
                    if "id" not in bin_data:
                        print(f"❌ Bin {i+1} missing ID")
                        continue
                    
                    if "latitude" not in bin_data or "longitude" not in bin_data:
                        print(f"❌ Bin {bin_data.get('id', i+1)} missing coordinates")
                        continue
                    
                    # Create bin object with proper validation
                    bin_obj = Bin(
                        id=str(bin_data["id"]),
                        lat=float(bin_data["latitude"]),
                        lon=float(bin_data["longitude"]), 
                        capacity_l=int(bin_data.get("capacity_l", 100)),
                        fill_level=float(bin_data.get("fill_level", 50.0)),
                        fill_rate_lph=float(bin_data.get("fill_rate_lph", 5.0)),
                        tile_id="",
                        bin_type=BinType.GENERAL
                    )
                    
                    # Store hourly fill rates in metadata
                    hourly_rates = bin_data.get("hourly_fill_rates", {})
                    if hourly_rates:
                        bin_obj.metadata["hourly_fill_rates"] = {
                            str(k): float(v) for k, v in hourly_rates.items()
                        }
                        bin_obj.metadata["has_hourly_rates"] = True
                        bin_obj.metadata["daily_fill_total"] = bin_data.get("avg_daily_fill_L", 0)
                        bin_obj.metadata["notes"] = bin_data.get("notes", "")
                    else:
                        bin_obj.metadata["has_hourly_rates"] = False
                    
                    # Set last updated to current simulation time
                    bin_obj.last_updated = self.simulation_current_time or datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
                    
                    bins.append(bin_obj)
                    print(f"✅ Created bin {bin_obj.id} at ({bin_obj.lat}, {bin_obj.lon})")
                    
                except Exception as e:
                    print(f"❌ Failed to create bin {i+1}: {e}")
                    continue
            
            # Parse trucks with enhanced validation
            trucks = []
            for i, truck_data in enumerate(trucks_data):
                try:
                    print(f"🔧 Processing truck {i+1}: {truck_data}")
                    
                    if "id" not in truck_data:
                        print(f"❌ Truck {i+1} missing ID")
                        continue
                    
                    truck_obj = Truck(
                        id=str(truck_data["id"]),
                        name=truck_data.get("name", f"Truck {truck_data['id']}"),
                        capacity_l=int(truck_data.get("capacity", 5000)),
                        lat=float(depot_data.get("latitude", 33.6844)),
                        lon=float(depot_data.get("longitude", 73.0479)),
                        current_load_l=0,
                        status=TruckStatus.IDLE,
                        route_id=None
                    )
                    trucks.append(truck_obj)
                    print(f"✅ Created truck {truck_obj.id} ({truck_obj.name})")
                    
                except Exception as e:
                    print(f"❌ Failed to create truck {i+1}: {e}")
                    continue
            
            # Create enhanced system state
            self.system_state = SystemState(
                timestamp=datetime.now(),
                bins=bins,
                trucks=trucks,
                active_routes=[],
                traffic_conditions=[],
                simulation_running=False,
                current_time=self.simulation_current_time or datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            )
            
            print(f"✅ SUPERVISOR: Configuration loaded successfully")
            print(f"   - Created {len(bins)} bins")
            print(f"   - Created {len(trucks)} trucks")
            print(f"   - Depot: {depot_data.get('name', 'Unknown')}")
            
            # Store depot info
            self.depot_info = depot_data
            
            # Log each bin for debugging
            for bin_obj in bins:
                print(f"   📍 Bin {bin_obj.id}: ({bin_obj.lat}, {bin_obj.lon}) - {bin_obj.fill_level}%")
            
            return True
            
        except Exception as e:
            print(f"❌ SUPERVISOR: Configuration loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _handle_start_simulation(self, data: Dict[str, Any]):
        """Handle simulation start"""
        try:
            print("🚀 SUPERVISOR: Starting simulation")
            
            self.simulation_running = True
            self.simulation_start_time = datetime.now()
            
            # Set simulation start time to 8:00 AM
            self.simulation_current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            self.last_update_time = datetime.now()
            
            if self.system_state:
                self.system_state.simulation_running = True
                self.system_state.current_time = self.simulation_current_time
            
            print(f"✅ SUPERVISOR: Simulation started at {self.simulation_current_time}")
            self.logger.info("Simulation started")
            
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
        """Handle simulation speed change - FIXED"""
        try:
            new_speed = data.get("speed", 1.0)
            self.simulation_speed = max(0.1, min(10.0, new_speed))
            
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
        """Publish system state to all agents"""
        if not self.system_state:
            return

        if not hasattr(self, 'redis_client') or self.redis_client is None:
            return

        try:
            # Convert system state to dict for JSON serialization
            state_dict = {
                "timestamp": self.system_state.timestamp.isoformat(),
                "bins": [self._bin_to_dict(bin_obj) for bin_obj in self.system_state.bins],
                "trucks": [self._truck_to_dict(truck) for truck in self.system_state.trucks],
                "active_routes": [self._route_to_dict(route) for route in self.system_state.active_routes],
                "simulation_running": self.system_state.simulation_running,
                "simulation_speed": self.simulation_speed,
                "current_time": self.system_state.current_time.isoformat() if self.system_state.current_time else None
            }
            
            # Broadcast to all agents
            await self.send_message(
                "system_state_update",
                state_dict,
                target_stream="cleanify:system:state"
            )
            
        except Exception as e:
            print(f"❌ SUPERVISOR: Failed to publish system state: {e}")

    def get_current_system_state(self):
        """Get current system state with proper time"""
        if not self.system_state:
            return None
        
        # Ensure simulation time is set
        if self.simulation_current_time is None:
            self.simulation_current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        
        return {
            "timestamp": self.system_state.timestamp.isoformat(),
            "bins": [self._bin_to_dict(bin_obj) for bin_obj in self.system_state.bins],
            "trucks": [self._truck_to_dict(truck) for truck in self.system_state.trucks],
            "active_routes": [self._route_to_dict(route) for route in self.system_state.active_routes],
            "simulation_running": self.simulation_running,
            "simulation_speed": self.simulation_speed,  # Add this line
            "current_time": self.simulation_current_time.isoformat()  # Ensure this is always set
        }

    def _bin_to_dict(self, bin_obj: Bin) -> Dict[str, Any]:
        """Convert Bin object to dictionary"""
        try:
            result = {
                "id": bin_obj.id,
                "lat": bin_obj.lat,
                "lon": bin_obj.lon,
                "capacity_l": bin_obj.capacity_l,
                "fill_level": bin_obj.fill_level,
                "fill_rate_lph": bin_obj.fill_rate_lph,
                "tile_id": bin_obj.tile_id,
                "threshold": getattr(bin_obj, 'threshold', 85.0),
                "assigned_truck": getattr(bin_obj, 'assigned_truck', None),
                "being_collected": getattr(bin_obj, 'being_collected', False),
                "last_updated": bin_obj.last_updated.isoformat() if getattr(bin_obj, 'last_updated', None) else None
            }
            
            # Add hourly rate info if available
            if bin_obj.metadata.get("has_hourly_rates", False):
                current_hour = self.simulation_current_time.hour if self.simulation_current_time else 8
                hourly_rates = bin_obj.metadata.get("hourly_fill_rates", {})
                current_rate = hourly_rates.get(str(current_hour), bin_obj.fill_rate_lph)
                
                result.update({
                    "current_hourly_rate": current_rate,
                    "has_hourly_rates": True,
                    "daily_fill_total": bin_obj.metadata.get("daily_fill_total", 0),
                    "notes": bin_obj.metadata.get("notes", "")
                })
            else:
                result["has_hourly_rates"] = False
            
            return result
        except Exception as e:
            print(f"❌ SUPERVISOR: _bin_to_dict failed for bin {bin_obj.id}: {e}")
            # Return minimal safe dict
            return {
                "id": str(bin_obj.id),
                "lat": float(bin_obj.lat),
                "lon": float(bin_obj.lon),
                "capacity_l": int(bin_obj.capacity_l),
                "fill_level": float(bin_obj.fill_level),
                "fill_rate_lph": float(bin_obj.fill_rate_lph),
                "has_hourly_rates": False
            }
    
    def _truck_to_dict(self, truck: Truck) -> Dict[str, Any]:
        """Convert Truck object to dictionary"""
        try:
            result = {
                "id": truck.id,
                "name": truck.name,
                "capacity_l": truck.capacity_l,
                "lat": truck.lat,
                "lon": truck.lon,
                "current_load_l": getattr(truck, 'current_load_l', 0),
                "status": getattr(truck, 'status', 'idle').value if hasattr(getattr(truck, 'status', 'idle'), 'value') else str(getattr(truck, 'status', 'idle')),
                "route_id": getattr(truck, 'route_id', None),
                "speed_kmh": getattr(truck, 'speed_kmh', 0.0)
            }
            return result
        except Exception as e:
            print(f"❌ SUPERVISOR: _truck_to_dict failed: {e}")
            # Return minimal safe dict
            return {
                "id": str(truck.id),
                "name": str(truck.name),
                "capacity_l": int(truck.capacity_l),
                "lat": float(truck.lat),
                "lon": float(truck.lon),
                "current_load_l": 0,
                "status": "idle"
            }
    
    def _route_to_dict(self, route: Route) -> Dict[str, Any]:
        """Convert Route object to dictionary"""
        try:
            return {
                "id": route.id,
                "truck_id": route.truck_id,
                "status": route.status.value if hasattr(route.status, 'value') else str(route.status),
                "total_distance_km": route.total_distance_km,
                "estimated_duration_min": route.estimated_duration_min,
                "stops": len(route.stops),
                "completed_stops": route.completed_stops(),
                "created_at": route.created_at.isoformat() if route.created_at else None
            }
        except Exception as e:
            print(f"❌ SUPERVISOR: _route_to_dict failed: {e}")
            return {"id": "unknown", "status": "error"}
