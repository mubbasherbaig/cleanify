"""
Cleanify v2-alpha Emergency Agent
Monitors for critical conditions and coordinates emergency responses
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import json

from .base import AgentBase
from core.models import Bin, Truck, Priority, TruckStatus
from core.settings import get_settings


class EmergencyEvent:
    """Emergency event data structure"""
    
    def __init__(self, event_id: str, event_type: str, severity: str, 
                 description: str, affected_bins: List[str] = None,
                 affected_trucks: List[str] = None):
        self.event_id = event_id
        self.event_type = event_type  # 'overflow', 'truck_failure', 'system_overload'
        self.severity = severity      # 'low', 'medium', 'high', 'critical'
        self.description = description
        self.affected_bins = affected_bins or []
        self.affected_trucks = affected_trucks or []
        self.created_at = datetime.now()
        self.resolved_at: Optional[datetime] = None
        self.response_actions: List[str] = []
        self.is_active = True


class EmergencyAgent(AgentBase):
    """
    Emergency agent that monitors system health and coordinates emergency responses
    """
    
    def __init__(self):
        super().__init__("emergency", "emergency")
        
        # Emergency monitoring state
        self.active_emergencies: Dict[str, EmergencyEvent] = {}
        self.emergency_history: List[EmergencyEvent] = []
        self.alert_thresholds = {
            'bin_overflow': 98.0,      # % fill level
            'bin_critical': 120.0,     # % fill level
            'truck_overload': 95.0,    # % capacity utilization
            'system_overload': 80.0,   # % of trucks active
            'response_time': 300.0     # seconds for emergency response
        }
        
        # Emergency response protocols
        self.response_protocols = {
            'overflow': self._handle_overflow_emergency,
            'truck_failure': self._handle_truck_failure,
            'system_overload': self._handle_system_overload,
            'critical_bin': self._handle_critical_bin
        }
        
        # Settings
        self.settings = get_settings()
        
        # Performance metrics
        self.emergencies_detected = 0
        self.emergencies_resolved = 0
        self.average_response_time = 0.0
        self.false_alarms = 0
        
        # Register handlers
        self._register_emergency_handlers()
    
    async def initialize(self):
        """Initialize emergency agent"""
        self.logger.info("Initializing Emergency Agent")
        
        # Load emergency thresholds from settings
        if hasattr(self.settings, 'CRITICAL_BIN_THRESHOLD'):
            self.alert_thresholds['bin_overflow'] = self.settings.CRITICAL_BIN_THRESHOLD
        
        self.logger.info("Emergency agent initialized",
                        thresholds=self.alert_thresholds)
    
    async def main_loop(self):
        """Main emergency monitoring loop"""
        while self.running:
            try:
                # Monitor system state for emergencies
                await self._monitor_system_state()
                
                # Process active emergencies
                await self._process_active_emergencies()
                
                # Clean up resolved emergencies
                await self._cleanup_resolved_emergencies()
                
                # Sleep based on monitoring interval
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error("Error in emergency main loop", error=str(e))
                await asyncio.sleep(30)
    
    async def cleanup(self):
        """Cleanup emergency agent"""
        # Resolve all active emergencies
        for emergency in self.active_emergencies.values():
            emergency.is_active = False
            emergency.resolved_at = datetime.now()
            emergency.response_actions.append("System shutdown")
        
        self.logger.info("Emergency agent cleanup")
    
    async def monitor(self):
        """
        Main monitoring method exposed to other agents
        """
        await self._monitor_system_state()
    
    async def _monitor_system_state(self):
        """Monitor system state for emergency conditions"""
        
        # Request current system state
        await self.send_message(
            "request_system_state",
            {"requester": "emergency_agent"},
            target_stream="cleanify:system:requests"
        )
        
        # Note: In real implementation, would receive system state update
        # For now, we'll process any emergency alerts that come in
    
    async def _detect_bin_emergencies(self, bins: List[Bin]) -> List[EmergencyEvent]:
        """Detect emergency conditions in bins"""
        emergencies = []
        
        for bin_obj in bins:
            # Check for overflow conditions
            if bin_obj.fill_level >= self.alert_thresholds['bin_critical']:
                emergency = EmergencyEvent(
                    event_id=f"overflow_{bin_obj.id}_{int(datetime.now().timestamp())}",
                    event_type="overflow",
                    severity="critical",
                    description=f"Bin {bin_obj.id} critically full at {bin_obj.fill_level:.1f}%",
                    affected_bins=[bin_obj.id]
                )
                emergencies.append(emergency)
                
            elif bin_obj.fill_level >= self.alert_thresholds['bin_overflow']:
                emergency = EmergencyEvent(
                    event_id=f"critical_{bin_obj.id}_{int(datetime.now().timestamp())}",
                    event_type="critical_bin",
                    severity="high",
                    description=f"Bin {bin_obj.id} approaching overflow at {bin_obj.fill_level:.1f}%",
                    affected_bins=[bin_obj.id]
                )
                emergencies.append(emergency)
        
        return emergencies
    
    async def _detect_truck_emergencies(self, trucks: List[Truck]) -> List[EmergencyEvent]:
        """Detect emergency conditions in trucks"""
        emergencies = []
        
        for truck in trucks:
            # Check for truck overload
            utilization = truck.capacity_utilization()
            if utilization >= self.alert_thresholds['truck_overload']:
                emergency = EmergencyEvent(
                    event_id=f"overload_{truck.id}_{int(datetime.now().timestamp())}",
                    event_type="truck_failure",
                    severity="medium",
                    description=f"Truck {truck.id} overloaded at {utilization:.1f}%",
                    affected_trucks=[truck.id]
                )
                emergencies.append(emergency)
            
            # Check for truck failures (status)
            if truck.status == TruckStatus.MAINTENANCE:
                emergency = EmergencyEvent(
                    event_id=f"maintenance_{truck.id}_{int(datetime.now().timestamp())}",
                    event_type="truck_failure",
                    severity="medium",
                    description=f"Truck {truck.id} requires maintenance",
                    affected_trucks=[truck.id]
                )
                emergencies.append(emergency)
        
        return emergencies
    
    async def _detect_system_emergencies(self, trucks: List[Truck], bins: List[Bin]) -> List[EmergencyEvent]:
        """Detect system-wide emergency conditions"""
        emergencies = []
        
        if not trucks:
            return emergencies
        
        # Check system overload
        active_trucks = len([t for t in trucks if t.status == TruckStatus.EN_ROUTE])
        total_trucks = len(trucks)
        system_utilization = (active_trucks / total_trucks) * 100
        
        if system_utilization >= self.alert_thresholds['system_overload']:
            emergency = EmergencyEvent(
                event_id=f"system_overload_{int(datetime.now().timestamp())}",
                event_type="system_overload",
                severity="high",
                description=f"System overloaded: {active_trucks}/{total_trucks} trucks active",
                affected_trucks=[t.id for t in trucks if t.status == TruckStatus.EN_ROUTE]
            )
            emergencies.append(emergency)
        
        # Check for widespread urgent bins
        urgent_bins = [b for b in bins if b.is_urgent()]
        available_trucks = [t for t in trucks if t.is_available()]
        
        if len(urgent_bins) > len(available_trucks) * 3:  # 3+ urgent bins per available truck
            emergency = EmergencyEvent(
                event_id=f"capacity_shortage_{int(datetime.now().timestamp())}",
                event_type="system_overload",
                severity="medium",
                description=f"Capacity shortage: {len(urgent_bins)} urgent bins, {len(available_trucks)} available trucks",
                affected_bins=[b.id for b in urgent_bins]
            )
            emergencies.append(emergency)
        
        return emergencies
    
    async def _process_active_emergencies(self):
        """Process and respond to active emergencies"""
        
        for emergency_id, emergency in list(self.active_emergencies.items()):
            if emergency.is_active:
                try:
                    # Execute response protocol
                    if emergency.event_type in self.response_protocols:
                        await self.response_protocols[emergency.event_type](emergency)
                    
                    # Check if emergency should be auto-resolved
                    if await self._should_auto_resolve(emergency):
                        await self._resolve_emergency(emergency_id)
                
                except Exception as e:
                    self.logger.error("Error processing emergency", 
                                    emergency_id=emergency_id, error=str(e))
    
    async def _handle_overflow_emergency(self, emergency: EmergencyEvent):
        """Handle bin overflow emergency"""
        
        self.logger.warning("Processing overflow emergency",
                          emergency_id=emergency.event_id,
                          affected_bins=emergency.affected_bins)
        
        # Request immediate dispatch
        await self.send_message(
            "emergency_dispatch_request",
            {
                "emergency_id": emergency.event_id,
                "priority": "critical",
                "bin_ids": emergency.affected_bins,
                "reason": "overflow_emergency",
                "max_response_time_min": 30
            },
            target_stream="cleanify:agents:route_planner:input"
        )
        
        emergency.response_actions.append(f"Emergency dispatch requested at {datetime.now().isoformat()}")
    
    async def _handle_truck_failure(self, emergency: EmergencyEvent):
        """Handle truck failure emergency"""
        
        self.logger.warning("Processing truck failure emergency",
                          emergency_id=emergency.event_id,
                          affected_trucks=emergency.affected_trucks)
        
        # Notify supervisor about truck issues
        await self.send_message(
            "truck_failure_alert",
            {
                "emergency_id": emergency.event_id,
                "truck_ids": emergency.affected_trucks,
                "failure_type": emergency.description,
                "recommended_action": "reassign_routes"
            },
            target_stream="cleanify:agents:supervisor:input"
        )
        
        emergency.response_actions.append(f"Truck failure reported at {datetime.now().isoformat()}")
    
    async def _handle_system_overload(self, emergency: EmergencyEvent):
        """Handle system overload emergency"""
        
        self.logger.warning("Processing system overload emergency",
                          emergency_id=emergency.event_id)
        
        # Request system optimization
        await self.send_message(
            "system_optimization_request",
            {
                "emergency_id": emergency.event_id,
                "optimization_type": "emergency_load_balancing",
                "affected_bins": emergency.affected_bins,
                "affected_trucks": emergency.affected_trucks
            },
            target_stream="cleanify:agents:route_planner:input"
        )
        
        emergency.response_actions.append(f"System optimization requested at {datetime.now().isoformat()}")
    
    async def _handle_critical_bin(self, emergency: EmergencyEvent):
        """Handle critical bin emergency"""
        
        self.logger.warning("Processing critical bin emergency",
                          emergency_id=emergency.event_id,
                          affected_bins=emergency.affected_bins)
        
        # Request priority collection
        await self.send_message(
            "priority_collection_request",
            {
                "emergency_id": emergency.event_id,
                "bin_ids": emergency.affected_bins,
                "priority_level": "high",
                "max_response_time_min": 60
            },
            target_stream="cleanify:agents:route_planner:input"
        )
        
        emergency.response_actions.append(f"Priority collection requested at {datetime.now().isoformat()}")
    
    async def _should_auto_resolve(self, emergency: EmergencyEvent) -> bool:
        """Check if emergency should be automatically resolved"""
        
        # Auto-resolve based on age and type
        emergency_age = (datetime.now() - emergency.created_at).total_seconds()
        
        if emergency.event_type == "critical_bin" and emergency_age > 3600:  # 1 hour
            return True
        
        if emergency.event_type == "truck_failure" and emergency_age > 1800:  # 30 minutes
            return True
        
        # Check if emergency conditions still exist
        # In real implementation, would check current system state
        
        return False
    
    async def _resolve_emergency(self, emergency_id: str):
        """Resolve an emergency"""
        
        if emergency_id in self.active_emergencies:
            emergency = self.active_emergencies[emergency_id]
            emergency.is_active = False
            emergency.resolved_at = datetime.now()
            
            # Calculate response time
            response_time = (emergency.resolved_at - emergency.created_at).total_seconds()
            
            # Update metrics
            self.emergencies_resolved += 1
            self.average_response_time = (
                (self.average_response_time * (self.emergencies_resolved - 1) + response_time) / 
                self.emergencies_resolved
            )
            
            # Move to history
            self.emergency_history.append(emergency)
            del self.active_emergencies[emergency_id]
            
            # Publish resolution
            await self.send_message(
                "emergency_resolved",
                {
                    "emergency_id": emergency_id,
                    "event_type": emergency.event_type,
                    "response_time_sec": response_time,
                    "resolution_time": emergency.resolved_at.isoformat()
                },
                target_stream="cleanify:events"
            )
            
            self.logger.info("Emergency resolved",
                           emergency_id=emergency_id,
                           response_time=response_time)
    
    async def _cleanup_resolved_emergencies(self):
        """Clean up old resolved emergencies from history"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.emergency_history = [
            emergency for emergency in self.emergency_history
            if emergency.resolved_at and emergency.resolved_at > cutoff_time
        ]
    
    def _register_emergency_handlers(self):
        """Register emergency-specific message handlers"""
        self.register_handler("emergency_alert", self._handle_emergency_alert)
        self.register_handler("system_state_update", self._handle_system_state_update)
        self.register_handler("resolve_emergency", self._handle_resolve_emergency)
        self.register_handler("get_emergency_status", self._handle_get_emergency_status)
        self.register_handler("update_thresholds", self._handle_update_thresholds)
        self.register_handler("simulate_emergency", self._handle_simulate_emergency)
    
    async def _handle_emergency_alert(self, data: Dict[str, Any]):
        """Handle incoming emergency alert"""
        try:
            event_type = data.get("event_type", "unknown")
            severity = data.get("severity", "medium")
            description = data.get("description", "Emergency condition detected")
            affected_bins = data.get("affected_bins", [])
            affected_trucks = data.get("affected_trucks", [])
            
            # Create emergency event
            emergency = EmergencyEvent(
                event_id=f"{event_type}_{int(datetime.now().timestamp())}",
                event_type=event_type,
                severity=severity,
                description=description,
                affected_bins=affected_bins,
                affected_trucks=affected_trucks
            )
            
            # Store active emergency
            self.active_emergencies[emergency.event_id] = emergency
            self.emergencies_detected += 1
            
            # Immediate response
            if emergency.event_type in self.response_protocols:
                await self.response_protocols[emergency.event_type](emergency)
            
            self.logger.warning("Emergency alert processed",
                              emergency_id=emergency.event_id,
                              event_type=event_type,
                              severity=severity)
            
        except Exception as e:
            self.logger.error("Error handling emergency alert", error=str(e))
    
    async def _handle_system_state_update(self, data: Dict[str, Any]):
        """Handle system state update for monitoring"""
        try:
            bins_data = data.get("bins", [])
            trucks_data = data.get("trucks", [])
            
            # Parse bins and trucks
            bins = [self._parse_bin_data(b) for b in bins_data]
            trucks = [self._parse_truck_data(t) for t in trucks_data]
            
            # Detect emergencies
            all_emergencies = []
            all_emergencies.extend(await self._detect_bin_emergencies(bins))
            all_emergencies.extend(await self._detect_truck_emergencies(trucks))
            all_emergencies.extend(await self._detect_system_emergencies(trucks, bins))
            
            # Process new emergencies
            for emergency in all_emergencies:
                if emergency.event_id not in self.active_emergencies:
                    self.active_emergencies[emergency.event_id] = emergency
                    self.emergencies_detected += 1
                    
                    # Execute response protocol
                    if emergency.event_type in self.response_protocols:
                        await self.response_protocols[emergency.event_type](emergency)
            
            if all_emergencies:
                self.logger.info("Emergencies detected from state update",
                                new_emergencies=len(all_emergencies))
            
        except Exception as e:
            self.logger.error("Error processing system state update", error=str(e))
    
    async def _handle_resolve_emergency(self, data: Dict[str, Any]):
        """Handle emergency resolution request"""
        emergency_id = data.get("emergency_id")
        
        if emergency_id in self.active_emergencies:
            await self._resolve_emergency(emergency_id)
            
            await self.send_message(
                "emergency_resolution_confirmed",
                {
                    "emergency_id": emergency_id,
                    "correlation_id": data.get("correlation_id")
                }
            )
        else:
            await self.send_message(
                "emergency_resolution_failed",
                {
                    "error": f"Emergency {emergency_id} not found",
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_get_emergency_status(self, data: Dict[str, Any]):
        """Handle emergency status request"""
        
        active_emergencies_data = []
        for emergency in self.active_emergencies.values():
            active_emergencies_data.append({
                "emergency_id": emergency.event_id,
                "event_type": emergency.event_type,
                "severity": emergency.severity,
                "description": emergency.description,
                "affected_bins": emergency.affected_bins,
                "affected_trucks": emergency.affected_trucks,
                "created_at": emergency.created_at.isoformat(),
                "response_actions": emergency.response_actions,
                "age_seconds": (datetime.now() - emergency.created_at).total_seconds()
            })
        
        status = {
            "active_emergencies": active_emergencies_data,
            "active_count": len(self.active_emergencies),
            "total_detected": self.emergencies_detected,
            "total_resolved": self.emergencies_resolved,
            "average_response_time_sec": self.average_response_time,
            "false_alarms": self.false_alarms,
            "alert_thresholds": self.alert_thresholds,
            "correlation_id": data.get("correlation_id")
        }
        
        await self.send_message("emergency_status", status)
    
    async def _handle_update_thresholds(self, data: Dict[str, Any]):
        """Handle threshold update request"""
        try:
            new_thresholds = data.get("thresholds", {})
            
            for threshold_name, value in new_thresholds.items():
                if threshold_name in self.alert_thresholds:
                    self.alert_thresholds[threshold_name] = float(value)
            
            await self.send_message(
                "thresholds_updated",
                {
                    "updated_thresholds": self.alert_thresholds,
                    "correlation_id": data.get("correlation_id")
                }
            )
            
            self.logger.info("Emergency thresholds updated",
                           thresholds=self.alert_thresholds)
            
        except Exception as e:
            self.logger.error("Error updating thresholds", error=str(e))
    
    async def _handle_simulate_emergency(self, data: Dict[str, Any]):
        """Handle emergency simulation request"""
        try:
            simulation_type = data.get("type", "overflow")
            severity = data.get("severity", "medium")
            description = data.get("description", f"Simulated {simulation_type} emergency")
            
            # Create simulated emergency
            emergency = EmergencyEvent(
                event_id=f"simulation_{simulation_type}_{int(datetime.now().timestamp())}",
                event_type=simulation_type,
                severity=severity,
                description=f"SIMULATION: {description}",
                affected_bins=data.get("affected_bins", []),
                affected_trucks=data.get("affected_trucks", [])
            )
            
            # Process simulated emergency
            self.active_emergencies[emergency.event_id] = emergency
            self.emergencies_detected += 1
            
            if emergency.event_type in self.response_protocols:
                await self.response_protocols[emergency.event_type](emergency)
            
            await self.send_message(
                "emergency_simulation_started",
                {
                    "emergency_id": emergency.event_id,
                    "simulation_type": simulation_type,
                    "correlation_id": data.get("correlation_id")
                }
            )
            
            self.logger.info("Emergency simulation started",
                           emergency_id=emergency.event_id,
                           simulation_type=simulation_type)
            
        except Exception as e:
            self.logger.error("Error simulating emergency", error=str(e))
    
    def _parse_bin_data(self, bin_data: Dict[str, Any]) -> Bin:
        """Parse bin data from message"""
        return Bin(
            id=bin_data["id"],
            lat=bin_data.get("lat", 0.0),
            lon=bin_data.get("lon", 0.0),
            capacity_l=bin_data.get("capacity_l", 1000),
            fill_level=bin_data.get("fill_level", 50.0),
            fill_rate_lph=bin_data.get("fill_rate_lph", 5.0),
            tile_id=bin_data.get("tile_id", ""),
            threshold=bin_data.get("threshold", 85.0)
        )
    
    def _parse_truck_data(self, truck_data: Dict[str, Any]) -> Truck:
        """Parse truck data from message"""
        return Truck(
            id=truck_data["id"],
            name=truck_data.get("name", truck_data["id"]),
            capacity_l=truck_data.get("capacity_l", 5000),
            lat=truck_data.get("lat", 0.0),
            lon=truck_data.get("lon", 0.0),
            current_load_l=truck_data.get("current_load_l", 0),
            status=TruckStatus(truck_data.get("status", "idle"))
        )
    
    def get_emergency_metrics(self) -> Dict[str, Any]:
        """Get emergency agent performance metrics"""
        return {
            "detection": {
                "emergencies_detected": self.emergencies_detected,
                "emergencies_resolved": self.emergencies_resolved,
                "active_emergencies": len(self.active_emergencies),
                "false_alarms": self.false_alarms
            },
            "response": {
                "average_response_time_sec": self.average_response_time,
                "response_threshold_sec": self.alert_thresholds['response_time']
            },
            "thresholds": self.alert_thresholds,
            "emergency_types": {
                emergency_type: len([e for e in self.emergency_history 
                                   if e.event_type == emergency_type])
                for emergency_type in ["overflow", "truck_failure", "system_overload", "critical_bin"]
            }
        }