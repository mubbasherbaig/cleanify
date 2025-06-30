"""
Cleanify v2-alpha Departure Agent  
Implements intelligent departure timing with traffic-aware waiting logic
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from .base import AgentBase
from core.models import Truck, Bin, WaitingDecision
from core.settings import get_settings


class DepartureAgent(AgentBase):
    """
    Departure agent that evaluates optimal truck departure timing
    Implements exact waiting-time recipe from specification
    """
    
    def __init__(self):
        super().__init__("departure", "departure")
        
        # Departure decision state
        self.pending_evaluations: Dict[str, Dict[str, Any]] = {}
        self.recent_decisions: List[WaitingDecision] = []
        
        # Settings
        self.settings = get_settings()
        
        # Performance metrics
        self.evaluations_performed = 0
        self.go_now_decisions = 0
        self.wait_decisions = 0
        self.average_wait_time = 0.0
        
        # Register handlers
        self._register_departure_handlers()
    
    async def initialize(self):
        """Initialize departure agent"""
        self.logger.info("Initializing Departure Agent")
        
        self.logger.info("Departure agent initialized",
                        safety_pad_min=self.settings.waiting.SAFETY_PAD_MIN)
    
    async def main_loop(self):
        """Main departure evaluation loop"""
        while self.running:
            try:
                # Process pending evaluations
                await self._process_pending_evaluations()
                
                # Clean up old decisions
                await self._cleanup_old_decisions()
                
                # Sleep briefly
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error("Error in departure main loop", error=str(e))
                await asyncio.sleep(30)
    
    async def cleanup(self):
        """Cleanup departure agent"""
        self.logger.info("Departure agent cleanup")
    
    async def evaluate_wait(self, truck: Truck, 
                          bin_eta_map: Dict[str, float], 
                          delta_traffic: float) -> WaitingDecision:
        """
        EXACT implementation from specification:
        Evaluate whether truck should wait based on bin ETAs and traffic
        
        Tmin = min(predicted_minutes_to_120.values())
        allowed_wait = Tmin - settings.SAFETY_PAD_MIN
        if allowed_wait <= 0 or delta_traffic > allowed_wait:
            decision = "GO_NOW"
        else:
            decision = f"WAIT_{delta_traffic}_MIN"
        """
        
        try:
            # Step 1: Find minimum time to 120% across all assigned bins
            if not bin_eta_map:
                # No bins assigned - immediate go
                decision = WaitingDecision(
                    truck_id=truck.id,
                    decision="GO_NOW",
                    reason="No bins assigned",
                    traffic_delay_min=delta_traffic,
                    safety_pad_min=self.settings.waiting.SAFETY_PAD_MIN
                )
                
                self.go_now_decisions += 1
                return decision
            
            # Get minimum predicted time to 120%
            Tmin = min(bin_eta_map.values())
            
            # Step 2: Calculate allowed wait time
            allowed_wait = Tmin - self.settings.waiting.SAFETY_PAD_MIN
            
            # Step 3: Apply decision logic exactly as specified
            if allowed_wait <= 0 or delta_traffic > allowed_wait:
                decision_text = "GO_NOW"
                reason = self._generate_go_now_reason(allowed_wait, delta_traffic)
                self.go_now_decisions += 1
            else:
                decision_text = f"WAIT_{delta_traffic}_MIN"
                reason = f"Traffic delay {delta_traffic:.1f}min < allowed wait {allowed_wait:.1f}min"
                self.wait_decisions += 1
                
                # Update average wait time
                self.average_wait_time = (
                    (self.average_wait_time * (self.wait_decisions - 1) + delta_traffic) / 
                    self.wait_decisions
                )
            
            # Create decision object
            decision = WaitingDecision(
                truck_id=truck.id,
                decision=decision_text,
                reason=reason,
                traffic_delay_min=delta_traffic,
                safety_pad_min=self.settings.waiting.SAFETY_PAD_MIN,
                predicted_overflow_risk=self._calculate_overflow_risk(bin_eta_map, allowed_wait)
            )
            
            # Store decision
            self.recent_decisions.append(decision)
            self.evaluations_performed += 1
            
            self.logger.debug("Departure decision made",
                            truck_id=truck.id,
                            decision=decision_text,
                            tmin=Tmin,
                            allowed_wait=allowed_wait,
                            delta_traffic=delta_traffic)
            
            return decision
            
        except Exception as e:
            self.logger.error("Error evaluating wait decision", 
                            truck_id=truck.id, error=str(e))
            
            # Fallback to immediate departure
            return WaitingDecision(
                truck_id=truck.id,
                decision="GO_NOW",
                reason=f"Error in evaluation: {str(e)}",
                traffic_delay_min=delta_traffic,
                safety_pad_min=self.settings.waiting.SAFETY_PAD_MIN
            )
    
    def _generate_go_now_reason(self, allowed_wait: float, delta_traffic: float) -> str:
        """Generate descriptive reason for GO_NOW decision"""
        
        if allowed_wait <= 0:
            return f"No wait time available (allowed: {allowed_wait:.1f}min)"
        elif delta_traffic > allowed_wait:
            return f"Traffic delay {delta_traffic:.1f}min > allowed wait {allowed_wait:.1f}min"
        else:
            return "Immediate departure required"
    
    def _calculate_overflow_risk(self, bin_eta_map: Dict[str, float], allowed_wait: float) -> float:
        """Calculate risk of bin overflow if waiting"""
        
        if not bin_eta_map or allowed_wait <= 0:
            return 1.0  # High risk if no wait time
        
        # Count bins that might overflow during wait period
        risk_bins = sum(1 for eta in bin_eta_map.values() if eta <= allowed_wait)
        total_bins = len(bin_eta_map)
        
        return risk_bins / total_bins if total_bins > 0 else 0.0
    
    async def _process_pending_evaluations(self):
        """Process any pending departure evaluations"""
        
        completed_evaluations = []
        
        for eval_id, eval_data in self.pending_evaluations.items():
            try:
                # Check if evaluation has timed out
                eval_time = eval_data.get("timestamp", datetime.now())
                if (datetime.now() - eval_time).total_seconds() > 300:  # 5 minute timeout
                    completed_evaluations.append(eval_id)
                    continue
                
                # Process evaluation if we have all required data
                if self._has_required_data(eval_data):
                    await self._complete_evaluation(eval_id, eval_data)
                    completed_evaluations.append(eval_id)
                
            except Exception as e:
                self.logger.error("Error processing evaluation", 
                                eval_id=eval_id, error=str(e))
                completed_evaluations.append(eval_id)
        
        # Remove completed evaluations
        for eval_id in completed_evaluations:
            self.pending_evaluations.pop(eval_id, None)
    
    def _has_required_data(self, eval_data: Dict[str, Any]) -> bool:
        """Check if evaluation has all required data"""
        required_fields = ["truck", "bin_eta_map", "delta_traffic"]
        return all(field in eval_data for field in required_fields)
    
    async def _complete_evaluation(self, eval_id: str, eval_data: Dict[str, Any]):
        """Complete a pending evaluation"""
        
        truck_data = eval_data["truck"]
        bin_eta_map = eval_data["bin_eta_map"]
        delta_traffic = eval_data["delta_traffic"]
        
        # Create truck object
        truck = Truck(
            id=truck_data["id"],
            name=truck_data.get("name", truck_data["id"]),
            capacity_l=truck_data.get("capacity_l", 5000),
            lat=truck_data.get("lat", 0.0),
            lon=truck_data.get("lon", 0.0)
        )
        
        # Perform evaluation
        decision = await self.evaluate_wait(truck, bin_eta_map, delta_traffic)
        
        # Send response
        await self.send_message(
            "departure_decision",
            {
                "evaluation_id": eval_id,
                "truck_id": truck.id,
                "decision": decision.decision,
                "reason": decision.reason,
                "traffic_delay_min": decision.traffic_delay_min,
                "overflow_risk": decision.predicted_overflow_risk,
                "timestamp": decision.timestamp.isoformat(),
                "correlation_id": eval_data.get("correlation_id")
            }
        )
    
    async def _cleanup_old_decisions(self):
        """Remove old decisions to prevent memory buildup"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        self.recent_decisions = [
            decision for decision in self.recent_decisions
            if decision.timestamp > cutoff_time
        ]
    
    def _register_departure_handlers(self):
        """Register departure-specific message handlers"""
        self.register_handler("evaluate_departure", self._handle_evaluate_departure)
        self.register_handler("get_departure_stats", self._handle_get_departure_stats)
        self.register_handler("get_recent_decisions", self._handle_get_recent_decisions)
        self.register_handler("simulate_departure_scenario", self._handle_simulate_departure_scenario)
    
    async def _handle_evaluate_departure(self, data: Dict[str, Any]):
        """Handle departure evaluation request"""
        try:
            truck_data = data.get("truck", {})
            bin_eta_map = data.get("bin_eta_map", {})
            delta_traffic = data.get("delta_traffic", 0.0)
            
            if not truck_data:
                raise ValueError("Truck data is required")
            
            # Create truck object
            truck = Truck(
                id=truck_data["id"],
                name=truck_data.get("name", truck_data["id"]),
                capacity_l=truck_data.get("capacity_l", 5000),
                lat=truck_data.get("lat", 0.0),
                lon=truck_data.get("lon", 0.0)
            )
            
            # Perform evaluation
            decision = await self.evaluate_wait(truck, bin_eta_map, delta_traffic)
            
            # Send response
            await self.send_message(
                "departure_decision",
                {
                    "truck_id": truck.id,
                    "decision": decision.decision,
                    "reason": decision.reason,
                    "traffic_delay_min": decision.traffic_delay_min,
                    "safety_pad_min": decision.safety_pad_min,
                    "overflow_risk": decision.predicted_overflow_risk,
                    "timestamp": decision.timestamp.isoformat(),
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error handling departure evaluation", error=str(e))
            
            await self.send_message(
                "departure_evaluation_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    async def _handle_get_departure_stats(self, data: Dict[str, Any]):
        """Handle departure statistics request"""
        
        total_decisions = self.go_now_decisions + self.wait_decisions
        
        stats = {
            "total_evaluations": self.evaluations_performed,
            "go_now_decisions": self.go_now_decisions,
            "wait_decisions": self.wait_decisions,
            "go_now_percentage": (self.go_now_decisions / max(1, total_decisions)) * 100,
            "wait_percentage": (self.wait_decisions / max(1, total_decisions)) * 100,
            "average_wait_time_min": self.average_wait_time,
            "pending_evaluations": len(self.pending_evaluations),
            "recent_decisions_count": len(self.recent_decisions),
            "safety_pad_min": self.settings.waiting.SAFETY_PAD_MIN,
            "correlation_id": data.get("correlation_id")
        }
        
        await self.send_message("departure_stats", stats)
    
    async def _handle_get_recent_decisions(self, data: Dict[str, Any]):
        """Handle recent decisions request"""
        
        limit = data.get("limit", 10)
        recent_decisions = self.recent_decisions[-limit:]
        
        decisions_data = []
        for decision in recent_decisions:
            decisions_data.append({
                "truck_id": decision.truck_id,
                "decision": decision.decision,
                "reason": decision.reason,
                "traffic_delay_min": decision.traffic_delay_min,
                "safety_pad_min": decision.safety_pad_min,
                "overflow_risk": decision.predicted_overflow_risk,
                "timestamp": decision.timestamp.isoformat()
            })
        
        await self.send_message(
            "recent_departure_decisions",
            {
                "decisions": decisions_data,
                "total_recent": len(self.recent_decisions),
                "correlation_id": data.get("correlation_id")
            }
        )
    
    async def _handle_simulate_departure_scenario(self, data: Dict[str, Any]):
        """Handle departure scenario simulation"""
        try:
            scenarios = data.get("scenarios", [])
            results = []
            
            for scenario in scenarios:
                truck_data = scenario.get("truck", {})
                bin_eta_map = scenario.get("bin_eta_map", {})
                delta_traffic = scenario.get("delta_traffic", 0.0)
                
                # Create mock truck
                truck = Truck(
                    id=truck_data.get("id", "simulation_truck"),
                    name="Simulation Truck",
                    capacity_l=truck_data.get("capacity_l", 5000),
                    lat=0.0,
                    lon=0.0
                )
                
                # Evaluate scenario
                decision = await self.evaluate_wait(truck, bin_eta_map, delta_traffic)
                
                results.append({
                    "scenario_id": scenario.get("id", "unknown"),
                    "decision": decision.decision,
                    "reason": decision.reason,
                    "traffic_delay_min": decision.traffic_delay_min,
                    "overflow_risk": decision.predicted_overflow_risk,
                    "inputs": {
                        "bin_eta_map": bin_eta_map,
                        "delta_traffic": delta_traffic,
                        "tmin": min(bin_eta_map.values()) if bin_eta_map else 0,
                        "allowed_wait": (
                            min(bin_eta_map.values()) - self.settings.waiting.SAFETY_PAD_MIN
                            if bin_eta_map else 0
                        )
                    }
                })
            
            await self.send_message(
                "departure_simulation_results",
                {
                    "scenarios_tested": len(scenarios),
                    "results": results,
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error in departure scenario simulation", error=str(e))
            
            await self.send_message(
                "departure_simulation_error",
                {
                    "error": str(e),
                    "correlation_id": data.get("correlation_id")
                }
            )
    
    def get_departure_metrics(self) -> Dict[str, Any]:
        """Get departure agent performance metrics"""
        
        total_decisions = self.go_now_decisions + self.wait_decisions
        
        return {
            "evaluations_performed": self.evaluations_performed,
            "go_now_decisions": self.go_now_decisions,
            "wait_decisions": self.wait_decisions,
            "decision_distribution": {
                "go_now_percentage": (self.go_now_decisions / max(1, total_decisions)) * 100,
                "wait_percentage": (self.wait_decisions / max(1, total_decisions)) * 100
            },
            "timing_metrics": {
                "average_wait_time_min": self.average_wait_time,
                "safety_pad_min": self.settings.waiting.SAFETY_PAD_MIN
            },
            "queue_status": {
                "pending_evaluations": len(self.pending_evaluations),
                "recent_decisions": len(self.recent_decisions)
            }
        }
    
    async def batch_evaluate_departures(self, 
                                      trucks: List[Truck], 
                                      bin_etas: Dict[str, Dict[str, float]], 
                                      traffic_delays: Dict[str, float]) -> Dict[str, WaitingDecision]:
        """
        Evaluate departure decisions for multiple trucks
        """
        
        decisions = {}
        
        for truck in trucks:
            truck_bin_etas = bin_etas.get(truck.id, {})
            truck_traffic_delay = traffic_delays.get(truck.id, 0.0)
            
            decision = await self.evaluate_wait(truck, truck_bin_etas, truck_traffic_delay)
            decisions[truck.id] = decision
        
        self.logger.info("Batch departure evaluation completed",
                        trucks=len(trucks),
                        go_now=len([d for d in decisions.values() if "GO_NOW" in d.decision]),
                        wait=len([d for d in decisions.values() if "WAIT" in d.decision]))
        
        return decisions
    
    def get_wait_time_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for wait time optimization"""
        
        recommendations = {
            "current_config": {
                "safety_pad_min": self.settings.waiting.SAFETY_PAD_MIN
            },
            "performance_analysis": {
                "total_decisions": self.go_now_decisions + self.wait_decisions,
                "go_now_rate": self.go_now_decisions / max(1, self.evaluations_performed),
                "average_wait": self.average_wait_time
            },
            "recommendations": []
        }
        
        # Generate recommendations based on decision patterns
        go_now_rate = self.go_now_decisions / max(1, self.evaluations_performed)
        
        if go_now_rate > 0.8:
            recommendations["recommendations"].append({
                "type": "safety_pad",
                "message": "High GO_NOW rate suggests safety pad might be too conservative",
                "suggestion": "Consider reducing safety pad from {:.1f} to {:.1f} minutes".format(
                    self.settings.waiting.SAFETY_PAD_MIN,
                    max(0.5, self.settings.waiting.SAFETY_PAD_MIN - 0.5)
                )
            })
        elif go_now_rate < 0.3:
            recommendations["recommendations"].append({
                "type": "safety_pad", 
                "message": "Low GO_NOW rate suggests safety pad might be too aggressive",
                "suggestion": "Consider increasing safety pad from {:.1f} to {:.1f} minutes".format(
                    self.settings.waiting.SAFETY_PAD_MIN,
                    self.settings.waiting.SAFETY_PAD_MIN + 0.5
                )
            })
        
        if self.average_wait_time > 10.0:
            recommendations["recommendations"].append({
                "type": "wait_time",
                "message": "Average wait time is quite high",
                "suggestion": "Review traffic prediction accuracy or adjust wait thresholds"
            })
        
        return recommendations