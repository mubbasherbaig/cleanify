"""
Cleanify v2-alpha Watchdog Agent
Monitors system health and agent performance
"""

import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from .base import AgentBase
from core.settings import get_settings


class SystemHealthMetrics:
    """System health metrics container"""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_available_gb = 0.0
        self.disk_usage_percent = 0.0
        self.network_io = {"bytes_sent": 0, "bytes_recv": 0}
        self.process_count = 0
        self.load_average = 0.0


class AgentHealthStatus:
    """Agent health status container"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.last_heartbeat = None
        self.is_healthy = False
        self.response_time_ms = 0.0
        self.message_count = 0
        self.error_count = 0
        self.uptime_seconds = 0.0
        self.status = "unknown"


class WatchdogAgent(AgentBase):
    """
    Watchdog agent that monitors system and agent health
    """
    
    def __init__(self):
        super().__init__("watchdog", "watchdog")
        
        # Health monitoring state
        self.system_metrics_history: List[SystemHealthMetrics] = []
        self.agent_health: Dict[str, AgentHealthStatus] = {}
        self.health_check_interval = 30.0  # seconds
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 5000.0,
            'agent_timeout_sec': 120.0
        }
        
        # Performance tracking
        self.performance_history = {
            'system_checks': 0,
            'agent_checks': 0,
            'alerts_sent': 0,
            'restarts_initiated': 0
        }
        
        # Settings
        self.settings = get_settings()
        
        # Register handlers
        self._register_watchdog_handlers()
    
    async def initialize(self):
        """Initialize watchdog agent"""
        self.logger.info("Initializing Watchdog Agent")
        
        # Initialize known agents
        expected_agents = [
            "supervisor", "forecast", "traffic", "route_planner", 
            "corridor", "departure", "emergency"
        ]
        
        if self.settings.llm.ENABLE_LLM_ADVISOR:
            expected_agents.append("llm_advisor")
        
        for agent_id in expected_agents:
            self.agent_health[agent_id] = AgentHealthStatus(agent_id)
        
        self.logger.info("Watchdog agent initialized",
                        monitoring_agents=len(expected_agents))
    
    async def main_loop(self):
        """Main watchdog monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check agent health
                await self._check_agent_health()
                
                # Analyze health and send alerts
                await self._analyze_health_and_alert()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
                # Sleep until next check
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error("Error in watchdog main loop", error=str(e))
                await asyncio.sleep(60)
    
    async def cleanup(self):
        """Cleanup watchdog agent"""
        self.logger.info("Watchdog agent cleanup")
    
    async def run(self):
        """Override run to implement health monitoring"""
        if not self.running:
            await self.startup()
        
        self.logger.info("Starting watchdog monitoring")
        
        # Start background tasks
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        message_task = asyncio.create_task(self._message_loop())
        health_task = asyncio.create_task(self._health_monitoring_loop())
        
        try:
            # Run main monitoring loop
            await self.main_loop()
            
        except Exception as e:
            self.logger.error("Error in watchdog run", error=str(e))
            raise
        finally:
            # Cancel background tasks
            heartbeat_task.cancel()
            message_task.cancel()
            health_task.cancel()
            
            # Wait for cleanup
            await asyncio.gather(heartbeat_task, message_task, health_task, return_exceptions=True)
            await self.shutdown()
    
    async def _health_monitoring_loop(self):
        """Dedicated health monitoring loop"""
        while self.running:
            try:
                # Ping all agents
                await self._ping_all_agents()
                
                # Check for unresponsive agents
                await self._check_unresponsive_agents()
                
                # Sleep between health checks
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            metrics = SystemHealthMetrics()
            
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            }
            
            # Process metrics
            metrics.process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                metrics.load_average = psutil.getloadavg()[0]
            except AttributeError:
                metrics.load_average = 0.0  # Windows doesn't have load average
            
            # Store metrics
            self.system_metrics_history.append(metrics)
            self.performance_history['system_checks'] += 1
            
            self.logger.debug("System metrics collected",
                            cpu=f"{metrics.cpu_percent:.1f}%",
                            memory=f"{metrics.memory_percent:.1f}%",
                            disk=f"{metrics.disk_usage_percent:.1f}%")
            
        except Exception as e:
            self.logger.error("Error collecting system metrics", error=str(e))
    
    async def _check_agent_health(self):
        """Check health of all monitored agents"""
        
        for agent_id in self.agent_health.keys():
            try:
                await self._check_single_agent_health(agent_id)
                self.performance_history['agent_checks'] += 1
                
            except Exception as e:
                self.logger.error("Error checking agent health", 
                                agent_id=agent_id, error=str(e))
    
    async def _check_single_agent_health(self, agent_id: str):
        """Check health of a single agent"""
        
        agent_status = self.agent_health[agent_id]
        
        # Send ping to agent
        start_time = datetime.now()
        
        response = await self.request_response(
            "ping",
            {"requester": "watchdog"},
            timeout_sec=5.0
        )
        
        end_time = datetime.now()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Update agent status
        if response:
            agent_status.last_heartbeat = datetime.now()
            agent_status.is_healthy = True
            agent_status.response_time_ms = response_time_ms
            agent_status.status = "healthy"
            
            # Extract additional metrics from response
            if "uptime_seconds" in response:
                agent_status.uptime_seconds = response["uptime_seconds"]
            if "message_count" in response:
                agent_status.message_count = response["message_count"]
            if "error_count" in response:
                agent_status.error_count = response["error_count"]
        else:
            agent_status.is_healthy = False
            agent_status.response_time_ms = float('inf')
            agent_status.status = "unresponsive"
            
            self.logger.warning("Agent unresponsive", agent_id=agent_id)
    
    async def _ping_all_agents(self):
        """Send ping to all monitored agents"""
        
        ping_tasks = []
        for agent_id in self.agent_health.keys():
            task = asyncio.create_task(self._ping_agent(agent_id))
            ping_tasks.append(task)
        
        # Wait for all pings with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*ping_tasks, return_exceptions=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            self.logger.warning("Agent ping timeout")
    
    async def _ping_agent(self, agent_id: str):
        """Ping a specific agent"""
        try:
            await self.send_message(
                "ping",
                {"requester": "watchdog", "timestamp": datetime.now().isoformat()},
                target_stream=f"cleanify:agents:{agent_id}:input"
            )
        except Exception as e:
            self.logger.error("Error pinging agent", agent_id=agent_id, error=str(e))
    
    async def _check_unresponsive_agents(self):
        """Check for agents that haven't responded recently"""
        
        timeout_threshold = timedelta(seconds=self.alert_thresholds['agent_timeout_sec'])
        current_time = datetime.now()
        
        for agent_id, agent_status in self.agent_health.items():
            if agent_status.last_heartbeat:
                time_since_heartbeat = current_time - agent_status.last_heartbeat
                
                if time_since_heartbeat > timeout_threshold:
                    agent_status.is_healthy = False
                    agent_status.status = "timeout"
                    
                    await self._handle_unresponsive_agent(agent_id, time_since_heartbeat)
    
    async def _handle_unresponsive_agent(self, agent_id: str, time_since_heartbeat: timedelta):
        """Handle unresponsive agent"""
        
        self.logger.error("Agent timeout detected",
                         agent_id=agent_id,
                         time_since_heartbeat=time_since_heartbeat.total_seconds())
        
        # Send alert
        await self._send_health_alert(
            "agent_timeout",
            f"Agent {agent_id} unresponsive for {time_since_heartbeat.total_seconds():.0f} seconds"
        )
        
        # Request agent restart from supervisor
        await self.send_message(
            "restart_agent_request",
            {
                "agent_id": agent_id,
                "reason": "timeout",
                "time_since_heartbeat_sec": time_since_heartbeat.total_seconds()
            },
            target_stream="cleanify:agents:supervisor:input"
        )
        
        self.performance_history['restarts_initiated'] += 1
    
    async def _analyze_health_and_alert(self):
        """Analyze system and agent health, send alerts if needed"""
        
        # Check system metrics
        if self.system_metrics_history:
            latest_metrics = self.system_metrics_history[-1]
            
            # CPU alert
            if latest_metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
                await self._send_health_alert(
                    "high_cpu",
                    f"CPU usage high: {latest_metrics.cpu_percent:.1f}%"
                )
            
            # Memory alert
            if latest_metrics.memory_percent > self.alert_thresholds['memory_percent']:
                await self._send_health_alert(
                    "high_memory",
                    f"Memory usage high: {latest_metrics.memory_percent:.1f}%"
                )
            
            # Disk alert
            if latest_metrics.disk_usage_percent > self.alert_thresholds['disk_percent']:
                await self._send_health_alert(
                    "high_disk",
                    f"Disk usage high: {latest_metrics.disk_usage_percent:.1f}%"
                )
        
        # Check agent response times
        for agent_id, agent_status in self.agent_health.items():
            if (agent_status.is_healthy and 
                agent_status.response_time_ms > self.alert_thresholds['response_time_ms']):
                
                await self._send_health_alert(
                    "slow_agent",
                    f"Agent {agent_id} slow response: {agent_status.response_time_ms:.0f}ms"
                )
    
    async def _send_health_alert(self, alert_type: str, message: str):
        """Send health alert"""
        
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "source": "watchdog",
            "severity": self._get_alert_severity(alert_type)
        }
        
        await self.send_message(
            "health_alert",
            alert_data,
            target_stream="cleanify:alerts"
        )
        
        self.performance_history['alerts_sent'] += 1
        
        self.logger.warning("Health alert sent",
                          alert_type=alert_type,
                          message=message)
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get severity level for alert type"""
        
        severity_map = {
            "high_cpu": "medium",
            "high_memory": "high",
            "high_disk": "high",
            "slow_agent": "medium",
            "agent_timeout": "high",
            "system_overload": "critical"
        }
        
        return severity_map.get(alert_type, "medium")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory buildup"""
        
        # Keep only last 24 hours of system metrics
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.system_metrics_history = [
            metrics for metrics in self.system_metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def _register_watchdog_handlers(self):
        """Register watchdog-specific message handlers"""
        self.register_handler("get_system_health", self._handle_get_system_health)
        self.register_handler("get_agent_health", self._handle_get_agent_health)
        self.register_handler("set_alert_thresholds", self._handle_set_alert_thresholds)
        self.register_handler("force_health_check", self._handle_force_health_check)
        self.register_handler("get_performance_stats", self._handle_get_performance_stats)
    
    async def _handle_get_system_health(self, data: Dict[str, Any]):
        """Handle system health request"""
        
        if self.system_metrics_history:
            latest_metrics = self.system_metrics_history[-1]
            
            health_data = {
                "timestamp": latest_metrics.timestamp.isoformat(),
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "memory_available_gb": latest_metrics.memory_available_gb,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "network_io": latest_metrics.network_io,
                "process_count": latest_metrics.process_count,
                "load_average": latest_metrics.load_average,
                "health_status": self._calculate_overall_health(),
                "correlation_id": data.get("correlation_id")
            }
        else:
            health_data = {
                "error": "No system metrics available",
                "correlation_id": data.get("correlation_id")
            }
        
        await self.send_message("system_health_response", health_data)
    
    async def _handle_get_agent_health(self, data: Dict[str, Any]):
        """Handle agent health request"""
        
        agent_health_data = {}
        
        for agent_id, agent_status in self.agent_health.items():
            agent_health_data[agent_id] = {
                "is_healthy": agent_status.is_healthy,
                "last_heartbeat": (
                    agent_status.last_heartbeat.isoformat() 
                    if agent_status.last_heartbeat else None
                ),
                "response_time_ms": agent_status.response_time_ms,
                "message_count": agent_status.message_count,
                "error_count": agent_status.error_count,
                "uptime_seconds": agent_status.uptime_seconds,
                "status": agent_status.status
            }
        
        response_data = {
            "agent_health": agent_health_data,
            "healthy_agents": len([a for a in self.agent_health.values() if a.is_healthy]),
            "total_agents": len(self.agent_health),
            "correlation_id": data.get("correlation_id")
        }
        
        await self.send_message("agent_health_response", response_data)
    
    async def _handle_set_alert_thresholds(self, data: Dict[str, Any]):
        """Handle alert threshold update"""
        
        try:
            new_thresholds = data.get("thresholds", {})
            
            for threshold_name, value in new_thresholds.items():
                if threshold_name in self.alert_thresholds:
                    self.alert_thresholds[threshold_name] = float(value)
            
            await self.send_message(
                "alert_thresholds_updated",
                {
                    "thresholds": self.alert_thresholds,
                    "correlation_id": data.get("correlation_id")
                }
            )
            
            self.logger.info("Alert thresholds updated",
                           thresholds=self.alert_thresholds)
            
        except Exception as e:
            self.logger.error("Error updating alert thresholds", error=str(e))
    
    async def _handle_force_health_check(self, data: Dict[str, Any]):
        """Handle forced health check request"""
        
        try:
            # Force immediate health check
            await self._collect_system_metrics()
            await self._check_agent_health()
            await self._analyze_health_and_alert()
            
            await self.send_message(
                "health_check_completed",
                {
                    "timestamp": datetime.now().isoformat(),
                    "correlation_id": data.get("correlation_id")
                }
            )
            
        except Exception as e:
            self.logger.error("Error in forced health check", error=str(e))
    
    async def _handle_get_performance_stats(self, data: Dict[str, Any]):
        """Handle performance statistics request"""
        
        stats = {
            "performance_history": self.performance_history,
            "monitoring_duration_hours": len(self.system_metrics_history) * (self.health_check_interval / 3600),
            "alert_thresholds": self.alert_thresholds,
            "system_metrics_count": len(self.system_metrics_history),
            "agents_monitored": len(self.agent_health),
            "correlation_id": data.get("correlation_id")
        }
        
        await self.send_message("watchdog_performance_stats", stats)
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health status"""
        
        if not self.system_metrics_history:
            return "unknown"
        
        latest_metrics = self.system_metrics_history[-1]
        
        # Check if any critical thresholds are exceeded
        if (latest_metrics.cpu_percent > self.alert_thresholds['cpu_percent'] or
            latest_metrics.memory_percent > self.alert_thresholds['memory_percent'] or
            latest_metrics.disk_usage_percent > self.alert_thresholds['disk_percent']):
            return "critical"
        
        # Check agent health
        healthy_agents = len([a for a in self.agent_health.values() if a.is_healthy])
        total_agents = len(self.agent_health)
        
        if healthy_agents < total_agents * 0.8:  # Less than 80% agents healthy
            return "degraded"
        elif healthy_agents < total_agents:  # Some agents unhealthy
            return "warning"
        else:
            return "healthy"
    
    def get_watchdog_metrics(self) -> Dict[str, Any]:
        """Get watchdog performance metrics"""
        
        return {
            "monitoring": {
                "system_checks": self.performance_history['system_checks'],
                "agent_checks": self.performance_history['agent_checks'],
                "health_check_interval_sec": self.health_check_interval
            },
            "alerts": {
                "alerts_sent": self.performance_history['alerts_sent'],
                "restarts_initiated": self.performance_history['restarts_initiated']
            },
            "system_health": self._calculate_overall_health(),
            "agent_status": {
                "total_agents": len(self.agent_health),
                "healthy_agents": len([a for a in self.agent_health.values() if a.is_healthy]),
                "unresponsive_agents": len([a for a in self.agent_health.values() if not a.is_healthy])
            },
            "thresholds": self.alert_thresholds
        }
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "system_health": {},
            "agent_health": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # System health summary
        if self.system_metrics_history:
            latest_metrics = self.system_metrics_history[-1]
            
            report["system_health"] = {
                "overall_status": self._calculate_overall_health(),
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "disk_percent": latest_metrics.disk_usage_percent,
                "process_count": latest_metrics.process_count,
                "uptime_hours": len(self.system_metrics_history) * (self.health_check_interval / 3600)
            }
        
        # Agent health summary
        healthy_agents = []
        unhealthy_agents = []
        
        for agent_id, agent_status in self.agent_health.items():
            if agent_status.is_healthy:
                healthy_agents.append({
                    "agent_id": agent_id,
                    "response_time_ms": agent_status.response_time_ms,
                    "uptime_hours": agent_status.uptime_seconds / 3600,
                    "message_count": agent_status.message_count,
                    "error_rate": agent_status.error_count / max(1, agent_status.message_count)
                })
            else:
                unhealthy_agents.append({
                    "agent_id": agent_id,
                    "status": agent_status.status,
                    "last_seen": (
                        agent_status.last_heartbeat.isoformat() 
                        if agent_status.last_heartbeat else "never"
                    )
                })
        
        report["agent_health"] = {
            "healthy_agents": healthy_agents,
            "unhealthy_agents": unhealthy_agents,
            "health_percentage": len(healthy_agents) / len(self.agent_health) * 100
        }
        
        # Performance summary
        report["performance_summary"] = self.performance_history.copy()
        
        # Generate recommendations
        recommendations = []
        
        if latest_metrics.cpu_percent > 70:
            recommendations.append("Consider CPU optimization or scaling")
        
        if latest_metrics.memory_percent > 80:
            recommendations.append("Monitor memory usage closely")
        
        if len(unhealthy_agents) > 0:
            recommendations.append(f"Investigate {len(unhealthy_agents)} unhealthy agents")
        
        if self.performance_history['alerts_sent'] > 10:
            recommendations.append("Review alert thresholds - many alerts generated")
        
        report["recommendations"] = recommendations
        
        return report