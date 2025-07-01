"""
Cleanify v2-alpha FastAPI Routes
REST API endpoints for system interaction
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from pydantic import BaseModel, Field

from core.settings import get_settings
from agents.supervisor import SupervisorAgent

# Global supervisor instance
supervisor: Optional[SupervisorAgent] = None
redis_client: Optional[redis.Redis] = None
settings = get_settings()


class SystemStateResponse(BaseModel):
    """System state response model"""
    timestamp: str
    simulation_running: bool
    simulation_speed: float
    bins: List[Dict[str, Any]]
    trucks: List[Dict[str, Any]]
    active_routes: List[Dict[str, Any]]
    traffic_conditions: List[Dict[str, Any]]


class ConfigurationRequest(BaseModel):
    """Configuration upload request model"""
    depot: Dict[str, Any] = Field(..., description="Depot configuration")
    trucks: List[Dict[str, Any]] = Field(..., description="Truck configurations")
    bins: List[Dict[str, Any]] = Field(..., description="Bin configurations")


class SimulationSpeedRequest(BaseModel):
    """Simulation speed change request"""
    multiplier: float = Field(..., ge=0.1, le=10.0, description="Speed multiplier (0.1-10.0)")


class DispatchRequest(BaseModel):
    """Manual dispatch request"""
    truck_id: str = Field(..., description="Truck ID to dispatch")
    bin_ids: List[str] = Field(..., description="Bin IDs to collect")
    priority: str = Field(default="normal", description="Dispatch priority")


class EmergencyRequest(BaseModel):
    """Emergency trigger request"""
    event_type: str = Field(..., description="Emergency event type")
    severity: str = Field(default="medium", description="Emergency severity")
    description: str = Field(..., description="Emergency description")
    affected_bins: List[str] = Field(default=[], description="Affected bin IDs")
    affected_trucks: List[str] = Field(default=[], description="Affected truck IDs")


# Create FastAPI app
app = FastAPI(
    title="Cleanify v2-alpha API",
    description="Intelligent waste collection system with agent-based optimization",
    version="2.0.0-alpha",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize API components"""
    global supervisor, redis_client
    
    print("🚀 Starting Cleanify v2-alpha API")
    
    # Initialize Redis client
    redis_client = redis.from_url(settings.get_redis_url(), decode_responses=True)
    
    # Test Redis connection
    try:
        await redis_client.ping()
        print("✅ Redis connection established")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        raise
    
    # Start supervisor in background
    supervisor = SupervisorAgent()
    asyncio.create_task(supervisor.run())
    
    print("✅ Cleanify v2-alpha API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup API components"""
    global supervisor, redis_client
    
    print("🛑 Shutting down Cleanify v2-alpha API")
    
    if supervisor:
        await supervisor.shutdown()
    
    if redis_client:
        await redis_client.close()
    
    print("✅ Cleanify v2-alpha API shutdown complete")


@app.get("/", tags=["System"])
async def root():
    """API root endpoint"""
    return {
        "service": "Cleanify v2-alpha",
        "version": "2.0.0-alpha",
        "status": "running",
        "docs": "/docs",
        "agents": "agent-based architecture",
        "optimization": "OR-Tools + Redis Streams"
    }
@app.get("/api/debug/system-state", tags=["Debug"])
async def debug_system_state():
    """Debug endpoint to see raw system state"""
    
    if not supervisor:
        return {"error": "Supervisor not available"}
    
    try:
        raw_state = {
            "supervisor_available": supervisor is not None,
            "system_state_exists": supervisor.system_state is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if supervisor.system_state:
            raw_state.update({
                "bins_count": len(supervisor.system_state.bins),
                "trucks_count": len(supervisor.system_state.trucks),
                "active_routes_count": len(supervisor.system_state.active_routes),
                "simulation_running": supervisor.system_state.simulation_running,
                "last_update": supervisor.system_state.timestamp.isoformat() if supervisor.system_state.timestamp else None
            })
            
            # Add sample data for debugging
            if supervisor.system_state.bins:
                raw_state["sample_bin"] = supervisor._bin_to_dict(supervisor.system_state.bins[0])
            
            if supervisor.system_state.trucks:
                raw_state["sample_truck"] = supervisor._truck_to_dict(supervisor.system_state.trucks[0])
        
        return raw_state
        
    except Exception as e:
        return {
            "error": f"Debug error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/debug/reload-config", tags=["Debug"])
async def debug_reload_config():
    """Force reload last configuration"""
    
    if not supervisor:
        raise HTTPException(status_code=503, detail="Supervisor not available")
    
    try:
        # This would reload the last known good config
        # For now, just return the current state
        if supervisor.system_state:
            return {
                "status": "current_state",
                "bins": len(supervisor.system_state.bins),
                "trucks": len(supervisor.system_state.trucks),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "no_state",
                "message": "No system state available",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Debug reload error: {str(e)}"
        )

@app.get("/api/system-state", response_model=SystemStateResponse, tags=["System"])
async def get_system_state():
    """Get current system state"""
    
    if not supervisor or not supervisor.system_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    
    try:
        state = supervisor.system_state
        
        return SystemStateResponse(
            timestamp=state.timestamp.isoformat(),
            simulation_running=state.simulation_running,
            simulation_speed=state.simulation_speed,
            bins=[supervisor._bin_to_dict(bin_obj) for bin_obj in state.bins],
            trucks=[supervisor._truck_to_dict(truck) for truck in state.trucks],
            active_routes=[supervisor._route_to_dict(route) for route in state.active_routes],
            traffic_conditions=[
                {
                    "timestamp": tc.timestamp.isoformat(),
                    "level": tc.level,
                    "multiplier": tc.multiplier,
                    "region": tc.region,
                    "source": tc.source
                }
                for tc in state.traffic_conditions
            ]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system state: {str(e)}"
        )


@app.post("/api/load-config", tags=["Configuration"])
async def load_config(config: ConfigurationRequest):
    """Load system configuration - FIXED VERSION"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        print(f"🔧 API: Loading config with {len(config.bins)} bins, {len(config.trucks)} trucks")
        
        # Call supervisor's config handler directly (no Redis needed for API calls)
        config_data = {
            "config": {
                "depot": config.depot,
                "trucks": config.trucks,
                "bins": config.bins
            }
        }
        
        # Call the supervisor method directly instead of sending Redis message
        success = await supervisor._handle_load_config(config_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Supervisor failed to load configuration"
            )
        
        # Verify the config was actually loaded by checking supervisor state
        if not supervisor.system_state or len(supervisor.system_state.bins) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Configuration loaded but no data found in supervisor"
            )
        
        print(f"✅ API: Config verified - {len(supervisor.system_state.bins)} bins, {len(supervisor.system_state.trucks)} trucks")
        
        return {
            "status": "success",
            "message": "Configuration loaded and verified",
            "depot": config.depot.get("name", "Depot"),
            "trucks": len(supervisor.system_state.trucks),
            "bins": len(supervisor.system_state.bins),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API: Configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Configuration error: {str(e)}"
        )


@app.get("/api/system-state", response_model=SystemStateResponse, tags=["System"])
async def get_system_state():
    """Get current system state - FIXED VERSION"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    # Get state directly from supervisor instead of Redis
    state = supervisor.get_current_system_state()
    
    if not state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System state not initialized"
        )
    
    try:
        print(f"🔍 API: Returning system state with {len(state.bins)} bins, {len(state.trucks)} trucks")
        
        return SystemStateResponse(
            timestamp=state.timestamp.isoformat(),
            simulation_running=state.simulation_running,
            simulation_speed=getattr(state, 'simulation_speed', 1.0),
            bins=[supervisor._bin_to_dict(bin_obj) for bin_obj in state.bins],
            trucks=[supervisor._truck_to_dict(truck) for truck in state.trucks],
            active_routes=[supervisor._route_to_dict(route) for route in state.active_routes],
            traffic_conditions=[
                {
                    "timestamp": tc.timestamp.isoformat(),
                    "level": tc.level,
                    "multiplier": tc.multiplier,
                    "region": tc.region,
                    "source": tc.source
                }
                for tc in state.traffic_conditions
            ]
        )
        
    except Exception as e:
        print(f"❌ API: Error getting system state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system state: {str(e)}"
        )

@app.post("/api/simulation/start", tags=["Simulation"])
async def start_simulation():
    """Start simulation"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        await supervisor.send_message("start_simulation", {})
        
        return {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "message": "Simulation started successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting simulation: {str(e)}"
        )


@app.post("/api/simulation/pause", tags=["Simulation"])
async def pause_simulation():
    """Pause simulation"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        await supervisor.send_message("pause_simulation", {})
        
        return {
            "status": "paused",
            "timestamp": datetime.now().isoformat(),
            "message": "Simulation paused successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error pausing simulation: {str(e)}"
        )


@app.post("/api/simulation/speed", tags=["Simulation"])
async def set_simulation_speed(speed_request: SimulationSpeedRequest):
    """Set simulation speed multiplier"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        await supervisor.send_message(
            "set_simulation_speed",
            {"speed": speed_request.multiplier}
        )
        
        return {
            "status": "success",
            "speed": speed_request.multiplier,
            "timestamp": datetime.now().isoformat(),
            "message": f"Simulation speed set to {speed_request.multiplier}x"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting simulation speed: {str(e)}"
        )


@app.post("/api/dispatch/plan", tags=["Operations"])
async def plan_routes():
    """Trigger route planning"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        # Send route planning request to supervisor
        await supervisor.send_message(
            "trigger_route_planning",
            {"timestamp": datetime.now().isoformat()}
        )
        
        return {
            "status": "success",
            "message": "Route planning triggered",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error triggering route planning: {str(e)}"
        )


@app.post("/api/dispatch/manual", tags=["Operations"])
async def manual_dispatch(dispatch_request: DispatchRequest):
    """Manual truck dispatch"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        # Send manual dispatch request
        await supervisor.send_message(
            "manual_dispatch",
            {
                "truck_id": dispatch_request.truck_id,
                "bin_ids": dispatch_request.bin_ids,
                "priority": dispatch_request.priority,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "status": "success",
            "truck_id": dispatch_request.truck_id,
            "bins_assigned": len(dispatch_request.bin_ids),
            "priority": dispatch_request.priority,
            "message": f"Truck {dispatch_request.truck_id} dispatched to {len(dispatch_request.bin_ids)} bins"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Manual dispatch error: {str(e)}"
        )


@app.get("/api/agents/status", tags=["Monitoring"])
async def get_agent_status():
    """Get agent health status"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        # Get agent health from supervisor
        agent_health = supervisor.agent_health
        
        health_summary = {
            "total_agents": len(agent_health),
            "healthy_agents": len([h for h in agent_health.values() if h.get("healthy", False)]),
            "supervisor_status": supervisor.get_health_status(),
            "agents": agent_health,
            "timestamp": datetime.now().isoformat()
        }
        
        return health_summary
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting agent status: {str(e)}"
        )


@app.get("/api/system/metrics", tags=["Monitoring"])
async def get_system_metrics():
    """Get system performance metrics"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        metrics = await supervisor.get_system_metrics()
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system metrics: {str(e)}"
        )


@app.post("/api/overflow/trigger", tags=["Testing"])
async def trigger_emergency(emergency_request: EmergencyRequest):
    """Trigger emergency for testing"""
    
    if not supervisor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supervisor not available"
        )
    
    try:
        # Send emergency trigger to emergency agent
        await supervisor.send_message(
            "trigger_emergency",
            {
                "event_type": emergency_request.event_type,
                "severity": emergency_request.severity,
                "description": emergency_request.description,
                "affected_bins": emergency_request.affected_bins,
                "affected_trucks": emergency_request.affected_trucks,
                "timestamp": datetime.now().isoformat()
            },
            target_stream="cleanify:agents:emergency:input"
        )
        
        return {
            "status": "success",
            "event_type": emergency_request.event_type,
            "severity": emergency_request.severity,
            "message": f"Emergency '{emergency_request.event_type}' triggered for testing",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error triggering emergency: {str(e)}"
        )


@app.get("/api/forecast/predictions", tags=["Analytics"])
async def get_overflow_predictions():
    """Get bin overflow predictions"""
    
    if not supervisor or not supervisor.system_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    
    try:
        # Send prediction request to forecast agent
        bins_data = [supervisor._bin_to_dict(bin_obj) for bin_obj in supervisor.system_state.bins]
        
        await supervisor.send_message(
            "predict_overflow",
            {"bins": bins_data},
            target_stream="cleanify:agents:forecast:input"
        )
        
        # In production, would wait for response with correlation ID
        return {
            "status": "success",
            "message": "Overflow prediction requested",
            "bins_analyzed": len(bins_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting predictions: {str(e)}"
        )


@app.get("/api/traffic/conditions", tags=["Analytics"])
async def get_traffic_conditions():
    """Get current traffic conditions"""
    
    try:
        # Send traffic conditions request
        if supervisor:
            await supervisor.send_message(
                "get_conditions",
                {"region": "default"},
                target_stream="cleanify:agents:traffic:input"
            )
        
        # In production, would wait for response
        return {
            "status": "success",
            "message": "Traffic conditions requested",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting traffic conditions: {str(e)}"
        )


@app.get("/api/routes/active", tags=["Operations"])
async def get_active_routes():
    """Get active routes"""
    
    if not supervisor or not supervisor.system_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    
    try:
        active_routes = [
            supervisor._route_to_dict(route) 
            for route in supervisor.system_state.active_routes
        ]
        
        return {
            "active_routes": active_routes,
            "total_routes": len(active_routes),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting active routes: {str(e)}"
        )


@app.get("/api/bins/urgent", tags=["Operations"])
async def get_urgent_bins():
    """Get bins requiring urgent collection"""
    
    if not supervisor or not supervisor.system_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    
    try:
        urgent_bins = supervisor.system_state.urgent_bins()
        
        urgent_bins_data = [
            {
                **supervisor._bin_to_dict(bin_obj),
                "urgency_score": bin_obj.urgency_score(),
                "estimated_full_time": bin_obj.estimated_full_time()
            }
            for bin_obj in urgent_bins
        ]
        
        return {
            "urgent_bins": urgent_bins_data,
            "total_urgent": len(urgent_bins_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting urgent bins: {str(e)}"
        )


@app.get("/api/settings", tags=["Configuration"])
async def get_settings():
    """Get current system settings"""
    
    try:
        return {
            "settings": settings.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting settings: {str(e)}"
        )


@app.get("/api/health", tags=["Monitoring"])
async def health_check():
    """System health check"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check supervisor
    if supervisor:
        supervisor_health = supervisor.get_health_status()
        health_status["components"]["supervisor"] = {
            "status": "healthy" if supervisor_health["healthy"] else "unhealthy",
            "uptime_seconds": supervisor_health["uptime_seconds"]
        }
    else:
        health_status["components"]["supervisor"] = {"status": "unavailable"}
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        if redis_client:
            await redis_client.ping()
            health_status["components"]["redis"] = {"status": "healthy"}
        else:
            health_status["components"]["redis"] = {"status": "unavailable"}
    except Exception:
        health_status["components"]["redis"] = {"status": "unhealthy"}
        health_status["status"] = "degraded"
    
    # Check agents (if supervisor available)
    if supervisor and supervisor.agent_health:
        healthy_agents = len([h for h in supervisor.agent_health.values() if h.get("healthy", False)])
        total_agents = len(supervisor.agent_health)
        
        health_status["components"]["agents"] = {
            "status": "healthy" if healthy_agents == total_agents else "degraded",
            "healthy_count": healthy_agents,
            "total_count": total_agents
        }
        
        if healthy_agents < total_agents * 0.8:
            health_status["status"] = "degraded"
    
    return health_status


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.api.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# Export app for uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.routes:app",
        host=settings.api.HOST,
        port=settings.api.PORT,
        workers=settings.api.WORKERS,
        log_level=settings.api.LOG_LEVEL.lower(),
        reload=settings.api.DEBUG
    )