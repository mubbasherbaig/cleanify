"""
Cleanify v2-alpha Agent Base
Foundation class for all agents with Redis streams and asyncio support
"""

import asyncio
import json
import uuid
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import redis.asyncio as redis
import structlog

from core.models import AgentMessage
from core.settings import get_settings


logger = structlog.get_logger()


class AgentBase(ABC):
    """
    Base class for all Cleanify v2 agents
    Provides Redis communication, heartbeat, and lifecycle management
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.settings = get_settings()
        
        # Runtime state
        self.running = False
        self.redis_client: Optional[redis.Redis] = None
        self.last_heartbeat = datetime.now()
        self.message_handlers: Dict[str, Callable] = {}
        self.startup_time = datetime.now()
        self.message_count = 0
        self.error_count = 0
        
        # Stream names
        self.input_stream = f"cleanify:agents:{self.agent_type}:input"
        self.output_stream = f"cleanify:agents:{self.agent_type}:output"
        self.heartbeat_stream = "cleanify:agents:heartbeat"
        self.consumer_group = self.settings.redis.CONSUMER_GROUP
        self.consumer_name = f"{self.agent_id}_{int(time.time())}"
        
        # Setup logging
        self.logger = logger.bind(agent_id=self.agent_id, agent_type=self.agent_type)
        
        # Register default message handlers
        self._register_default_handlers()
    
    async def startup(self):
        """Initialize agent and connect to Redis"""
        try:
            self.logger.info("Starting agent startup sequence")
            
            # Connect to Redis
            await self._connect_redis()
            
            # Setup consumer groups
            await self._setup_consumer_groups()
            
            # Agent-specific initialization
            await self.initialize()
            
            self.running = True
            self.logger.info("Agent startup complete", 
                           startup_time=self.startup_time.isoformat())
            
        except Exception as e:
            self.logger.error("Agent startup failed", error=str(e))
            raise
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            self.logger.info("Starting agent shutdown")
            self.running = False
            
            # Agent-specific cleanup
            await self.cleanup()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Agent shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during shutdown", error=str(e))
    
    async def run(self):
        """Main agent execution loop"""
        if not self.running:
            await self.startup()
        
        self.logger.info("Starting main execution loop")
        
        # Start background tasks
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        message_task = asyncio.create_task(self._message_loop())
        
        try:
            # Run agent-specific logic
            await self.main_loop()
            
        except Exception as e:
            self.logger.error("Error in main loop", error=str(e))
            raise
        finally:
            # Cancel background tasks
            heartbeat_task.cancel()
            message_task.cancel()
            
            # Wait for cleanup
            await asyncio.gather(heartbeat_task, message_task, return_exceptions=True)
            await self.shutdown()
    
    @abstractmethod
    async def initialize(self):
        """Agent-specific initialization - implement in subclasses"""
        pass
    
    @abstractmethod
    async def main_loop(self):
        """Main agent logic - implement in subclasses"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Agent-specific cleanup - implement in subclasses"""
        pass
    
    async def _connect_redis(self):
        """Connect to Redis with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.redis_client = redis.from_url(
                    self.settings.get_redis_url(),
                    decode_responses=True
                )
                
                # Test connection
                await self.redis_client.ping()
                self.logger.info("Connected to Redis")
                return
                
            except Exception as e:
                self.logger.warning(f"Redis connection attempt {attempt + 1} failed", 
                                  error=str(e))
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def _setup_consumer_groups(self):
        """Setup Redis consumer groups for message streams"""
        try:
            # Create consumer group for input stream
            await self.redis_client.xgroup_create(
                self.input_stream, 
                self.consumer_group, 
                id='0', 
                mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        
        self.logger.info("Consumer groups configured")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.running:
            try:
                await self.send_heartbeat()
                await asyncio.sleep(self.settings.agents.HEARTBEAT_INTERVAL_SEC)
            except Exception as e:
                self.logger.error("Heartbeat error", error=str(e))
                await asyncio.sleep(1)
    
    async def _message_loop(self):
        """Process incoming messages from Redis streams"""
        while self.running:
            try:
                # Read messages from input stream
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.input_stream: '>'},
                    count=10,
                    block=self.settings.redis.CONSUMER_TIMEOUT_MS
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self._process_message(message_id, fields)
                        
                        # Acknowledge message
                        await self.redis_client.xack(
                            self.input_stream, 
                            self.consumer_group, 
                            message_id
                        )
                        
            except Exception as e:
                self.logger.error("Message processing error", error=str(e))
                self.error_count += 1
                await asyncio.sleep(1)
    
    async def _process_message(self, message_id: str, fields: Dict[str, str]):
        """Process individual message"""
        try:
            # Parse message
            message_data = json.loads(fields.get('data', '{}'))
            message_type = fields.get('type', 'unknown')
            
            self.message_count += 1
            
            # Find and execute handler
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](message_data)
            else:
                await self.handle_unknown_message(message_type, message_data)
                
        except Exception as e:
            self.logger.error("Error processing message", 
                            message_id=message_id, error=str(e))
            self.error_count += 1
    
    async def send_message(self, message_type: str, payload: Dict[str, Any], 
                          target_stream: Optional[str] = None, 
                          priority: int = 0) -> str:
        """Send message to Redis stream"""
        
        if target_stream is None:
            target_stream = self.output_stream
        
        message = AgentMessage(
            agent_id=self.agent_id,
            message_type=message_type,
            payload=payload,
            priority=priority
        )
        
        message_id = await self.redis_client.xadd(
            target_stream,
            {
                'agent_id': self.agent_id,
                'type': message_type,
                'data': json.dumps(payload),
                'priority': priority,
                'timestamp': message.timestamp.isoformat()
            },
            maxlen=self.settings.redis.STREAM_MAXLEN
        )
        
        # self.logger.debug("Message sent", 
        #                  message_type=message_type, 
        #                  target_stream=target_stream,
        #                  message_id=message_id)
        
        return message_id
    
    async def send_heartbeat(self):
        """Send heartbeat message"""
        self.last_heartbeat = datetime.now()
        
        await self.send_message(
            "heartbeat",
            {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "timestamp": self.last_heartbeat.isoformat(),
                "uptime_seconds": (self.last_heartbeat - self.startup_time).total_seconds(),
                "message_count": self.message_count,
                "error_count": self.error_count,
                "status": "healthy" if self.running else "shutdown"
            },
            target_stream=self.heartbeat_stream
        )
    
    async def request_response(self, message_type: str, payload: Dict[str, Any], 
                             timeout_sec: float = 30.0) -> Optional[Dict[str, Any]]:
        """Send request and wait for response"""
        correlation_id = str(uuid.uuid4())
        
        # Add correlation ID to payload
        payload['correlation_id'] = correlation_id
        
        # Send request
        await self.send_message(message_type, payload)
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            # Check for response in output stream
            messages = await self.redis_client.xread(
                {f"{self.output_stream}:responses": '$'},
                count=10,
                block=1000
            )
            
            for stream, stream_messages in messages:
                for message_id, fields in stream_messages:
                    response_data = json.loads(fields.get('data', '{}'))
                    if response_data.get('correlation_id') == correlation_id:
                        return response_data
            
            await asyncio.sleep(0.1)
        
        self.logger.warning("Request timeout", 
                           message_type=message_type, 
                           correlation_id=correlation_id)
        return None
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler
        self.logger.debug("Handler registered", message_type=message_type)
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler("ping", self._handle_ping)
        self.register_handler("status", self._handle_status)
        self.register_handler("shutdown", self._handle_shutdown)
    
    async def _handle_ping(self, data: Dict[str, Any]):
        """Handle ping message"""
        await self.send_message(
            "pong",
            {
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": data.get('correlation_id')
            }
        )
    
    async def _handle_status(self, data: Dict[str, Any]):
        """Handle status request"""
        status = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "running": self.running,
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "correlation_id": data.get('correlation_id')
        }
        
        await self.send_message("status_response", status)
    
    async def _handle_shutdown(self, data: Dict[str, Any]):
        """Handle shutdown request"""
        self.logger.info("Shutdown requested via message")
        self.running = False
    
    async def handle_unknown_message(self, message_type: str, data: Dict[str, Any]):
        """Handle unknown message types - override in subclasses"""
        self.logger.warning("Unknown message type", message_type=message_type)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        now = datetime.now()
        time_since_heartbeat = (now - self.last_heartbeat).total_seconds()
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "healthy": self.running and time_since_heartbeat < 60,
            "running": self.running,
            "uptime_seconds": (now - self.startup_time).total_seconds(),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.message_count),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "time_since_heartbeat_sec": time_since_heartbeat
        }
    
    async def wait_for_termination(self):
        """Wait for agent termination signal"""
        while self.running:
            await asyncio.sleep(1)
    
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """Publish event to event stream"""
        await self.send_message(
            "event",
            {
                "event_type": event_type,
                "event_data": event_data,
                "source_agent": self.agent_id,
                "timestamp": datetime.now().isoformat()
            },
            target_stream="cleanify:events"
        )
    
    async def get_system_time(self) -> datetime:
        """Get system time (can be simulation time)"""
        # In real implementation, this might come from supervisor
        return datetime.now()
    
    def __str__(self) -> str:
        return f"{self.agent_type}Agent({self.agent_id})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id='{self.agent_id}', "
                f"agent_type='{self.agent_type}', running={self.running})")