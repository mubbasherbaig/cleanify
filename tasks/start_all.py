# tasks/start_all.py
"""
Cleanify v2 startup script - boots Supervisor which spawns all other agents
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.supervisor import SupervisorAgent
from core.settings import settings
import redis.asyncio as redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_redis_connection():
    """Verify Redis is available before starting agents"""
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        await redis_client.close()
        logger.info("Redis connection verified")
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


async def main():
    """Main entry point - start the Supervisor which spawns all other agents"""
    logger.info("Starting Cleanify v2 system...")
    
    # Check Redis connectivity
    if not await check_redis_connection():
        logger.error("Cannot start without Redis. Please ensure Redis is running.")
        return 1
    
    # Create and start the Supervisor
    supervisor = SupervisorAgent()
    
    try:
        logger.info("Initializing Supervisor...")
        await supervisor.run()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Supervisor crashed: {e}")
        return 1
    finally:
        logger.info("Shutting down Cleanify v2 system")
        await supervisor.shutdown()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)