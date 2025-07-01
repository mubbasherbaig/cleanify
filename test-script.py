#!/usr/bin/env python3
"""
Focused Backend Test - Isolate the exact issue
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_step_by_step():
    """Test each component step by step"""
    
    print("🧪 FOCUSED BACKEND TEST")
    print("=" * 50)
    
    # Test 1: Import test
    print("\n📦 Step 1: Testing imports...")
    try:
        from agents.supervisor import SupervisorAgent
        from core.models import Bin, Truck, SystemState
        from core.settings import settings
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Create objects directly
    print("\n🔧 Step 2: Testing object creation...")
    try:
        # Test Bin creation
        test_bin = Bin(
            id="TEST_BIN",
            lat=33.6844,
            lon=73.0479,
            capacity_l=100,
            fill_level=75.0,
            fill_rate_lph=3.0,
            tile_id=""
        )
        print(f"✅ Bin created: {test_bin.id} at ({test_bin.lat}, {test_bin.lon})")
        
        # Test Truck creation
        test_truck = Truck(
            id="TEST_TRUCK",
            name="Test Truck",
            capacity_l=5000,
            lat=33.6844,
            lon=73.0479
        )
        print(f"✅ Truck created: {test_truck.id} ({test_truck.name})")
        
        # Test SystemState creation
        test_state = SystemState(
            timestamp=asyncio.get_event_loop().time(),
            bins=[test_bin],
            trucks=[test_truck],
            active_routes=[],
            traffic_conditions=[],
            simulation_running=False
        )
        print(f"✅ SystemState created with {len(test_state.bins)} bins, {len(test_state.trucks)} trucks")
        
    except Exception as e:
        print(f"❌ Object creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Supervisor creation and initialization
    print("\n🤖 Step 3: Testing supervisor creation...")
    try:
        supervisor = SupervisorAgent()
        print("✅ Supervisor created")
        
        # Test without full initialization first
        print("🔧 Testing direct config handling...")
        
        # Set up minimal state
        supervisor.system_state = None
        
        # Test config data
        test_config_data = {
            "config": {
                "depot": {
                    "name": "Test Depot",
                    "latitude": 33.6844,
                    "longitude": 73.0479
                },
                "trucks": [
                    {
                        "id": "T001",
                        "name": "Test Truck",
                        "capacity": 5000
                    }
                ],
                "bins": [
                    {
                        "id": "BIN001",
                        "latitude": 33.6844,
                        "longitude": 73.0479,
                        "capacity_l": 100,
                        "fill_level": 75.0,
                        "fill_rate_lph": 3.0
                    }
                ]
            }
        }
        
        print(f"🔧 Test config: {json.dumps(test_config_data, indent=2)}")
        
        # Call the handler directly
        print("🔧 Calling _handle_load_config...")
        await supervisor._handle_load_config(test_config_data)
        
        # Check result
        if supervisor.system_state:
            bins_count = len(supervisor.system_state.bins)
            trucks_count = len(supervisor.system_state.trucks)
            print(f"✅ Config handling successful: {bins_count} bins, {trucks_count} trucks loaded")
            
            # Test conversion methods
            print("🔧 Testing conversion methods...")
            if bins_count > 0:
                bin_dict = supervisor._bin_to_dict(supervisor.system_state.bins[0])
                print(f"✅ Bin conversion successful: {bin_dict}")
            
            if trucks_count > 0:
                truck_dict = supervisor._truck_to_dict(supervisor.system_state.trucks[0])
                print(f"✅ Truck conversion successful: {truck_dict}")
            
            return True
        else:
            print("❌ System state is still None after config loading")
            return False
            
    except Exception as e:
        print(f"❌ Supervisor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'supervisor' in locals():
            try:
                await supervisor.shutdown()
            except:
                pass

async def test_redis_connection():
    """Test Redis connection"""
    print("\n🔗 Step 4: Testing Redis connection...")
    try:
        import redis.asyncio as redis
        from core.settings import get_settings
        
        settings = get_settings()
        redis_client = redis.from_url(settings.get_redis_url())
        
        await redis_client.ping()
        print("✅ Redis connection successful")
        await redis_client.close()
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Make sure Redis is running: sudo systemctl start redis")
        return False

async def test_full_supervisor():
    """Test supervisor with full initialization"""
    print("\n🚀 Step 5: Testing full supervisor initialization...")
    
    try:
        from agents.supervisor import SupervisorAgent
        
        supervisor = SupervisorAgent()
        
        # Initialize supervisor
        print("🔧 Initializing supervisor...")
        await supervisor.initialize()
        print("✅ Supervisor initialized")
        
        # Test config loading
        test_config_data = {
            "config": {
                "depot": {
                    "name": "Test Depot",
                    "latitude": 33.6844,
                    "longitude": 73.0479
                },
                "trucks": [
                    {
                        "id": "T001",
                        "name": "Test Truck",
                        "capacity": 5000
                    }
                ],
                "bins": [
                    {
                        "id": "BIN001",
                        "latitude": 33.6844,
                        "longitude": 73.0479,
                        "capacity_l": 100,
                        "fill_level": 75.0,
                        "fill_rate_lph": 3.0
                    }
                ]
            }
        }
        
        print("🔧 Loading config...")
        await supervisor._handle_load_config(test_config_data)
        
        if supervisor.system_state and len(supervisor.system_state.bins) > 0:
            print(f"✅ Full supervisor test successful: {len(supervisor.system_state.bins)} bins loaded")
            return True
        else:
            print("❌ Full supervisor test failed: no data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Full supervisor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'supervisor' in locals():
            try:
                await supervisor.shutdown()
            except:
                pass

async def main():
    """Run focused tests"""
    
    results = {}
    
    # Run tests in sequence
    results['step_by_step'] = await test_step_by_step()
    results['redis'] = await test_redis_connection()
    results['full_supervisor'] = await test_full_supervisor()
    
    print("\n" + "=" * 50)
    print("📊 FOCUSED TEST RESULTS:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All focused tests PASSED!")
        print("💡 The backend components work correctly.")
        print("💡 The issue might be in the API integration or message passing.")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        
        if not results['redis']:
            print("💡 Start Redis: sudo systemctl start redis")
        if not results['step_by_step']:
            print("💡 Check the detailed error logs above")
        if not results['full_supervisor']:
            print("💡 Issue with supervisor initialization or message handling")
    
    return all(results.values())

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)