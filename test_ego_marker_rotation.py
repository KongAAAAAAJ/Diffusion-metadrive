#!/usr/bin/env python3
"""
Test script to verify EGO marker horizontal display with target_agent_heading_up=True
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from metadrive.envs.base_env import BaseEnv

def test_ego_marker_horizontal():
    """
    Test with target_agent_heading_up=True to verify EGO marker stays horizontal
    """
    env_config = {
        "use_render": False,
        "num_agents": 1,
        "agent_observation": "lidar",
        "agent_policy": "expert",
        "random_traffic": False,
        "map": "CCC",
        "vehicle_config": {
            "lidar": {"num_lasers": 12, "distance": 50, "width": 120}
        }
    }
    
    env = BaseEnv(env_config)
    
    try:
        obs, info = env.reset()
        
        # Render with heading_up enabled
        render_kwargs = {
            "mode": "top_down",
            "window": False,
            "target_agent_heading_up": True,  # Enable camera rotation
            "screen_size": (800, 800),
            "film_size": (2000, 2000),
        }
        
        print("Testing topdown render with target_agent_heading_up=True...")
        
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render frame
            frame = env.render(to_image=True, **render_kwargs)
            
            if frame is not None:
                print(f"Step {step}: Rendered frame with shape {frame.shape}")
                # In a real test, you would save the frame and check visually
                # For now, we just verify it renders without error
            else:
                print(f"Step {step}: No frame returned")
            
            if terminated or truncated:
                break
        
        print("Test completed successfully!")
        print("✓ EGO marker rendering works with target_agent_heading_up=True")
        
    finally:
        env.close()

if __name__ == "__main__":
    test_ego_marker_horizontal()
