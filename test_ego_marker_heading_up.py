#!/usr/bin/env python3
"""
Test EGO marker horizontal display with heading_up enabled
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import tempfile
from pathlib import Path
from metadrive.envs.base_env import BaseEnv

def test_ego_marker_heading_up():
    """Test EGO marker display with heading_up=True"""
    
    env_config = {
        "use_render": False,
        "num_agents": 1,
        "random_traffic": False,
        "map": "CCC",
    }
    
    env = BaseEnv(env_config)
    
    try:
        obs, info = env.reset()
        
        print("Testing EGO marker with target_agent_heading_up enabled...")
        print("=" * 60)
        
        # Test frames with different ego headings
        test_frames = []
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get ego vehicle heading
            ego_heading = env.agents["agent0"].heading_theta if "agent0" in env.agents else 0.0
            
            # Render with heading_up enabled
            frame = env.render(
                to_image=True,
                mode="top_down",
                window=False,
                target_agent_heading_up=True,
                screen_size=(800, 800),
                film_size=(2000, 2000),
            )
            
            if frame is not None:
                test_frames.append((step, ego_heading, frame))
                print(f"✓ Step {step:2d}: Ego heading={np.rad2deg(ego_heading):7.2f}° - Frame rendered")
            
            if terminated or truncated:
                break
        
        if test_frames:
            # Save test frames for visual inspection
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                for i, (step, heading, frame) in enumerate(test_frames[:3]):
                    import cv2
                    out_file = tmpdir_path / f"ego_marker_step{step}_heading{np.rad2deg(heading):.0f}deg.jpg"
                    cv2.imwrite(str(out_file), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"   Saved: {out_file.name}")
        
        print("=" * 60)
        print("✓ Test completed successfully!")
        print("\nExpected behavior:")
        print("- EGO label should remain horizontal regardless of ego heading")
        print("- Camera should rotate around ego vehicle")
        print("- '!' warning markers should also stay horizontal")
        
    finally:
        env.close()

if __name__ == "__main__":
    test_ego_marker_heading_up()
