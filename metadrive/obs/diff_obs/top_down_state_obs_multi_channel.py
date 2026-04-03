import os
import types
import cv2
import numpy as np
import gymnasium as gym
import pygame as _pg

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.obs.top_down_obs_impl import ObjectGraphics
from metadrive.utils import clip


def _draw_scene_multi_agent(self):
    """
    Replacement for TopDownMultiChannel.draw_scene() that uses self.target_vehicle
    instead of engine.agents["default_agent"], enabling multi-agent envs.
    Bound to the TopDownMultiChannel instance via types.MethodType.
    """
    vehicle = self.target_vehicle
    pos = self.canvas_runtime.pos2pix(*vehicle.position)
    clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))

    self._refresh(self.canvas_runtime, pos, clip_size)
    self.canvas_past_pos.fill((0, 0, 0))

    ego_heading = vehicle.heading_theta
    ego_heading = ego_heading if abs(ego_heading) > 2 * np.pi / 180 else 0

    for v in self.engine.get_objects(
        lambda o: isinstance(o, BaseVehicle) or isinstance(o, BaseTrafficParticipant)
    ).values():
        if v is vehicle:
            continue
        h = v.heading_theta
        h = h if abs(h) > 2 * np.pi / 180 else 0
        ObjectGraphics.display(object=v, surface=self.canvas_runtime, heading=h, color=ObjectGraphics.BLUE)

    raw_pos = vehicle.position
    self.stack_past_pos.append(raw_pos)
    for p_index in self._get_stack_indices(len(self.stack_past_pos)):
        p_old = self.stack_past_pos[p_index]
        diff = p_old - raw_pos
        diff = (diff[0] * self.scaling, diff[1] * self.scaling)
        diff = (diff[1], diff[0])
        p = _pg.math.Vector2(tuple(diff))
        p = p.rotate(np.rad2deg(ego_heading) + 90)
        p = (p[1], p[0])
        p = (
            clip(p[0] + self.resolution[0] / 2, -self.resolution[0], self.resolution[0]),
            clip(p[1] + self.resolution[1] / 2, -self.resolution[1], self.resolution[1]),
        )
        self.canvas_past_pos.fill((255, 255, 255), (p, (1, 1)))

    ret = self.obs_window.render(
        canvas_dict=dict(
            road_network=self.canvas_road_network,
            traffic_flow=self.canvas_runtime,
            target_vehicle=self.canvas_ego,
        ),
        position=pos,
        heading=vehicle.heading_theta,
    )
    ret["past_pos"] = self.canvas_past_pos
    return ret


class TopDownLidarStateObservation(LidarStateObservation):
    """
    Combined observation space:
      - "state": lidar point cloud + ego state + navi info (1-D vector, from LidarStateObservation)
      - "image": top-down multi-channel image stack (H x W x C, from TopDownMultiChannel)

    Instantiation follows the standard agent_observation pattern:
        agent_observation = TopDownLidarStateObservation   →   obs(config)

    Relevant env config keys (with defaults):
        norm_pixel      (bool,  True)   – normalise image pixels to [0, 1]
        frame_stack     (int,   3)      – number of stacked traffic-flow frames
        post_stack      (int,   5)      – number of stacked past-position frames
        frame_skip      (int,   5)      – frame interval between consecutive stacks
        resolution_size (int,   84)     – image side length in pixels
        distance        (float, 50)     – top-down view radius in metres
    """

    def __init__(self, config):
        # Initialises StateObservation + lidar obs as self.state_obs
        super().__init__(config)

        self.topdown_obs = TopDownMultiChannel(
            config["vehicle_config"],
            onscreen=False,  # always off-screen: obs extraction only, display handled by env.render()
            clip_rgb=config.get("norm_pixel", True),
            frame_stack=config.get("frame_stack", 3),
            post_stack=config.get("post_stack", 5),
            frame_skip=config.get("frame_skip", 5),
            resolution=(config.get("resolution_size", 84), config.get("resolution_size", 84)),
            max_distance=config.get("distance", 50),
        )
        # Patch draw_scene on this instance to use target_vehicle instead of
        # engine.agents["default_agent"], which doesn't exist in multi-agent envs.
        self.topdown_obs.draw_scene = types.MethodType(_draw_scene_multi_agent, self.topdown_obs)

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "state": super().observation_space,          # Box(275,)
            "image": self.topdown_obs.observation_space, # Box(H, W, C)
        })

    def observe(self, vehicle):
        return {
            "state": super().observe(vehicle),
            "image": self.topdown_obs.observe(vehicle),
        }

    def reset(self, env, vehicle=None):
        super().reset(env, vehicle)
        self.topdown_obs.reset(env, vehicle)

    def destroy(self):
        super().destroy()
        self.topdown_obs.destroy()


class DatasetCollectObservation(TopDownLidarStateObservation):
    """
    Observation for dataset collection.

    Output fields:
      - ego_state: ego state + navigation info
      - others_state: surrounding vehicles state inferred from lidar
      - lidar: lidar rays
      - topdown: top-down multi-channel image
      - rgb_left / rgb_front / rgb_right: three perspective RGB views

    Expected config additions:
      - sensors["rgb_camera"] = (RGBCamera, W, H) or dedicated sensor ids per view
      - dataset_camera_views: optional dict overriding sensor/pose for each view
    """

    DEFAULT_CAMERA_VIEWS = {
        "rgb_left": {
            "sensor_id": "rgb_camera",
            "position": (-0.5, 0, 2),  # (0.0, 0.8, 1.5)
            "hpr": (100.0, -20.0, 0.0),  # (90.0, -5.0, 0.0)
        },
        "rgb_front": {
            "sensor_id": "rgb_camera",
            "position": (0, 5, 2),  # (左右，前后，高度) (0.0, 0.8, 1.5)
            "hpr": (0.0, -7.5, 0.0),  # (水平旋转，俯仰旋转，滚转旋转) (0.0, -5.0, 0.0)
        },
        "rgb_right": {
            "sensor_id": "rgb_camera",
            "position": (0.5, 0, 2),  # (0.0, 0.8, 1.5)
            "hpr": (-100.0, -20.0, 0.0),  # (-90.0, -5.0, 0.0)
        },
    }

    def __init__(self, config):
        super().__init__(config)
        self.norm_pixel = config.get("norm_pixel", True)
        self.camera_views = self._build_camera_views(config)
        self.last_rgb_obs = {view_name: None for view_name in self.camera_views}

    @property
    def observation_space(self):
        obs_space = {
            "ego_state": self.state_obs.observation_space,
            "others_state": self._others_state_observation_space(),
            "lidar": self._lidar_observation_space(),
            "topdown": self.topdown_obs.observation_space,
        }
        for view_name in self.camera_views:
            obs_space[view_name] = self._camera_observation_space(view_name)
        return gym.spaces.Dict(obs_space)

    def observe(self, vehicle):
        ego_state = self.state_observe(vehicle).astype(np.float32)
        others_state, lidar = self._split_lidar_observation(vehicle)
        topdown = self.topdown_obs.observe(vehicle)

        ret = {
            "ego_state": ego_state,
            "others_state": others_state,
            "lidar": lidar,
            "topdown": topdown,
        }
        for view_name, image in self._observe_rgb_views(vehicle).items():
            ret[view_name] = image
        self.current_observation = ret
        return ret

    def save_camera_images(self, vehicle, output_dir, prefix="obs", save_topdown=False):
        os.makedirs(output_dir, exist_ok=True)

        rgb_images = self._observe_rgb_views(vehicle, to_float=False)
        for view_name, image in rgb_images.items():
            file_name = os.path.join(output_dir, f"{prefix}_{vehicle.name}_{view_name}.png")
            cv2.imwrite(file_name, image)

        if save_topdown:
            topdown = self.topdown_obs.observe(vehicle)
            topdown_img = self._to_uint8_image(topdown)
            file_name = os.path.join(output_dir, f"{prefix}_{vehicle.name}_topdown.png")
            cv2.imwrite(file_name, topdown_img)

    def _others_state_observation_space(self):
        lidar_cfg = self.config["vehicle_config"]["lidar"]
        others_state_dim = 0
        if lidar_cfg["num_lasers"] > 0 and lidar_cfg["distance"] > 0:
            others_state_dim = lidar_cfg["num_others"] * 4
            if lidar_cfg["add_others_navi"]:
                others_state_dim += lidar_cfg["num_others"] * 4
        return gym.spaces.Box(-0.0, 1.0, shape=(others_state_dim,), dtype=np.float32)

    def _lidar_observation_space(self):
        lidar_dim = 0
        lidar_cfg = self.config["vehicle_config"]["lidar"]
        if lidar_cfg["num_lasers"] > 0 and lidar_cfg["distance"] > 0:
            lidar_dim = lidar_cfg["num_lasers"]
        return gym.spaces.Box(-0.0, 1.0, shape=(lidar_dim,), dtype=np.float32)

    def _split_lidar_observation(self, vehicle):
        lidar_obs = np.asarray(self.lidar_observe(vehicle), dtype=np.float32)
        others_state_dim = self._others_state_dim(vehicle)
        others_state = lidar_obs[:others_state_dim]
        lidar = lidar_obs[others_state_dim:]
        return others_state, lidar

    @staticmethod
    def _others_state_dim(vehicle):
        lidar_cfg = vehicle.config["lidar"]
        if not (lidar_cfg["num_lasers"] > 0 and lidar_cfg["distance"] > 0):
            return 0
        others_state_dim = lidar_cfg["num_others"] * 4
        if lidar_cfg["add_others_navi"]:
            others_state_dim += lidar_cfg["num_others"] * 4
        return others_state_dim

    def _camera_observation_space(self, view_name):
        sensor_id = self.camera_views[view_name]["sensor_id"]
        sensor_cfg = self.config["sensors"][sensor_id]
        sensor_cls = sensor_cfg[0]
        assert issubclass(sensor_cls, BaseCamera), "Dataset camera sensor must be a BaseCamera subclass"
        shape = (sensor_cfg[2], sensor_cfg[1], sensor_cls.num_channels)
        if self.norm_pixel:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def _build_camera_views(self, config):
        custom_views = config.get("dataset_camera_views", {})
        camera_views = {}
        for view_name, default_cfg in self.DEFAULT_CAMERA_VIEWS.items():
            merged_cfg = dict(default_cfg)
            if view_name in custom_views:
                merged_cfg.update(custom_views[view_name])
            sensor_id = merged_cfg["sensor_id"]
            if sensor_id not in config["sensors"]:
                raise KeyError(
                    "Camera sensor '{}' for view '{}' is missing from env config['sensors']".format(
                        sensor_id, view_name
                    )
                )
            camera_views[view_name] = merged_cfg
        return camera_views

    def _observe_rgb_views(self, vehicle, to_float=None):
        if to_float is None:
            to_float = self.norm_pixel

        rgb_obs = {}
        for view_name, view_cfg in self.camera_views.items():
            sensor = self.engine.get_sensor(view_cfg["sensor_id"])
            image = sensor.perceive(
                to_float=to_float,
                new_parent_node=vehicle.origin,
                position=view_cfg["position"],
                hpr=view_cfg["hpr"],
            )
            rgb_obs[view_name] = image
            self.last_rgb_obs[view_name] = image
        return rgb_obs

    @staticmethod
    def _to_uint8_image(image):
        if image.dtype == np.uint8:
            return image
        image = np.clip(image, 0.0, 1.0)
        return (image * 255).astype(np.uint8)
