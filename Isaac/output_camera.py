import tools
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils import prims as prims_utils, stage as stage_utils
import numpy as np
import cv2
import omni.replicator.core as rep


stage_utils.open_stage("/media/avent/DATA/IsaacAssets/Collected_warehouse_trailer/warehouse_trailer.usd")

tools.app_update(10)

def get_cameras(camera_prim_paths: list[str]):
    camera_prims = []
    for prim_path in camera_prim_paths:
        camera_prims.append(Camera(prim_path=prim_path))
    
    return camera_prims


from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

# world = World(stage_units_in_meters=1.0)






# import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera

camera_path = "/World/lola/cameras/left_01/camera"
render_product = rep.create.render_product(camera_path, resolution=(960, 600))

camera = Camera(prim_path=camera_path, resolution=(960, 600), render_product_path=render_product.path)

camera.initialize()

tools.app_update(20)

# rgba = camera.get_rgba()
# rgb = rgba[..., :3].copy()
rgb = camera.get_rgb()
if rgb is None:
    print("rgb is None!")
else:
    print(rgb.shape)

tools.app_loop()