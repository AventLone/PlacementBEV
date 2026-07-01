import cv2
import numpy as np

def compute_ipm_homography(K, R, t, bev_resolution, bev_width, bev_height, world_center_x, world_center_y):
    """
    Computes the Homography matrix mapping 2D image pixels to a global 2D BEV grid.
    
    bev_resolution: meters per pixel (e.g., 0.05 means 1 pixel = 5cm)
    world_center_x, world_center_y: The physical (X, Y) world position mapped to the center of the BEV image.
    """
    # 1. Build Extrinsic Projection matrix (3x4)
    Rt = np.hstack((R, t.reshape(3, 1)))
    
    # 2. Build the full Camera Projection Matrix P (3x4) -> links World 3D to Image 2D
    P = K @ Rt
    
    # 3. Extract columns corresponding to X, Y, and T (Drop Z since Z=0 for flat ground)
    # P_homo links World Ground (X_w, Y_w, 1) to Image Pixel (u, v, 1)
    P_homo = P[:, [0, 1, 3]]
    
    # 4. Define transformation from physical World Ground (meters) to target BEV Image Grid (pixels)
    # BEV_pixel_x = (World_x - world_center_x) / bev_resolution + (bev_width / 2)
    # BEV_pixel_y = -(World_y - world_center_y) / bev_resolution + (bev_height / 2)  # Inverted Y for image coordinates
    
    W_to_BEV = np.array([
        [1.0 / bev_resolution,  0.0,                  -world_center_x / bev_resolution + (bev_width / 2.0)],
        [0.0,                  -1.0 / bev_resolution,  world_center_y / bev_resolution + (bev_height / 2.0)],
        [0.0,                   0.0,                   1.0]
    ])
    
    # 5. Invert it to map BEV Image Grid coordinates back to physical World Ground
    BEV_to_W = np.linalg.inv(W_to_BEV)
    
    # 6. Final Homography maps BEV pixel positions -> World Ground -> Camera Image Pixel
    H_bev_to_img = P_homo @ BEV_to_W
    
    return H_bev_to_img

def generate_multi_camera_bev(camera_images: dict, camera_calibrations, config):
    """
    Stitches 4 camera images into a single BEV image.
    
    camera_images: Dictionary of {'front': img, 'back': img, 'left': img, 'right': img}
    camera_calibrations: Dictionary containing {'K', 'R', 't'} matrices for each camera.
    config: Dictionary with 'width', 'height', 'resolution', 'center_x', 'center_y'
    """
    w, h = config['width'], config['height']
    
    # Initialize the output master BEV canvas and a weight map for smooth blending
    master_bev = np.zeros((h, w, 3), dtype=np.uint8)
    weight_mask = np.zeros((h, w), dtype=np.float32)
    
    for cam_name, img in camera_images.items():
        calib = camera_calibrations[cam_name]
        
        # Calculate Homography matrix
        H = compute_ipm_homography(
            calib['K'], calib['R'], calib['t'],
            config['resolution'], w, h, 
            config['center_x'], config['center_y']
        )
        
        # Warp individual camera view directly into the master BEV coordinate system
        # We use WARP_INVERSE_MAP because H maps from Target (BEV) to Source (Image)
        warped_view = cv2.warpPerspective(
            img, H, (w, h), 
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        
        # Create a binary mask of where this specific camera provides visual data
        gray_warped = cv2.cvtColor(warped_view, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 1, 1, cv2.THRESH_BINARY)
        
        # Accumulate image content and update valid pixel tracking
        # For overlapping zones, this smoothly blends pixel intensities
        master_bev = cv2.add(master_bev, warped_view) 
        weight_mask += mask.astype(np.float32)

    # Normalize overlapping camera regions to prevent overexposure lines where cameras cross
    weight_mask[weight_mask == 0] = 1.0  # Prevent zero-division
    normalized_bev = (master_bev / weight_mask[:, :, np.newaxis]).astype(np.uint8)
    
    return normalized_bev

# ==========================================
# EXAMPLE CONFIGURATION & USAGE
# ==========================================
bev_config = {
    'width': 800,       # Size of output image in pixels
    'height': 800,      
    'resolution': 0.05, # 1 pixel = 0.05 meters (5 cm). Total area = 40m x 40m
    'center_x': 0.0,    # Vehicle center X in world coordinates
    'center_y': 0.0     # Vehicle center Y in world coordinates
}

# Dummy placeholder data for 1 camera to show structure (Fill with real K, R, t)
example_calibrations = {
    'front': {
        'K': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
        'R': np.eye(3, dtype=np.float32), # Replace with actual rotation matrix
        't': np.array([0, 1.5, 2.0], dtype=np.float32) # Camera height/offset from ground
    },
    # Repeat format for 'back', 'left', 'right'...
}
