import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ==========================================
# CONFIGURATION CLASS
# ==========================================
class PipelineConfig:
    """
    Holds parameters for the point cloud processing pipeline.
    You can add, change or remove any of the parameters here.
    """
    def __init__(self):
        # Topic settings
        self.topic = '/oakd/points' # Topic containing the pointcloud
        
        # Voxel Downsampling
        self.voxel_size = 0.02
        
        # Passthrough/Box Filter (Min/Max XYZ)
        self.box_min = np.array([-1.0, -0.8, 0.2]) 
        self.box_max = np.array([ 1.0,  0.8, 2.0]) 

        # Plane RANSAC
        self.floor_dist = 0.02
        self.target_normal = np.array([0, 1, 0]) # Assuming Y-up for floor
        self.normal_thresh = 0.85
        
        # Cylinder RANSAC
        self.cyl_radius = 0.055
        self.max_cylinders = 3  

        # Increase plane RANSAC iterations
        self.plane_iters = 300

        # Increase cylinder RANSAC iterations  
        self.cyl_iters = 500

# ==========================================
# VISUALIZER CLASS
# ==========================================
class CylinderVisualizer:
    """
    Handles the creation and publishing of RViz MarkerArrays to represent 
    detected cylinders.
    """
    def __init__(self, publisher):
        self.pub_markers = publisher

    def create_cylinder_marker(self, center, radius, rgb, marker_id, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        
        m.pose.position.x = float(center[0])
        m.pose.position.y = float(0.0) # Snap to floor level for visualization
        m.pose.position.z = float(center[2])
        
        # Rotate cylinder to stand upright
        m.pose.orientation.x = 0.7071
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 0.7071
        
        m.scale.x = float(radius * 2.0)
        m.scale.y = float(radius * 2.0)
        m.scale.z = 0.4 
        
        m.color.r = float(rgb[0])
        m.color.g = float(rgb[1])
        m.color.b = float(rgb[2])
        m.color.a = 0.8
        return m

    def publish_viz(self, cylinders, frame_id):
        ma = MarkerArray()
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        ma.markers.append(clear_marker)

        for i, (model, rgb, name) in enumerate(cylinders):
            center, _, radius = model
            marker = self.create_cylinder_marker(center, radius, rgb, 2000 + i, frame_id)
            ma.markers.append(marker)
        
        self.pub_markers.publish(ma)

# ==========================================
# PIPELINE LOGIC 
# ==========================================
class CylinderPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
    
    def rgb_to_hsv(self, r, g, b):
        """
        Converts a single RGB point to HSV color space.
        
        :param r: Red component (0.0 - 1.0)
        :param g: Green component (0.0 - 1.0)
        :param b: Blue component (0.0 - 1.0)
        :return: Tuple (h, s, v) where H is [0, 360], S and V are [0, 1]
        """
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        
        # Calculate Hue
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
            
        # Calculate Saturation
        s = 0 if mx == 0 else (df / mx)
        
        # Calculate Value
        v = mx
        
        return h, s, v

    def get_neighbors(self, pts, queries, k=15):
        """
        Calculates k-nearest neighbors using a KDTree.
        
        :param pts: The source point cloud (Nx3).
        :param queries: The points for which we want neighbors (Mx3).
        :param k: Number of neighbors to find.
        :return: Indices of neighbors in the 'pts' array.
        """
        if len(pts) < k: return None
        tree = cKDTree(pts)
        _, idxs = tree.query(queries, k=k)
        return idxs

    def box_filter(self, pts, colors):

        mask = (
            (pts[:, 0] >= self.cfg.box_min[0]) & (pts[:, 0] <= self.cfg.box_max[0]) &
            (pts[:, 1] >= self.cfg.box_min[1]) & (pts[:, 1] <= self.cfg.box_max[1]) &
            (pts[:, 2] >= self.cfg.box_min[2]) & (pts[:, 2] <= self.cfg.box_max[2])
        )
        return pts[mask], colors[mask]

    def downsample(self, pts, colors):

        voxel_coords = np.floor(pts / self.cfg.voxel_size).astype(int)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        return pts[unique_idx], colors[unique_idx]

    def estimate_normals(self, pts, k=15):
        normals = np.zeros_like(pts)
        
        # Get k nearest neighbors for every point at once
        idxs = self.get_neighbors(pts, pts, k=k)
        if idxs is None:
            return normals
        
        for i in range(len(pts)):
            # Get neighbors and center them
            neighbors = pts[idxs[i]]
            centered = neighbors - neighbors.mean(axis=0)
            
            # SVD - normal is last row of Vt (smallest singular value)
            _, _, Vt = np.linalg.svd(centered)
            normals[i] = Vt[-1]
        
        return normals

    def find_plane_ransac(self, pts, iters=100):

        best_inliers = None
        best_count = 0

        for _ in range(iters):
            # Step 1: Sample 3 random points
            idx = np.random.choice(len(pts), 3, replace=False)
            p1, p2, p3 = pts[idx]

            # Step 2: Compute plane normal using cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-6:
                continue  # degenerate case, skip
            normal = normal / norm_len

            # Step 3: Check if normal is vertical (reject walls)
            alignment = abs(np.dot(normal, self.cfg.target_normal))
            if alignment < self.cfg.normal_thresh:
                continue

            # Step 4: Count inliers
            d = -np.dot(normal, p1)
            distances = np.abs(pts @ normal + d)
            inlier_mask = distances < self.cfg.floor_dist
            count = np.sum(inlier_mask)

            # Step 5: Save if best
            if count > best_count:
                best_count = count
                best_inliers = inlier_mask

        # Return points with floor removed
        if best_inliers is not None:
            return pts[~best_inliers], best_inliers
        return pts, None
    
    def cluster_points(self, pts, colors, eps=0.1, min_pts=10, max_pts=2000):
        if len(pts) == 0:
            return []
        
        # Build KDTree for fast neighbor search
        tree = cKDTree(pts)
        
        visited = np.zeros(len(pts), dtype=bool)
        clusters = []

        for i in range(len(pts)):
            if visited[i]:
                continue
            
            # Find all neighbors within eps distance
            neighbors = tree.query_ball_point(pts[i], r=eps)
            
            if len(neighbors) < 2:
                visited[i] = True
                continue
            
            # Grow cluster using BFS
            cluster_idx = []
            queue = list(neighbors)
            
            while queue:
                idx = queue.pop(0)
                if visited[idx]:
                    continue
                visited[idx] = True
                cluster_idx.append(idx)
                
                # Find neighbors of this point too
                new_neighbors = tree.query_ball_point(pts[idx], r=eps)
                queue.extend(new_neighbors)
            
            # Filter by size
            if min_pts <= len(cluster_idx) <= max_pts:
                clusters.append((
                    pts[np.array(cluster_idx)],
                    colors[np.array(cluster_idx)]
                ))
        
        return clusters

    def find_single_cylinder(self, pts, normals, iters=300):
        best_inliers = None
        best_count = 0
        best_model = None

        for _ in range(iters):
            # Step 1: Sample 2 random points and their normals
            idx = np.random.choice(len(pts), 2, replace=False)
            p1, p2 = pts[idx]
            n1, n2 = normals[idx]

            # Step 2: Axis = cross product of normals
            axis = np.cross(n1, n2)
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-6:
                continue
            axis = axis / axis_len

            # Step 3: Check axis is roughly vertical (Y axis for OAK-D)
            if abs(axis[1]) < 0.7:
                continue

            # Step 4: Count inliers
            # Vector from p1 to every point
            vec = pts - p1
            # Project onto axis
            proj = np.dot(vec, axis).reshape(-1, 1) * axis
            # Perpendicular distance to axis
            perp = vec - proj
            dist = np.linalg.norm(perp, axis=1)

            # Inliers are points close to the target radius
            inlier_mask = np.abs(dist - self.cfg.cyl_radius) < 0.02
            count = np.sum(inlier_mask)

            # Step 5: Save if best
            if count > best_count:
                best_count = count
                best_inliers = inlier_mask
                center = p1 + np.dot(p2 - p1, axis) * axis
                best_model = (center, axis, self.cfg.cyl_radius)

        if best_model is not None and best_count > 50:
            return best_model, best_inliers
        return None, None
    
    def classify_color(self, colors):
        avg_r = np.mean(colors[:, 0])
        avg_g = np.mean(colors[:, 1])
        avg_b = np.mean(colors[:, 2])
        h, s, v = self.rgb_to_hsv(avg_r, avg_g, avg_b)
        print(f"HSV: h={h:.1f}, s={s:.2f}, v={v:.2f}")
        if s < 0.2:
            return "Unknown", [0.5, 0.5, 0.5]
        if h < 15 or h > 345:
            return "Red", [1.0, 0.0, 0.0]
        elif 40 < h < 180:
            return "Green", [0.0, 1.0, 0.0]
        elif 180 < h < 270:
            return "Blue", [0.0, 0.0, 1.0]
        elif 270 < h < 345:
            return "Pink", [1.0, 0.4, 0.7]
        else:
            return "Unknown", [0.5, 0.5, 0.5]

    

# ==========================================
# ROS NODE
# ==========================================
class CylinderProcessorNode(Node):
    def __init__(self):
        super().__init__('cylinder_processor_node')
        self.cfg = PipelineConfig()
        self.pipeline = CylinderPipeline(self.cfg)
        self.last_time = 0
        self.min_interval = 0.5  # process only every 0.5 seconds (2Hz)
        
        # Publishers for debugging the pipeline stages in RViz
        self.pub_stage0 = self.create_publisher(PointCloud2, 'pipeline/stage0_box', 10)
        self.pub_stage3 = self.create_publisher(PointCloud2, 'pipeline/stage3_candidates', 10)
        
        # Marker publisher for the final detection results
        marker_pub = self.create_publisher(MarkerArray, 'viz/detections', 10)
        self.visualizer = CylinderVisualizer(marker_pub)
        
        self.sub = self.create_subscription(PointCloud2, self.cfg.topic, self.listener_callback, 10)
        
    def numpy_to_pc2_rgb(self, pts, colors, frame_id):
        """
        Converts Nx3 XYZ coordinates and Nx3 RGB color arrays into a ROS 2 PointCloud2 message.
        
        This utility handles the conversion of floating-point spatial data and the packing
        of three 8-bit color channels (R, G, B) into a single 32-bit float field, which is 
        the standard format for RGB point clouds in ROS and RViz.

        :param pts: A numpy array of shape (N, 3) containing [x, y, z] coordinates.
        :param colors: A numpy array of shape (N, 3) containing [r, g, b] values (0.0 to 1.0).
        :param frame_id: The TF frame string (e.g., 'camera_link') for the message header.
        :return: A sensor_msgs/PointCloud2 message ready for publishing.
        """
        msg = PointCloud2()
        msg.header.frame_id, msg.height, msg.width = frame_id, 1, len(pts)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian, msg.point_step, msg.is_dense = False, 16, True
        msg.row_step = 16 * len(pts)
        c = (np.clip(colors, 0, 1) * 255).astype(np.uint32)
        rgb_packed = (255 << 24) | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        data = np.hstack([pts.astype(np.float32), rgb_packed.view(np.float32).reshape(-1, 1)])
        msg.data = data.tobytes()
        return msg
    
    def listener_callback(self, msg):
        """
        Main ROS Callback. Orchestrates the flow from PointCloud2 to Cylinder detection.
        """
        import time
        current_time = time.time()
        if current_time - self.last_time < self.min_interval:
            return
        self.last_time = current_time
        frame_id = msg.header.frame_id
        stride = msg.point_step // 4 
        raw_data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, stride)
        
        # 1. Extraction: Get XYZ points and Filter NaNs
        pts = raw_data[:, :3]
        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite_mask]
        
        # 2. Color Extraction: Decode packed float32 RGB values
        rgb_uint32 = raw_data[finite_mask, 4].view(np.uint32)
        raw_colors = np.vstack([
            ((rgb_uint32 >> 16) & 0xFF) / 255.0, # Red
            ((rgb_uint32 >> 8) & 0xFF) / 255.0,  # Green
            (rgb_uint32 & 0xFF) / 255.0          # Blue
        ]).T

        #box_filter
        pts_box, colors_box = self.pipeline.box_filter(pts, raw_colors)
       
        #downsample
        pts_v, colors_v = self.pipeline.downsample(pts_box, colors_box)
    

        #ransac
        pts_no_floor, floor_mask = self.pipeline.find_plane_ransac(pts_v, iters=300)
        colors_no_floor = colors_v[~floor_mask] if floor_mask is not None else colors_v
        
        #visulization

        pc_msg = self.numpy_to_pc2_rgb(pts_no_floor, colors_no_floor, frame_id)
        self.pub_stage0.publish(pc_msg)

        # 6. Cluster points
        clusters = self.pipeline.cluster_points(pts_no_floor, colors_no_floor, 
                                                eps=0.15, min_pts=50, max_pts=2000)
        print(f"Found {len(clusters)} clusters")

        # After clustering
        cylinders_found = []
        for cluster_pts, cluster_colors in clusters:
            # Estimate normals for this cluster
            normals = self.pipeline.estimate_normals(cluster_pts)
            
            # Find cylinder
            model, inliers = self.pipeline.find_single_cylinder(cluster_pts, normals, iters=500)
            
            if model is not None:
                inlier_colors = cluster_colors[inliers]
                name, rgb = self.pipeline.classify_color(inlier_colors)
                cylinders_found.append((model, rgb, name))
                print(f"Cylinder found: {name} at {model[0]}")

        self.visualizer.publish_viz(cylinders_found, frame_id)

def main():
    rclpy.init()
    node = CylinderProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()