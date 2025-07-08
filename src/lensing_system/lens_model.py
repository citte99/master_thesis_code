import torch
import torch.nn as nn
from .lens_mass_components import NFW, PEMD, SIS, ExternalPotential
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPoint, Point
from scipy.spatial import ConvexHull
from shared_utils import units

class LensModel(nn.Module):
    def __init__(self, config_dict, precomp_dict, device):
        """
        Initialize the LensModel with its mass components.
        """
        super().__init__()
        self.device = device
        self.precomp_dict = precomp_dict
        self.mass_components = []
        self.add_mass_component(config_dict["mass_components"])

        self.jacobian = None
        self.A = None

    def add_mass_component(self, config_dictionaries):
        type_mapping = {
            "NFW": NFW,
            "PEMD": PEMD,
            "SIS": SIS,
            "ExternalPotential": ExternalPotential
        }
        for config in config_dictionaries:
            mass_type = config.get("type")
            params = config.get("params")
            if mass_type not in type_mapping:
                raise ValueError(f"Unknown mass component type: {mass_type}")
            mass_component = type_mapping[mass_type](params, self.device)
            self.mass_components.append(mass_component)

    def deflection_field(self, lens_grid, precomp_dict=None, z_source=None):
        # Working with precomputed distances instead of z_source
        precomp_dict = self.precomp_dict

        total_deflection_field = torch.zeros_like(lens_grid, device=self.device)
        for component in self.mass_components:
            total_deflection_field = total_deflection_field + component.deflection_angle(lens_grid, precomp_dict=precomp_dict)
        return total_deflection_field

    def jacobian_deflection(self, lens_grid, precomp_dict=None, z_source=None, mask=None, shift=1e-9, return_jacobian=False):
        #working with precomputed distances instead of z_source
        precomp_dict = self.precomp_dict
        '''
        Compute the Jacobian of the deflection field using finite differences.
        
        Assumes:
        - lens_grid is of shape [n, n, 2] with the last dim = [x1, x2].
        The Jacobian J is defined as:
        J = [[dy1/dx1, dy1/dx2],
            [dy2/dx1, dy2/dx2]]
            
        If a mask (of shape [n, n]) is provided, the function computes the derivatives 
        only for the pixels where mask==True, and then fills the rest of the output with NaN.
        
        Returns:
        - self.jacobian: Tensor of shape [n, n, 2, 2] where for each grid point,
            jacobian[i,j, :, 0] ~ derivative with respect to x1 and
            jacobian[i,j, :, 1] ~ derivative with respect to x2.
        '''
        n1, n2, _ = lens_grid.shape
        # Create an all-on mask if none is provided.
        if mask is None:
            mask = torch.ones(n1, n2, dtype=torch.bool, device=lens_grid.device)

        # Extract only the active pixels.
        active_points = lens_grid[mask]  # shape: [N_active, 2]

        # Create finite difference shift tensors.
        shift_tensor_x = torch.tensor([shift, 0.0], device=lens_grid.device, dtype=lens_grid.dtype)
        shift_tensor_y = torch.tensor([0.0, shift], device=lens_grid.device, dtype=lens_grid.dtype)
        # Compute deflection for shifted active points.
        higher_x = self.deflection_field(active_points + shift_tensor_x, precomp_dict=precomp_dict)
        lower_x  = self.deflection_field(active_points - shift_tensor_x, precomp_dict=precomp_dict)
        higher_y = self.deflection_field(active_points + shift_tensor_y, precomp_dict=precomp_dict)
        lower_y  = self.deflection_field(active_points - shift_tensor_y, precomp_dict=precomp_dict)
        # Compute the finite differences.
        d_alpha_dx = (higher_x - lower_x) / (2 * shift)
        d_alpha_dy = (higher_y - lower_y) / (2 * shift)

        # Stack the derivatives to form the Jacobian for active pixels.
        active_jacobian = torch.stack([d_alpha_dx, d_alpha_dy], dim=-1)  # shape: [N_active, 2, 2]

        # Create a full-grid tensor and fill with NaN.
        full_jacobian = torch.empty(n1, n2, 2, 2, device=lens_grid.device, dtype=lens_grid.dtype)
        full_jacobian.fill_(float('nan'))

        # Place the computed Jacobians back into the corresponding positions.
        full_jacobian[mask] = active_jacobian
        if return_jacobian:
            return full_jacobian
        else:
            self.jacobian = full_jacobian
        return 0
    
    def compute_A(self, lens_grid, return_A=False, jac=None):
        if self.jacobian is None and jac is None:
            self.jacobian_deflection(lens_grid)
            jac=self.jacobian
        elif jac is not None:
            jac=jac
        elif self.jacobian is not None:
            jac=self.jacobian
        else:
            raise ValueError("Jacobian not computed")
        # Create an identity matrix of shape [n, n, 2, 2]
        n1, n2, _ = lens_grid.shape
        I = torch.eye(2, device=lens_grid.device, dtype=lens_grid.dtype)
        I = I.unsqueeze(0).unsqueeze(0).expand(n1, n2, 2, 2)
        A = I - jac
        if return_A:
            return A
        else:
            self.A = A
        return 0
    

    def compute_convergence(self, lens_grid):
        if self.jacobian is None:
            self.jacobian_deflection(lens_grid)
        convergence = 0.5 * (self.jacobian[..., 0, 0] + self.jacobian[..., 1, 1])
        return convergence


    def compute_magnification(self, lens_grid):
        if self.A is None:
            self.compute_A(lens_grid)
        magnification = 1.0 / torch.det(self.A)
        return magnification

    def compute_critical_curve(self, lens_grid, iterations=3):
        # Create an initial coarse grid with 2^7 steps per side.
        half_width_image = torch.abs(lens_grid[0, 0, 0])
        steps = 2**7
        xx = torch.linspace(-half_width_image, half_width_image, steps=steps,
                            dtype=lens_grid.dtype, device=lens_grid.device)
        yy = torch.linspace(-half_width_image, half_width_image, steps=steps,
                            dtype=lens_grid.dtype, device=lens_grid.device)
        critical_grid = torch.stack(torch.meshgrid(xx, yy, indexing='xy'), dim=2)  # shape: [steps, steps, 2]
        # Compute the Jacobian, A, and then det(A) on the coarse grid.
        jac=self.jacobian_deflection(critical_grid, return_jacobian=True)
        A=self.compute_A(critical_grid, jac=jac, return_A=True)
        detA = torch.det(A)  # shape: [steps, steps]
        # Detect sign flips in detA along horizontal and vertical directions.
        flip_h = (detA[:, :-1] * detA[:, 1:] < 0)
        flip_v = (detA[:-1, :] * detA[1:, :] < 0)
        flipping_pixels_mask = torch.zeros_like(detA, dtype=torch.bool)
        flipping_pixels_mask[:, :-1] |= flip_h
        flipping_pixels_mask[:-1, :] |= flip_v
        def finer_grid(previous_grid):
            n1, n2, _ = previous_grid.shape
            xx = torch.linspace(previous_grid[0, 0, 0], previous_grid[0, -1, 0],
                                steps=n2 * 2, device=previous_grid.device, dtype=previous_grid.dtype)
            yy = torch.linspace(previous_grid[0, 0, 1], previous_grid[-1, 0, 1],
                                steps=n1 * 2, device=previous_grid.device, dtype=previous_grid.dtype)
            return torch.stack(torch.meshgrid(xx, yy, indexing='xy'), dim=2)
        
        # Iteratively refine the grid and update the mask.
        for i in range(iterations):
            flipping_pixels_mask = (F.max_pool2d(flipping_pixels_mask.unsqueeze(0).unsqueeze(0).float(),
                                     kernel_size=3, stride=1, padding=1) > 0).squeeze(0).squeeze(0)

            critical_grid = finer_grid(critical_grid)
            # Upsample the previous flipping mask to the new grid resolution.
            mask = F.interpolate(flipping_pixels_mask.unsqueeze(0).unsqueeze(0).float(),
                                size=(critical_grid.shape[0], critical_grid.shape[1]),
                                mode='nearest').bool().squeeze()
            jac=self.jacobian_deflection(critical_grid, mask=mask, return_jacobian=True)
            A=self.compute_A(critical_grid, jac=jac, return_A=True)
            detA = torch.det(A)
            flip_h = (detA[:, :-1] * detA[:, 1:] < 0)
            flip_v = (detA[:-1, :] * detA[1:, :] < 0)
            new_mask = torch.zeros_like(detA, dtype=torch.bool)
            new_mask[:, :-1] |= flip_h
            new_mask[:-1, :] |= flip_v
            flipping_pixels_mask = new_mask
       

        # Return the final refined grid and the mask image (1 where critical curve is, 0 elsewhere).
        return flipping_pixels_mask, critical_grid

    def compute_caustics(self, lens_grid, iterations=2, verbose=True):
        """
        Computes the caustics by clustering critical-curve (flipping) pixels 
        directly in the lens plane, then ray tracing the clustered points to the source plane.
        
        Returns:
        caustic_mask: A binary mask (on a 500x500 grid) marking pixels in the source plane 
                        where caustic points are present.
        source_grid: The 500x500 grid over the source plane.
        caustic_points: The critical points (ray-traced from the lens plane) in the source plane.
        labels: The DBSCAN labels corresponding to each critical point (from the lens plane).
        clusters: A dict mapping each cluster label to a (N,2) NumPy array of polygon vertices 
                    (in source plane coordinates) computed via convex hull.
        """
        # 1. Compute the critical curve (flipping pixels) on the lens plane.
        flipping_pixels_mask, critical_grid = self.compute_critical_curve(lens_grid, iterations=iterations)
        # 2. Extract the critical points in the lens plane.
        critical_points = critical_grid[flipping_pixels_mask]  # tensor of shape [N, 2]
        critical_points_np = critical_points.cpu().numpy()
        
        if critical_points_np.size == 0:
            # Define a default point at [0,0]
            default_point = np.array([[0, 0]])
            default_point_tensor = torch.tensor(default_point, device=critical_points.device, dtype=critical_points.dtype)
            
            # Create a default 500x500 source grid centered around 0.
            # (Here, we choose a grid spanning from -1 to 1 in both directions; adjust as needed.)
            grid_x = torch.linspace(-1, 1, 500, device=default_point_tensor.device, dtype=default_point_tensor.dtype)
            grid_y = torch.linspace(-1, 1, 500, device=default_point_tensor.device, dtype=default_point_tensor.dtype)
            X, Y = torch.meshgrid(grid_x, grid_y, indexing='xy')
            source_grid = torch.stack((X, Y), dim=2)
            
            # Create a binary mask and mark the pixel nearest to [0, 0]
            caustic_mask = torch.zeros_like(X, dtype=torch.bool)
            grid_x_np = grid_x.cpu().numpy()
            grid_y_np = grid_y.cpu().numpy()
            i = np.argmin(np.abs(grid_x_np - 0))
            j = np.argmin(np.abs(grid_y_np - 0))
            caustic_mask[j, i] = True
            
            # Default labels and clusters
            labels = np.array([0])
            clusters = {0: default_point}
            
            return caustic_mask, source_grid, default_point_tensor, labels, clusters
        
        # 3. Cluster these lens-plane points using DBSCAN.
        spread = max(
            critical_points_np[:, 0].max() - critical_points_np[:, 0].min(),
            critical_points_np[:, 1].max() - critical_points_np[:, 1].min()
        )
        eps = spread * 0.05  # adjust as needed
        min_samples = 4   # minimum points to form a cluster
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(critical_points_np)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        if verbose:
            print("Found {} clusters (expected caustics)".format(len(unique_labels)))
        
        # 4. Ray trace the lens-plane critical points to the source plane.
        #    (Assuming self.source_position() maps points from the lens plane to the source plane.)
        
        caustic_points = self.forward(critical_points)  # tensor of shape [N, 2]

        caustic_points_np = caustic_points.cpu().numpy()
        
        # 5. Build a 500x500 grid on the source plane spanning the ray-traced caustic points.
        x_min, x_max = caustic_points[:, 0].min(), caustic_points[:, 0].max()
        y_min, y_max = caustic_points[:, 1].min(), caustic_points[:, 1].max()
        margin_x = (x_max - x_min) / 2.0
        margin_y = (y_max - y_min) / 2.0
        

        grid_x = torch.linspace(x_min - margin_x, x_max + margin_x, 500, 
                                device=caustic_points.device, dtype=caustic_points.dtype)
        grid_y = torch.linspace(y_min - margin_y, y_max + margin_y, 500, 
                                device=caustic_points.device, dtype=caustic_points.dtype)
        X, Y = torch.meshgrid(grid_x, grid_y, indexing='xy')
        source_grid = torch.stack((X, Y), dim=2)
        
        # 6. Build a binary mask on the source plane grid by marking the nearest pixel for each caustic point.
        caustic_mask = torch.zeros_like(X, dtype=torch.bool)
        grid_x_np = grid_x.cpu().numpy()
        grid_y_np = grid_y.cpu().numpy()
        """
        following:performance improvement
        """
        # for pt in caustic_points_np:
        #     i = np.argmin(np.abs(grid_x_np - pt[0]))
        #     j = np.argmin(np.abs(grid_y_np - pt[1]))
        #     caustic_mask[j, i] = True
        # Using np.searchsorted for a vectorized approach
        i_indices = np.clip(np.searchsorted(grid_x_np, caustic_points_np[:, 0]), 0, len(grid_x_np)-1)
        j_indices = np.clip(np.searchsorted(grid_y_np, caustic_points_np[:, 1]), 0, len(grid_y_np)-1)
        caustic_mask[j_indices, i_indices] = True
        # Depending on your coordinate convention, you might want to flip the mask:
        #caustic_mask = caustic_mask.flip(0)        
        # 7. For each cluster (from the lens plane), compute the convex hull in the source plane.
        clusters = {}
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) == 0:
                continue
            # Get the corresponding source plane points for this cluster.
            cluster_points = caustic_points_np[indices]
            if cluster_points.shape[0] >= 3:
                hull = ConvexHull(cluster_points)
                polygon = cluster_points[hull.vertices]
            else:
                polygon = cluster_points  # too few points; return as is.
            clusters[label] = polygon
        
        return caustic_mask, source_grid, caustic_points, labels, clusters

    def random_pos_inside_caustics(self, lens_grid, max_distance_arcsec=0, max_distance_std_caustic=False, max_iter=1000, verbose=False):
        """
        Samples a random position within the bounding box defined by the caustic clusters,
        extended by max_distance on all sides.
        
        The candidate is accepted if it is either inside any caustic polygon or if its 
        distance to any caustic is less than or equal to max_distance.
        
        For concave caustics, alphashape is used to better represent their boundaries.
        
        If max_distance_arcsec is 0 and max_distance_std_caustic is True, the standard deviation
        of the largest caustic is used as the max_distance.
        
        When verbose=True, a visualization of the caustics, sampling region, and selected point
        (or lack thereof) is saved to 'caustics_visualization.png'.
        
        Parameters:
            lens_grid: Grid for lens computations
            max_distance_arcsec (float): Maximum allowed distance from a caustic boundary in arcseconds.
            max_distance_std_caustic (bool): If True and max_distance_arcsec is 0, use the std of 
                                        the largest caustic as the max_distance.
            max_iter (int): Maximum number of sampling attempts.
            
        Returns:
            pos (tuple): A tuple (x, y) representing the accepted position,
                        or None if no valid position was found after max_iter attempts.
        """
        # Initialize max_distance based on max_distance_arcsec
        max_distance = units._arcsec_to_rad(max_distance_arcsec)
        frac_of_std_distance = 0.35
        
        # 1. Compute the caustics
        _, _, _, _, clusters = self.compute_caustics(lens_grid, verbose=verbose)
        
        # Track caustic properties but don't discard any yet
        cluster_vertex_counts = {}
        cluster_variances = {}
        
        for label, poly in clusters.items():
            if poly.shape[0] > 0:
                # Store vertex count
                vertex_count = poly.shape[0]
                cluster_vertex_counts[label] = vertex_count
                
                # Calculate variance for potential use in max_distance
                var_x = np.var(poly[:, 0])
                var_y = np.var(poly[:, 1])
                total_var = var_x + var_y
                cluster_variances[label] = total_var
        
        # 2. Create Shapely polygons for each cluster, using alphashape for concave hulls
        shapely_polys = []
        simplified_clusters = {}
        
        for label, poly_pts in clusters.items():
            if poly_pts.shape[0] >= 3:
                # Only use original points for smaller clusters (less complex shapes)
                if poly_pts.shape[0] <= 20:
                    shapely_poly = Polygon(poly_pts)
                    shapely_polys.append(shapely_poly)
                    simplified_clusters[label] = poly_pts
                else:
                    # For caustics with many vertices, use alphashape for better concave representation
                    try:
                        import alphashape
                        alpha = 0.5  # Adjust alpha as needed - smaller values create tighter fits
                        concave = alphashape.alphashape(poly_pts, alpha)
                        
                        # Handle multipolygons
                        if concave.geom_type == 'MultiPolygon':
                            concave = max(concave.geoms, key=lambda p: p.area)
                        
                        shapely_polys.append(concave)
                        
                        # Extract coordinates from the alphashape polygon for simplified_clusters
                        if hasattr(concave, 'exterior'):
                            simplified_coords = np.array(concave.exterior.coords)
                            simplified_clusters[label] = simplified_coords
                    except ImportError:
                        # Fallback to convex hull if alphashape is not available
                        if verbose:
                            print("alphashape not available, using convex hull instead")
                        shapely_poly = Polygon(poly_pts)
                        shapely_polys.append(shapely_poly)
                        simplified_clusters[label] = poly_pts
            else:
                # Too few points: create a MultiPoint and use its convex hull
                hull = MultiPoint(poly_pts).convex_hull
                shapely_polys.append(hull)
                
                # Handle different geometry types that can result from convex_hull
                if hull.geom_type == 'Polygon':
                    simplified_clusters[label] = np.array(hull.exterior.coords)
                elif hull.geom_type == 'LineString':
                    simplified_clusters[label] = np.array(hull.coords)
                elif hull.geom_type == 'Point':
                    # For a single point, create a tiny square around it
                    x, y = hull.x, hull.y
                    epsilon = 1e-6  # Small offset
                    simplified_clusters[label] = np.array([[x-epsilon, y-epsilon], 
                                                        [x+epsilon, y-epsilon],
                                                        [x+epsilon, y+epsilon], 
                                                        [x-epsilon, y+epsilon]])
        
        # 3. Determine max_distance using std if requested and max_distance_arcsec is 0
        if max_distance_arcsec == 0 and max_distance_std_caustic and cluster_variances:
            # Get variances for all clusters
            if cluster_variances:
                # Get the maximum variance
                max_var_label = max(cluster_variances, key=cluster_variances.get)
                max_var = cluster_variances[max_var_label]
                # Use the square root of variance (std) as the max_distance
                max_distance = frac_of_std_distance * np.sqrt(max_var)
                if verbose:
                    print(f"Using std of largest caustic as max_distance: {max_distance}")
        
        # 4. Compute the overall bounds from all simplified clusters
        all_vertices = []
        for poly in simplified_clusters.values():
            if poly.shape[0] > 0:
                all_vertices.append(poly)
        
        if not all_vertices:
            print("No caustic vertices provided.")
            return None
        
        all_points = np.concatenate(all_vertices, axis=0)
        x_min = np.min(all_points[:, 0])
        x_max = np.max(all_points[:, 0])
        y_min = np.min(all_points[:, 1])
        y_max = np.max(all_points[:, 1])
        
        # Extend the bounds by max_distance
        x_bounds = (x_min - max_distance, x_max + max_distance)
        y_bounds = (y_min - max_distance, y_max + max_distance)
        
        # 5. Sample random positions in the computed bounding box
        for _ in range(max_iter):
            x = np.random.uniform(x_bounds[0], x_bounds[1])
            y = np.random.uniform(y_bounds[0], y_bounds[1])
            candidate = Point(x, y)
            
            # Check each caustic polygon
            for poly in shapely_polys:
                # Accept if candidate is inside the polygon...
                if poly.contains(candidate):
                    if verbose:
                        self._visualize_caustics_and_result(simplified_clusters, shapely_polys, max_distance, x_bounds, y_bounds, (x, y))
                    return (x, y)
                # ...or if its distance to the polygon is within max_distance
                if candidate.distance(poly) <= max_distance:
                    if verbose:
                        self._visualize_caustics_and_result(simplified_clusters, shapely_polys, max_distance, x_bounds, y_bounds, (x, y))
                    return (x, y)
        
        # If no valid point is found after max_iter attempts, return None
        if verbose:
            self._visualize_caustics_and_result(simplified_clusters, shapely_polys, max_distance, x_bounds, y_bounds, None)
        return None

    def _visualize_caustics_and_result(self, clusters, shapely_polys, max_distance, x_bounds, y_bounds, selected_point=None):
        """
        Visualize the caustics, the sampling region, and the selected point.
        
        Parameters:
            clusters (dict): Dictionary of caustic clusters.
            shapely_polys (list): List of Shapely polygons representing the caustics.
            max_distance (float): Maximum distance from caustic boundary.
            x_bounds (tuple): (min_x, max_x) bounds for sampling.
            y_bounds (tuple): (min_y, max_y) bounds for sampling.
            selected_point (tuple): (x, y) coordinates of the selected point, or None if no point was selected.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        from shapely.geometry import Point, LineString
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot bounding box
        rect = plt.Rectangle((x_bounds[0], y_bounds[0]), 
                            x_bounds[1] - x_bounds[0], 
                            y_bounds[1] - y_bounds[0],
                            fill=False, edgecolor='gray', linestyle='--', label='Sampling Region')
        ax.add_patch(rect)
        
        # Plot each caustic cluster with a different color
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        for (label, points), color in zip(clusters.items(), colors):
            if points.shape[0] > 0:
                ax.scatter(points[:, 0], points[:, 1], s=10, color=color, alpha=0.7, 
                        label=f'Cluster {label} ({points.shape[0]} vertices)')
        
        # Plot Shapely polygons for caustics
        patches = []
        for poly in shapely_polys:
            if poly.geom_type == 'Polygon':
                x, y = poly.exterior.xy
                patches.append(MplPolygon(np.column_stack([x, y]), True))
        
        p = PatchCollection(patches, alpha=0.3, facecolor='blue', edgecolor='black')
        ax.add_collection(p)
        
        # Plot expanded region around caustics (buffer)
        if max_distance > 0:
            buffer_patches = []
            for poly in shapely_polys:
                buffered = poly.buffer(max_distance)
                if buffered.geom_type == 'Polygon':
                    x, y = buffered.exterior.xy
                    buffer_patches.append(MplPolygon(np.column_stack([x, y]), True))
                elif buffered.geom_type == 'MultiPolygon':
                    for geom in buffered.geoms:
                        x, y = geom.exterior.xy
                        buffer_patches.append(MplPolygon(np.column_stack([x, y]), True))
            
            if buffer_patches:
                p_buffer = PatchCollection(buffer_patches, alpha=0.2, facecolor='green', edgecolor='green')
                ax.add_collection(p_buffer)
                ax.plot([], [], color='green', alpha=0.4, linewidth=10, label=f'Max Distance Buffer ({max_distance:.6f})')
        
        # Plot selected point if it exists
        if selected_point is not None:
            ax.scatter(selected_point[0], selected_point[1], s=100, color='red', 
                    marker='*', label='Selected Point')
        
        # Add a title indicating success or failure
        title = "Caustic Visualization: "
        if selected_point is not None:
            title += f"Successfully found point at {selected_point}"
        else:
            title += "Failed to find valid point after max iterations"
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        # Adjust the plot limits to show everything
        margin = max_distance * 1.1 if max_distance > 0 else 0.05 * (x_bounds[1] - x_bounds[0])
        ax.set_xlim(x_bounds[0] - margin, x_bounds[1] + margin)
        ax.set_ylim(y_bounds[0] - margin, y_bounds[1] + margin)
        
        plt.tight_layout()
        plt.savefig('caustics_visualization.png')
        plt.close()
    # #this version discards the biggest caustic
    # def random_pos_inside_caustics(self, lens_grid, max_distance_arcsec=0, max_iter=1000, verbose=False):
    #     """
    #     Samples a random position within the bounding box defined by the caustic clusters,
    #     extended by max_distance on all sides.
        
    #     The candidate is accepted if it is either inside any caustic polygon or if its 
    #     distance to any caustic is less than or equal to max_distance.
        
    #     If multiple caustic clusters are found, the one with the largest position variance
    #     is discarded.
        
    #     Parameters:
    #         clusters (dict): Dictionary mapping cluster labels to polygons.
    #                         Each polygon is represented as an (N,2) NumPy array of vertices.
    #         max_distance (float): Maximum allowed distance from a caustic boundary.
    #         max_iter (int): Maximum number of sampling attempts.
            
    #     Returns:
    #         pos (tuple): A tuple (x, y) representing the accepted position,
    #                     or None if no valid position was found after max_iter attempts.
    #     """
    #     max_distance = units._arcsec_to_rad(max_distance_arcsec)
    #     # 1. Compute the overall bounds from all clusters.
    #     #    We concatenate all vertices from all clusters.
    #     _, _, _, _, clusters = self.compute_caustics(lens_grid, verbose=verbose)
        
    #     # Filter out the cluster with largest variance if there are multiple clusters
    #     if len(clusters) > 1:
    #         # Calculate variance for each cluster
    #         cluster_variances = {}
    #         for label, poly in clusters.items():
    #             if poly.shape[0] > 0:
    #                 # Calculate the total variance (sum of variances along x and y axes)
    #                 var_x = np.var(poly[:, 0])
    #                 var_y = np.var(poly[:, 1])
    #                 total_var = var_x + var_y
    #                 cluster_variances[label] = total_var
            
    #         # Find the cluster with maximum variance
    #         max_var_label = max(cluster_variances, key=cluster_variances.get)
            
    #         # Remove the cluster with largest variance
    #         if verbose:
    #             print(f"Discarding cluster {max_var_label} with variance {cluster_variances[max_var_label]}")
    #         clusters.pop(max_var_label)
        
    #     all_vertices = []
    #     for poly in clusters.values():
    #         if poly.shape[0] > 0:
    #             all_vertices.append(poly)
    #     if not all_vertices:
    #         print("No caustic vertices provided.")
    #         return None
    #     all_points = np.concatenate(all_vertices, axis=0)
    #     x_min = np.min(all_points[:, 0])
    #     x_max = np.max(all_points[:, 0])
    #     y_min = np.min(all_points[:, 1])
    #     y_max = np.max(all_points[:, 1])
        
    #     # Extend the bounds by max_distance
    #     x_bounds = (x_min - max_distance, x_max + max_distance)
    #     y_bounds = (y_min - max_distance, y_max + max_distance)
        
    #     # 2. Convert each cluster's vertices to a Shapely polygon.
    #     #    If a cluster has fewer than 3 points, use its convex hull.
    #     shapely_polys = []
    #     for label, poly_pts in clusters.items():
    #         if poly_pts.shape[0] >= 3:
    #             # Option A: Use convex hull (or change to alphashape for a concave hull)
    #             shapely_polys.append(Polygon(poly_pts))
    #             # Option B (uncomment to use alphashape for concave hull):
    #             # alpha = 1.0  # Adjust as needed
    #             # concave = alphashape.alphashape(poly_pts, alpha)
    #             # if concave.geom_type == 'MultiPolygon':
    #             #     concave = max(concave.geoms, key=lambda p: p.area)
    #             # shapely_polys.append(concave)
    #         else:
    #             # Too few points: create a MultiPoint and use its convex hull.
    #             shapely_polys.append(MultiPoint(poly_pts).convex_hull)
        
    #     # 3. Sample random positions in the computed bounding box.
    #     for _ in range(max_iter):
    #         x = np.random.uniform(x_bounds[0], x_bounds[1])
    #         y = np.random.uniform(y_bounds[0], y_bounds[1])
    #         candidate = Point(x, y)
            
    #         # Check each caustic polygon.
    #         for poly in shapely_polys:
    #             # Accept if candidate is inside the polygon...
    #             if poly.contains(candidate):
    #                 return (x, y)
    #             # ...or if its distance to the polygon is within max_distance.
    #             if candidate.distance(poly) <= max_distance:
    #                 return (x, y)
        
    #     # If no valid point is found after max_iter attempts, return None.
    #     return None
    #this version did not exclude the biggest caustic
    # def random_pos_inside_caustics(self, lens_grid, max_distance_arcsec=0, max_iter=1000, verbose=False):
    #     """
    #     Samples a random position within the bounding box defined by the caustic clusters,
    #     extended by max_distance on all sides.
        
    #     The candidate is accepted if it is either inside any caustic polygon or if its 
    #     distance to any caustic is less than or equal to max_distance.
        
    #     Parameters:
    #         clusters (dict): Dictionary mapping cluster labels to polygons.
    #                         Each polygon is represented as an (N,2) NumPy array of vertices.
    #         max_distance (float): Maximum allowed distance from a caustic boundary.
    #         max_iter (int): Maximum number of sampling attempts.
            
    #     Returns:
    #         pos (tuple): A tuple (x, y) representing the accepted position,
    #                     or None if no valid position was found after max_iter attempts.
    #     """
    #     max_distance = units._arcsec_to_rad(max_distance_arcsec)
    #     # 1. Compute the overall bounds from all clusters.
    #     #    We concatenate all vertices from all clusters.
    #     _, _, _, _, clusters = self.compute_caustics(lens_grid, verbose=verbose)
    #     all_vertices = []
    #     for poly in clusters.values():
    #         if poly.shape[0] > 0:
    #             all_vertices.append(poly)
    #     if not all_vertices:
    #         print("No caustic vertices provided.")
    #         return None
    #     all_points = np.concatenate(all_vertices, axis=0)
    #     x_min = np.min(all_points[:, 0])
    #     x_max = np.max(all_points[:, 0])
    #     y_min = np.min(all_points[:, 1])
    #     y_max = np.max(all_points[:, 1])
        
    #     # Extend the bounds by max_distance
    #     x_bounds = (x_min - max_distance, x_max + max_distance)
    #     y_bounds = (y_min - max_distance, y_max + max_distance)
        
    #     # 2. Convert each cluster's vertices to a Shapely polygon.
    #     #    If a cluster has fewer than 3 points, use its convex hull.
    #     shapely_polys = []
    #     for label, poly_pts in clusters.items():
    #         if poly_pts.shape[0] >= 3:
    #             # Option A: Use convex hull (or change to alphashape for a concave hull)
    #             shapely_polys.append(Polygon(poly_pts))
    #             # Option B (uncomment to use alphashape for concave hull):
    #             # alpha = 1.0  # Adjust as needed
    #             # concave = alphashape.alphashape(poly_pts, alpha)
    #             # if concave.geom_type == 'MultiPolygon':
    #             #     concave = max(concave.geoms, key=lambda p: p.area)
    #             # shapely_polys.append(concave)
    #         else:
    #             # Too few points: create a MultiPoint and use its convex hull.
    #             shapely_polys.append(MultiPoint(poly_pts).convex_hull)
        
    #     # 3. Sample random positions in the computed bounding box.
    #     for _ in range(max_iter):
    #         x = np.random.uniform(x_bounds[0], x_bounds[1])
    #         y = np.random.uniform(y_bounds[0], y_bounds[1])
    #         candidate = Point(x, y)
            
    #         # Check each caustic polygon.
    #         for poly in shapely_polys:
    #             # Accept if candidate is inside the polygon...
    #             if poly.contains(candidate):
    #                 return (x, y)
    #             # ...or if its distance to the polygon is within max_distance.
    #             if candidate.distance(poly) <= max_distance:
    #                 return (x, y)
        
    #     # If no valid point is found after max_iter attempts, return None.
    #     return None
    
    def forward(self, lens_grid, z_source=None):
        "This forward computes the position of the grid in the source plane"
        source_grid = lens_grid - self.deflection_field(lens_grid, precomp_dict=self.precomp_dict)
        # save source_grid in a file
        
        return source_grid


 