import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections import deque

from metrics import DrivingStyleConsistencyMetric

import torch

plt.rcParams['axes.titlesize'] = 16   
plt.rcParams['axes.labelsize'] = 14  
plt.rcParams['xtick.labelsize'] = 12  
plt.rcParams['ytick.labelsize'] = 12  
plt.rcParams['legend.fontsize'] = 12 

class TrajectoryScoring:

    def __init__(self, gt_dict, pred_dict, save_vis_path):
        self.gt_dict = gt_dict
        self.pred_dict = pred_dict
        self.save_vis_path = save_vis_path
        self.driving_style = 'aggressive' # aggressive or conservative
        self.driving_scene = 'city' # city or highway
        self.speed_history = deque(maxlen=12)  # Store the last average speeds
        self.b_spline_path = os.path.join(self.save_vis_path, 'b_spline')
        self.speed_range = (None, None) 
        self.sample = {}
        self.speed_lower_bound = -0.1
        self.speed_upper_bound = 40.0
        self.default_city_road_speed_limit = 15.67
        self.default_highway_speed_limit = 29.06
        self.best_trajectory_queue = deque(maxlen=3)  # 保存最佳轨迹的队列，最多3条
        
    def ensure_directory_exists(self, path):
        """Ensure the directory exists, and if not, create it."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory created: {path}")
        else:
            pass

    def transform_data(self):
        agent_bboxes = self.pred_dict["agent_bboxes"][0] #N x 5, x, y, w, h, yaw

        #print(agent_bboxes.shape)
        agent_bboxes = agent_bboxes[:, [0, 1, 3, 5, 6]]

        
        #print(agent_bboxes.shape)

        #agent_traj = self.pred_dict["agent_traj"]  # N x T x 2
        #print(agent_traj.shape)

        #map_pts = self.pred_dict["map_pts"]  # N x 20 x 2
        #map_labels = self.pred_dict["map_labels"]  # N


        pred_ego_traj = self.pred_dict["ego_traj"]  # T x 2
        gt_ego_traj = self.gt_dict["ego_traj"]
        target_point = self.gt_dict["target_point"]

        agent_num = agent_bboxes.shape[0]
        agent_bbox = agent_bboxes[:, :4]
        agent_yaw = agent_bboxes[:, 4]


        #agent_trajreg = agent_traj[..., [1, 0]]
        #agent_trajreg[..., 0] = -agent_trajreg[..., 0]
       
        # if agent_num == 0:
        #     agent_trajreg = np.zeros((0, 60))
        # else:
        #     agent_trajreg = agent_trajreg.reshape(agent_num, -1)
        agent_mask = np.ones((agent_num, 6))

        agent_bbox_vcs = agent_bbox
        agent_bbox_vcs = agent_bbox_vcs[..., [1, 0, 2, 3]]
        agent_bbox_vcs[..., 0] = -agent_bbox_vcs[..., 0]

        vcs_yaw = agent_yaw[:, None]
        #print(agent_trajreg.shape)
        #print(agent_bbox_vcs[:, :2].shape)
        #print(vcs_yaw.shape)

        self.sample["gt_attr_labels"] = np.concatenate([agent_mask, agent_bbox_vcs[:, :2], vcs_yaw, agent_bbox_vcs[:, 2:], ], axis=1)
        self.sample["gt_labels_3d"] = np.ones(agent_num)

        # odo 处理
        pred_vcs_odo_info = pred_ego_traj
        pred_vcs_odo_info = pred_vcs_odo_info[..., [1, 0]]
        pred_vcs_odo_info[..., 0] = -pred_vcs_odo_info[..., 0]
        self.sample["pred_ego_fut_trajs"] = pred_vcs_odo_info

        # new_anchors = self.pred_dict["anchors"]
        # new_anchors = new_anchors[..., [1, 0]]
        # new_anchors[..., 0] = -new_anchors[..., 0]
        # self.sample["anchors"] = new_anchors.cpu().numpy()

        # self.sample["alpha"] = (alpha - alpha.min()) / alpha.max()

        #print(gt_ego_traj.shape)

        gt_vcs_odo_info = gt_ego_traj
        gt_vcs_odo_info = gt_vcs_odo_info[..., [1, 0]]
        gt_vcs_odo_info[..., 0] = -gt_vcs_odo_info[..., 0]
        self.sample["gt_ego_fut_trajs"] = gt_vcs_odo_info

        # map 计算
        #lines = map_pts
        #labels = map_labels

        # self.sample["map_gt_bboxes_3d"] = lines
        # self.sample["map_gt_labels_3d"] = labels

        target_point = target_point[:, [1, 0]]
        target_point[:, 0] = -target_point[:, 0]
        self.sample["target_info"] = target_point[0]

        # self.sample["expert_idxs"] = self.pred_dict["expert_idxs"]
        # self.sample["post_process_idxs"] = self.pred_dict["post_process_idxs"]

    
    def calculate_angle_deviation(self, trajectory, current_position, target_position):
        target_vector = np.array(target_position) - np.array(current_position)
        angle_deviation_scores = []
        for point in trajectory:
            point_vector = np.array(point) - np.array(current_position)
            dot_product = np.dot(target_vector, point_vector)
            norm_product = np.linalg.norm(target_vector) * np.linalg.norm(point_vector)
            angle_cosine = dot_product / norm_product
            angle_deviation = np.arccos(angle_cosine)  # Value in radians
            angle_deviation_scores.append(angle_deviation)
        return np.mean(angle_deviation_scores)
    
    def calculate_distance_to_target(self, trajectory, target_point):
        return np.linalg.norm(trajectory[-1].cpu() - target_point)

    def check_collision(self, trajectory, other_vehicles_bboxes):
        """
        Check if the trajectory collides with any bounding boxes of other vehicles.
        :param trajectory: Array of shape (T, 2), where T is the number of timesteps.
        :param other_vehicles_bboxes: List of bounding boxes, each defined as [x_center, y_center, width, height, yaw].
        :return: True if collision detected, False otherwise.
        """
        for bbox in other_vehicles_bboxes:
            x_center, y_center, width, height, yaw = bbox
            # Create bounds for the bounding box
            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2

            # Check each point in the trajectory
            for point in trajectory:
                x, y = point
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True  # Collision detected
        return False  # No collision detected
    
    def calculate_collisions_with_agents(self):
        trajectories = self.sample['pred_ego_fut_trajs']
        agent_boxes = self.sample['gt_attr_labels']
        collision_scores = []
        for traj in trajectories:
            collision = 0
            for agent in agent_boxes:
                # Convert agent box format to [x_center, y_center, width, height, yaw]
                bbox = [agent[0], agent[1], agent[3], agent[4], agent[2]]
                if self.check_collision(traj, [bbox]):  # Check collision with each agent
                    collision = 1
                    break
            collision_scores.append(collision)
        return collision_scores


    def calculate_average_speeds(self, trajectories):
        """
        Calculate average speeds for each trajectory.
        :param trajectories: Array of shape (num_trajectories, num_points, 2)
        :return: Array of average speeds of shape (num_trajectories,)
        """
        speeds = self.calculate_speeds(trajectories.cpu())
        average_speeds = np.mean(speeds, axis=1)
        return average_speeds

    def calculate_speeds(self, trajectories):
        """
        Calculate speeds for each point in each trajectory.
        :param trajectories: Array of shape (num_trajectories, num_points, 2)
        :return: Array of speeds of shape (num_trajectories, num_points-1)
        """
        speeds = np.linalg.norm(np.diff(trajectories, axis=1), axis=2)
        return speeds

    def calculate_dynamics(self, trajectory):
        """
        Calculate longitudinal and lateral velocities, accelerations, and jerks for a trajectory.
        """
        dt = 0.1  # assuming trajectory points are at 0.1 second intervals
        velocities = np.diff(trajectory, axis=0) / dt
        long_velocities = velocities[:, 0]  # Assuming x is the longitudinal direction
        lat_velocities = velocities[:, 1]  # Assuming y is the lateral direction

        long_accelerations = np.diff(long_velocities) / dt
        lat_accelerations = np.diff(lat_velocities) / dt

        long_jerks = np.diff(long_accelerations) / dt
        lat_jerks = np.diff(lat_accelerations) / dt

        return long_velocities, lat_velocities, long_accelerations, lat_accelerations, long_jerks, lat_jerks
    
    def lat_comfort_cost(self, long_velocities, lat_velocities, long_accelerations, lat_jerks):
        max_cost = 0.0
        for i in range(len(long_velocities)):
            s_dot = long_velocities[i]
            s_dotdot = long_accelerations[i] if i < len(long_accelerations) else 0.0
            l_prime = lat_velocities[i]
            l_primeprime = lat_jerks[i] if i < len(lat_jerks) else 0.0
            cost = l_primeprime * s_dot * s_dot + l_prime * s_dotdot
            max_cost = max(max_cost, abs(cost))
        return max_cost

    def lon_comfort_cost(self, long_jerks):
        cost_sqr_sum = 0.0
        cost_abs_sum = 0.0
        longitudinal_jerk_upper_bound = 5.0  
        numerical_epsilon = 1e-6  
        for jerk in long_jerks:
            cost = jerk / longitudinal_jerk_upper_bound
            cost_sqr_sum += cost * cost
            cost_abs_sum += abs(cost)
        return cost_sqr_sum / (cost_abs_sum + numerical_epsilon)

    def calculate_speed_cost(self, average_speeds, speed_range):
        # 根据场景设置速度限制
        if self.driving_scene == 'city':
            speed_limit = self.default_city_road_speed_limit
        elif self.driving_scene == 'highway':
            speed_limit = self.default_highway_speed_limit
        else:
            raise ValueError("Invalid driving scene type. Must be 'city' or 'highway'.")

        speed_costs = []
        for speed in average_speeds:
            # 如果速度超过限制，设置无穷大的成本
            if speed > speed_limit:
                speed_cost = float('inf')
            else:
                # 在速度范围内计算成本
                if self.driving_style == 'aggressive':
                    # 激进模式下，速度慢的轨迹施加成本
                    if speed < speed_range[0]:
                        speed_cost = (speed_range[0] - speed) / speed_range[0]
                    else:
                        speed_cost = 0
                elif self.driving_style == 'conservative':
                    # 保守模式下，速度快的轨迹施加成本
                    if speed > speed_range[1]:
                        speed_cost = (speed - speed_range[1]) / speed_range[1]
                    else:
                        speed_cost = 0
            speed_costs.append(speed_cost)

        return speed_costs


    def calculate_centripetal_acceleration_cost(self, trajectory, speeds):
        """
        Calculate the centripetal acceleration cost for a single trajectory.
        :param trajectory: Array of shape (num_points, 2) containing a single trajectory.
        :param speeds: Array of speeds of shape (num_points-1,) for the trajectory.
        :return: Centripetal acceleration cost for the trajectory.
        """
        centripetal_acc_sum = 0.0
        centripetal_acc_sqr_sum = 0.0
        numerical_epsilon = 1e-6  # 避免除以零

        # 计算曲率
        curvatures = [self.calculate_curvature(trajectory[i - 1], trajectory[i], trajectory[i + 1])
                      for i in range(1, len(trajectory) - 1)]

        # 计算向心加速度和成本
        for i, curvature in enumerate(curvatures):
            centripetal_acc = speeds[i] ** 2 * curvature
            centripetal_acc_sum += np.fabs(centripetal_acc)
            centripetal_acc_sqr_sum += centripetal_acc ** 2

        cost = centripetal_acc_sqr_sum / (centripetal_acc_sum + numerical_epsilon)
        return cost

    def calculate_curvature(self, p1, p2, p3):
        """
        Calculate the curvature given three consecutive points on the trajectory.
        :param p1, p2, p3: Consecutive points on the trajectory.
        :return: Curvature.
        """
        # 使用三点计算曲率的方法
        k = np.linalg.norm(np.cross(p2 - p1, p3 - p2)) / np.linalg.norm(p2 - p1)**2
        return k

    def compute_scores(self):
        weights = {
            'target_distance': 1.5,
            'collision': 5.0,
            'speed': 2.5,  
            'lat_comfort': 1.5,
            'lon_comfort': 4.5,
            'centripetal_acceleration': 3.0,  
            'angle_deviation': 3.5,  # 新增的角度偏离权重
        }

       
        pred_trajectories = self.sample['pred_ego_fut_trajs']
        #target_point = self.sample['target_info'] 
        
        # 如果队列中有上一帧的最佳轨迹，将它们添加到当前帧的轨迹列表中
        if self.best_trajectory_queue:
            previous_best_trajectories = np.array(self.best_trajectory_queue)
            pred_trajectories = np.concatenate([previous_best_trajectories, pred_trajectories], axis=0)
        
        # Calculate average speeds for each trajectory
        average_speeds = self.calculate_average_speeds(pred_trajectories)

        # Update the speed range based on historical average speeds
        if self.speed_history:
            historical_avg_speed = np.mean(self.speed_history)
            self.speed_range = (0.8 * historical_avg_speed, 1.2 * historical_avg_speed)
        else:
            # If there's no history yet, use the current average speeds
            current_avg_speed = np.mean(average_speeds)
            self.speed_range = (0.8 * current_avg_speed, 1.2 * current_avg_speed)

        scores = []
        for i, pred_traj in enumerate(pred_trajectories):
            target_distance = self.calculate_distance_to_target(pred_traj.cpu(), target_point)
            # 计算当前轨迹的角度偏离度
            angle_deviation_cost = self.calculate_angle_deviation(pred_traj.cpu(), pred_traj[0].cpu(), target_point)
            collision = self.calculate_collisions_with_agents()[i]

            # Calculate speed cost for the current trajectory
            speed_cost = self.calculate_speed_cost([average_speeds[i]], self.speed_range)[0]

            # Calculate dynamics for the trajectory
            long_velocities, lat_velocities, long_accelerations, lat_accelerations, long_jerks, lat_jerks = self.calculate_dynamics(pred_traj.cpu())

            # Calculate comfort costs
            lat_comfort = self.lat_comfort_cost(long_velocities, lat_velocities, long_accelerations, lat_jerks)
            lon_comfort = self.lon_comfort_cost(long_jerks)

            # Calculate centripetal acceleration cost for the current trajectory
            speeds = self.calculate_speeds(np.expand_dims(pred_traj.cpu(), axis=0))[0]
            centripetal_acceleration_cost = self.calculate_centripetal_acceleration_cost(pred_traj.cpu(), speeds)

            # Calculate total score
            total_score = ( weights['target_distance'] * target_distance +
                            weights['collision'] * collision +
                            weights['speed'] * speed_cost + 
                            weights['lat_comfort'] * lat_comfort + 
                            weights['lon_comfort'] * lon_comfort + 
                            weights['centripetal_acceleration'] * centripetal_acceleration_cost +
                            weights['angle_deviation'] * angle_deviation_cost
                           )
            
            scores.append(total_score)

        # Update the historical average speed
        self.speed_history.append(np.mean(average_speeds))
        # 在得分计算结束后，找到成本最小的轨迹并更新队列
        min_score_idx = np.argmin(scores[len(self.best_trajectory_queue):])  # 忽略队列中的轨迹
        self.best_trajectory_queue.append(pred_trajectories[min_score_idx])

        return scores
  
    def run(self, timestamp):
        self.ensure_directory_exists(self.b_spline_path)
        self.transform_data()
        score = self.compute_scores()
        metric = DrivingStyleConsistencyMetric(self.sample["gt_ego_fut_trajs"].cpu())
        min_score_idx = np.argmin(score)
        score_differences, overall_scores = metric.score(self.sample["pred_ego_fut_trajs"][min_score_idx])
        print("Score Differences by Duration: ", score_differences)
        print("Overall Scores: ", overall_scores)
        
        return min_score_idx

    
   


