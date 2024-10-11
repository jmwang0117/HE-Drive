import numpy as np

class DrivingStyleConsistencyMetric:
    def __init__(self, expert_positions):
        self.expert_trajectory = self._preprocess_trajectory(expert_positions)

    def _preprocess_trajectory(self, positions):
        print('postion:')
        
        positions = positions.squeeze(0)
        positions = positions.squeeze(0).cpu()
        velocities = np.gradient(positions, axis=0) / 0.5
        accelerations = np.gradient(velocities, axis=0) / 0.5
        trajectory = {
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations
        }
        return trajectory

    def _dynamic_feasibility(self, trajectory, num_points):
        max_velocity = 20.0
        max_acceleration = 10.0 / 0.5
        vel = np.linalg.norm(trajectory['velocity'][:num_points], axis=1)
        acc = np.linalg.norm(trajectory['acceleration'][:num_points], axis=1)
        vel_feasible = np.all(vel <= max_velocity)
        acc_feasible = np.all(acc <= max_acceleration)
        return int(vel_feasible and acc_feasible)

    def _comfort(self, trajectory, num_points):
        jerk = np.gradient(trajectory['acceleration'], axis=0) / 0.5
        max_jerk = 5.0 / 0.5 / 0.5
        jerk_metric = np.linalg.norm(jerk[:num_points-1], axis=1)  # Subtract one due to diff
        comfort = np.all(jerk_metric <= max_jerk)
        return int(comfort)

    def _curvature(self, trajectory, num_points):
        positions = trajectory['position'][:num_points]
        dx_dt = np.gradient(positions[:, 0]) / 0.5
        dy_dt = np.gradient(positions[:, 1]) / 0.5
        curvature = (dx_dt * np.gradient(dy_dt) - dy_dt * np.gradient(dx_dt)) / (dx_dt**2 + dy_dt**2)**1.5
        max_curvature = 0.5
        curvature_feasible = np.all(np.abs(curvature) <= max_curvature)
        return int(curvature_feasible)

    def _smoothness(self, trajectory, num_points):
        accelerations = trajectory['acceleration'][:num_points, 0]  # 0 for X (longitudinal)
        # Calculate the variance of longitudinal acceleration
        return np.var(accelerations)



    def _compute_scores(self, trajectory, segment_duration):
        num_points = int(segment_duration / 0.5)
        scores = {
            'dynamic_feasibility': self._dynamic_feasibility(trajectory, num_points),
            'comfort': self._comfort(trajectory, num_points),
            'curvature': self._curvature(trajectory, num_points),
            'smoothness': self._smoothness(trajectory, num_points),
        }
        return scores

    def score(self, predicted_positions):
        predicted_trajectory = self._preprocess_trajectory(predicted_positions)
        durations = [1,2,3]  # Duration segments for evaluation
        score_diffs = {duration: {} for duration in durations}

        for duration in durations:
            expert_metrics = self._compute_scores(self.expert_trajectory, duration)
            predicted_metrics = self._compute_scores(predicted_trajectory, duration)

            # Calculate the differences in metrics between the predicted and expert trajectories
            for metric in expert_metrics:
                difference = predicted_metrics[metric] - expert_metrics[metric]
                score_diffs[duration][metric] = np.sum(np.abs(difference)) if isinstance(difference, np.ndarray) else abs(difference)

        # Calculate overall scores for each duration by summing the absolute differences
        overall_scores = {duration: sum(score_diffs[duration].values()) for duration in durations}

        return score_diffs, overall_scores


# Example usage
# expert_positions = np.random.rand(100, 2)  # Replace with your expert trajectory positions
# predicted_positions = np.random.rand(100, 2)  # Replace with your predicted trajectory positions

# metric = DrivingStyleConsistencyMetric(expert_positions)
# score_differences, overall_scores = metric.score(predicted_positions)

# print("Score Differences by Duration: ", score_differences)
# print("Overall Scores: ", overall_scores)
