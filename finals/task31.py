import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib

# Set the model type: "neural_network" or "random_forest"
neural_network_or_random_forest = "neural_network"  # Change to "random_forest" to use Random Forest models

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(128, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)

def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Extract time array and normalize if needed
    time_array = np.array(data['time'])

    # Load all models in a list
    models = []
    if neural_network_or_random_forest == "neural_network":
        for joint_idx in range(7):
            model = MLP()
            model_filename = os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            models.append(model)
    elif neural_network_or_random_forest == "random_forest":
        for joint_idx in range(7):
            model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib')
            model = joblib.load(model_filename)
            models.append(model)

    # Generate a test goal position
    goal_position = [0.7, 0.0, 0.12]
    test_time_array = np.arange(time_array.min(), time_array.max(), 0.02)  # Simulated time steps
    test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))
    test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))

    # Predict joint positions over time for the goal
    predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))
    for joint_idx in range(7):
        if neural_network_or_random_forest == "neural_network":
            test_input_tensor = torch.from_numpy(test_input).float()
            with torch.no_grad():
                predictions = models[joint_idx](test_input_tensor).numpy().flatten()
        elif neural_network_or_random_forest == "random_forest":
            predictions = models[joint_idx].predict(test_input)
        predicted_joint_positions_over_time[:, joint_idx] = predictions

    # Plot each joint's predicted trajectory to assess smoothness
    plt.figure(figsize=(12, 8))
    for i in range(7):
        plt.subplot(3, 3, i + 1)
        plt.plot(predicted_joint_positions_over_time[:, i], label=f"Joint {i+1}")
        plt.title(f"Joint {i+1} Predicted Trajectory")
        plt.xlabel("Time Steps")
        plt.ylabel("Joint Position")
        plt.legend()

    plt.suptitle(f"Predicted Joint Trajectories - Model: {neural_network_or_random_forest}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    main()
