import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    kp = 1000
    kd = 100

    # Noise covariance values to test
    noise_covariances = [0.01, 0.1, 1.0]
    
    for noise_cov in noise_covariances:
        print(f"Testing with noise covariance: {noise_cov}")

        # Initialize data storage for each noise level
        tau_mes_all = []
        regressor_all = []
        
        # Data collection loop
        while current_time < max_time:
            # Initialize q_mes, qd_mes, and qdd_mes to None
            q_mes = None
            qd_mes = None
            qdd_mes = None
    
            # Measure current state and add noise
            q_mes = sim.GetMotorAngles(0)# + np.random.normal(0, noise_cov, size=sim.GetMotorAngles(0).shape)
            qd_mes = sim.GetMotorVelocities(0)# + np.random.normal(0, noise_cov, size=sim.GetMotorVelocities(0).shape)
            qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)# + np.random.normal(0, noise_cov, size=sim.ComputeMotorAccelerationTMinusOne(0).shape)
    
            # Compute sinusoidal reference trajectory
            q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity

            # Control command
            cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
            sim.Step(cmd, "torque")

            # Get measured torque
            tau_mes = sim.GetMotorTorques(0)

            if dyn_model.visualizer: 
                for index in range(len(sim.bot)):  # Conditionally display the robot model
                    q = sim.GetMotorAngles(index)
                    dyn_model.DisplayModel(q)  # Update the display of the robot model

            # Exit logic with 'q' key
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')
            if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                break

            # Compute regressor and store it
            cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)

            current_time += time_step
            # Collect measured torque and regressor data
            tau_mes_all.append(tau_mes)
            regressor_all.append(cur_regressor)


        # Convert lists to NumPy arrays for easier manipulation
        tau_mes_all = np.array(tau_mes_all)
        regressor_all = np.array(regressor_all)

        # Linear regression to estimate parameters
        regressor_matrix = np.vstack(regressor_all)  # Shape (n_samples, n_features)
        tau_mes_vector = tau_mes_all.flatten()  # Shape (n_samples,)

        # Perform linear regression
        reg_model = LinearRegression(fit_intercept=False)  # Assuming regressor doesn't include a bias term
        reg_model.fit(regressor_matrix, tau_mes_vector)
        
        # Get estimated parameters
        a_estimated = reg_model.coef_

        # Compute metrics for model validation
        y_pred = reg_model.predict(regressor_matrix)

        # Adjusted R-squared
        r_squared = reg_model.score(regressor_matrix, tau_mes_vector)
        n = len(tau_mes_vector)  # number of observations
        p = regressor_matrix.shape[1]  # number of predictors
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

        # F-statistics
        ss_total = np.sum((tau_mes_vector - np.mean(tau_mes_vector)) ** 2)
        ss_residual = np.sum((tau_mes_vector - y_pred) ** 2)
        f_statistic = (ss_total - ss_residual) / (ss_residual / (n - p - 1))

        # Confidence intervals for parameters
        X_with_const = sm.add_constant(regressor_matrix)  # Add constant for intercept
        model = sm.OLS(tau_mes_vector, X_with_const).fit()
        conf_int = model.conf_int()

        # Print results
        # print(f"Estimated parameters: {a_estimated}")
        print(f"Adjusted R-squared: {adjusted_r_squared}")
        print(f"F-statistic: {f_statistic}\n")
        # print(f"Confidence intervals for parameters:\n{conf_int}")

        # Optional: plot the torque prediction error for each joint
        error = tau_mes_vector - y_pred
        plt.figure(figsize=(10, 6))
        plt.plot(error)
        plt.title(f"Torque Prediction Error for Each Joint (Noise Covariance: {noise_cov})")
        plt.xlabel("Time Steps")
        plt.ylabel("Prediction Error (Torque)")
        plt.grid()
        plt.show()

        # Reset current time for next iteration
        current_time = 0

if __name__ == '__main__':
    main()
