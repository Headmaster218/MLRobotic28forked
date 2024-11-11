import numpy as np
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
from scipy import stats

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)

    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())
    time_step = sim.GetTimeStep()
    max_time = 10
    kp, kd = 1000, 100

    noise_levels = [0.0, 0.01, 0.1, 1.0]  # Including no noise for baseline comparison
    results = {}

    for noise_covariance in noise_levels:
        print(f"\nTesting with noise covariance: {noise_covariance}")
        tau_mes_all, regressor_all = [], []
        current_time = 0

        while current_time < max_time:
            q_mes = sim.GetMotorAngles(0) + np.random.normal(0, noise_covariance, num_joints)
            qd_mes = sim.GetMotorVelocities(0) + np.random.normal(0, noise_covariance, num_joints)
            qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0) + np.random.normal(0, noise_covariance, num_joints)

            q_d, qd_d = ref.get_values(current_time)
            cmd = MotorCommands()
            cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
            sim.Step(cmd, "torque")

            tau_mes = sim.GetMotorTorques(0)
            if current_time > 1:
                cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
                regressor_all.append(cur_regressor)
                tau_mes_all.append(tau_mes.reshape(-1, 1))

            current_time += time_step

        if not tau_mes_all:
            print("Warning: No torque data collected.")
            continue

        tau_mes_all = np.vstack(tau_mes_all)
        regressor_all = np.vstack(regressor_all)
        a = np.linalg.pinv(regressor_all) @ tau_mes_all

        predicted_tau = regressor_all @ a
        residuals = tau_mes_all - predicted_tau
        SS_res = np.sum(residuals ** 2)
        SS_tot = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2)
        R_squared = 1 - (SS_res / SS_tot)
        n, p = tau_mes_all.shape[0], regressor_all.shape[1]
        adjusted_R_squared = 1 - (1 - R_squared) * (n - 1) / (n - p - 1)

        # Calculate F-statistic
        F_statistic = (SS_tot - SS_res) / (p - 1) / (SS_res / (n - p))

        # Calculate 95% confidence intervals for parameters
        residual_variance = SS_res / (n - p)
        confidence_intervals = stats.t.interval(0.95, n - p - 1, loc=a, scale=np.sqrt(residual_variance))

        # Print results
        print(f"Estimated parameters (a): {a.flatten()}")
        print(f"R-squared: {R_squared}")
        print(f"Adjusted R-squared: {adjusted_R_squared}")
        print(f"F-statistic: {F_statistic}")
        print(f"95% Confidence intervals for parameters:\n{confidence_intervals}")

        results[noise_covariance] = {
            'R_squared': R_squared,
            'Adjusted_R_squared': adjusted_R_squared,
            'residuals': residuals,
            'measured_tau': tau_mes_all,
            'predicted_tau': predicted_tau
        }

    # Plotting results for comparison
    for noise_covariance, result in results.items():
        measured_tau = result['measured_tau']
        predicted_tau = result['predicted_tau']
        residuals = result['residuals']
        num_time_steps = len(measured_tau) // num_joints
        residuals_matrix = residuals.reshape(num_time_steps, num_joints)
        measured_torque = measured_tau.reshape(num_time_steps, num_joints)
        predicted_torque = predicted_tau.reshape(num_time_steps, num_joints)
        time_vector = np.arange(num_time_steps)

        # Plot residuals for each joint
        fig1, axs1 = plt.subplots(num_joints, 1, figsize=(10, 20), sharex=True)
        for i in range(num_joints):
            axs1[i].plot(time_vector, residuals_matrix[:, i], label=f'Joint {i+1}')
            axs1[i].set_ylabel('Torque Error')
            axs1[i].legend()
        axs1[-1].set_xlabel('Time Steps')
        fig1.suptitle(f'Torque Prediction Error for Each Joint (Noise Covariance = {noise_covariance})')
        plt.show()

        # Plot measured vs predicted torque for each joint
        fig2, axs2 = plt.subplots(num_joints, 1, figsize=(10, 20), sharex=True)
        for i in range(num_joints):
            axs2[i].plot(time_vector, measured_torque[:, i], label='Measured Torque')
            axs2[i].plot(time_vector, predicted_torque[:, i], linestyle='--', label='Predicted Torque')
            axs2[i].set_title(f'Joint {i+1} Torque Comparison')
            axs2[i].legend()
        axs2[-1].set_xlabel('Time Steps')
        fig2.suptitle(f'Original vs Predicted Torque for Each Joint (Noise Covariance = {noise_covariance})')
        plt.show()

if __name__ == '__main__':
    main()
