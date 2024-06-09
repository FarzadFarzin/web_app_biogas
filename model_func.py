
import joblib
import numpy as np
import pyswarms as ps
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

def load_model():
    model = joblib.load('Saved_files/best_model.pkl')
    df = pd.read_csv('Saved_files/Final_input.csv', parse_dates=True, index_col=0)
    X = df[['DS_PS_in(%)', 'VS_PS_in(%)', 'DS_WS_in(%)', 'VS_WS_in(%)', 'DS_Digester(%)', 'FA(mg/L)', 'Q_PS/Q_WS', 'Q_PS_in(m³/d)', 'Q_WS_in(m³/d)', 'Q_Total(m³/d)', 'Pre_DS_eff(%)', 'Pre_VS_eff(%)', 'ALK(mg CaCO3/L)', 'pH', 'T(°C)', 'Q_eff(m³/d)']]
    A_model = joblib.load('Saved_files/A_model.pkl')
    B_model = joblib.load('Saved_files/B_model.pkl')
    best_model = joblib.load('Saved_files/best_model.pkl')
    return A_model, B_model, best_model,X, df ,model




def opt_func(user_values):
    A_model, B_model, best_model, X, df ,model = load_model()
    # Prepare User_Value as a 2D array
    user_value_array = np.array(list(user_values)).reshape(1, -1)
    # Predict with A_model
    pred_DS_eff = A_model.predict(user_value_array).reshape(1, -1)
    # Concatenate User_Value and Pred_DS_eff for B_model input
    b_input = np.concatenate((user_value_array, pred_DS_eff), axis=1)
    # Predict with B_model
    pred_VS_eff = B_model.predict(b_input).reshape(1, -1)
    # Concatenate all inputs and predictions for final output
    c_input = np.concatenate((b_input, pred_VS_eff), axis=1)
    # Extract the first 10 values from C_input as constants
    constant_values = {i: c_input[0, i] for i in range(len(c_input[0, :]))}
    # Define the objective function to maximize the output of the model
    def objective_function(features):
        # Replace certain indices with constant values
        for index, value in constant_values.items():
            features[:, index] = value
        prediction = best_model.predict(features)
        return -prediction

    # Prepare bounds, setting constant parameters to their respective values
    lower_bounds = X.min().values
    upper_bounds = X.max().values

    for index, value in constant_values.items():
        lower_bounds[index] = value
        upper_bounds[index] = value
    bounds = (lower_bounds, upper_bounds)
    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=16, options=options, bounds=bounds)
    # Perform optimization
    cost, pos = optimizer.optimize(objective_function, iters=500)
    final_value = -cost

    # Plot the cost over iterations
    plt.figure(figsize=(6, 6))
    plt.plot(np.abs(optimizer.cost_history))
    plt.xlabel('Iterations')
    plt.ylabel('Biogas yield')
    plt.title('Optimization Cost Over Iterations')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)  # Rewind the buffer
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')  # Encode as base64 for HTML embedding
    pred_DS_eff = float(pred_DS_eff)
    pred_VS_eff = float(pred_VS_eff)
    return   pred_DS_eff, pred_VS_eff ,pos ,final_value,plot_url
