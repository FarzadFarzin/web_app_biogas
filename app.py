from flask import Flask, request, render_template,session
import joblib
from datetime import datetime
import numpy as np
import joblib
import pyswarms as ps
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management


model = joblib.load('best_model.pkl')

# Load the models
A_model = joblib.load('A_model.pkl')
B_model = joblib.load('B_model.pkl')
best_model = joblib.load('best_model.pkl')

df = pd.read_csv('Final_input.csv', parse_dates=True, index_col=0)
X = df[['DS_PS_in(%)', 'VS_PS_in(%)', 'DS_WS_in(%)', 'VS_WS_in(%)', 'DS_Digester(%)', 'FA(mg/L)', 'Q_PS/Q_WS', 'Q_PS_in(m³/d)', 'Q_WS_in(m³/d)', 'Q_Total(m³/d)', 'Pre_DS_eff(%)', 'Pre_VS_eff(%)', 'ALK(mg CaCO3/L)', 'pH', 'T(°C)', 'Q_eff(m³/d)']]

def opt_func(user_values):
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


    return   pred_DS_eff, pred_VS_eff ,pos ,final_value,plot_url


@app.route('/', methods=['GET', 'POST'])
def home():
    error_mmessage = None
    if request.method == 'POST':
        
        required_fields = ['DS_PS_in', 'VS_PS_in', 'DS_WS_in', 'VS_WS_in', 'DS_Digester', 'FA', 'Q_PS_Q_WS', 'Q_PS_in', 'Q_WS_in', 'Q_Total']
        missing_fields = [field for field in required_fields if not request.form.get(field)]
        
        if missing_fields:
            error_message = f"Please fill in all required fields. Missing: {', '.join(missing_fields)}"
        else:
            if 'optimize' in request.form:
                data = request.form
                user_values = {
                    'DS_PS_in': data.get('DS_PS_in', ''),
                    'VS_PS_in': data.get('VS_PS_in', ''),
                    'DS_WS_in': data.get('DS_WS_in', ''),
                    'VS_WS_in': data.get('VS_WS_in', ''),
                    'DS_Digester': data.get('DS_Digester', ''),
                    'FA': data.get('FA', ''),
                    'Q_PS_Q_WS': data.get('Q_PS_Q_WS', ''),
                    'Q_PS_in': data.get('Q_PS_in', ''),
                    'Q_WS_in': data.get('Q_WS_in', ''),
                    'Q_Total': data.get('Q_Total', ''),
                }
                
                
                user_values_list = [float(value) for value in user_values.values()]
                pred_DS_eff, pred_VS_eff ,pos ,final_value,plot_url = opt_func(user_values_list)

                ALK_optimized = round (pos[12],2)
                pH_optimized = round (pos[13],2)
                T_optimized = round (pos[14],2)
                Q_eff_optimized = round(pos[15],2 )
                pred_DS_eff = round(float(pred_DS_eff), 2)
                pred_VS_eff = round(float(pred_VS_eff),2)
                
                # Convert NumPy arrays to lists for JSON serialization
                pred_DS_eff_list = pred_DS_eff.tolist() if isinstance(pred_DS_eff, np.ndarray) else pred_DS_eff
                pred_VS_eff_list = pred_VS_eff.tolist() if isinstance(pred_VS_eff, np.ndarray) else pred_VS_eff
                pos_list = pos.tolist() if isinstance(pos, np.ndarray) else pos
                final_value_list = final_value.tolist() if isinstance(final_value, np.ndarray) else final_value
                
                session['user_values'] = user_values_list
                session['pred_DS_eff'] = pred_DS_eff_list
                session['pred_VS_eff'] = pred_VS_eff_list
                session['pos'] = pos_list
                session['final_value'] =final_value_list
                session['plot_url']=plot_url

                
                return render_template('index.html', user_values=user_values, pred_DS_eff=pred_DS_eff_list,
                                        pred_VS_eff=pred_VS_eff_list,
                                        ALK_optimized=ALK_optimized,
                                        pH_optimized=pH_optimized,
                                        T_optimized=T_optimized,
                                        Q_eff_optimized=Q_eff_optimized,
                                        plot_url=plot_url)

            
            elif 'predict' in request.form:
                user_values = session.get('user_values', {})
                pred_DS_eff = session.get('pred_DS_eff', [0])
                pred_VS_eff = session.get('pred_VS_eff', [0])
                pos = session.get('pos', [0]*16)
                final_value_list = session.get('final_value', [0])
                plot_url = session.get('plot_url',None)

                user2_values = {
                    'ALK': float(request.form.get('ALK', 0)),
                    'pH': float(request.form.get('pH', 0)),
                    'T': float(request.form.get('T', 0)),
                    'Q_eff': float(request.form.get('Q_eff', 0))
                }

                # Flatten pred_DS_eff and pred_VS_eff if they are not already 1D
                pred_DS_eff_flat = np.ravel(pred_DS_eff)
                pred_VS_eff_flat = np.ravel(pred_VS_eff)
                print (user_values)
                # Ensure all elements are floats
                All_features = np.concatenate([list(user_values.values()), pred_DS_eff_flat, pred_VS_eff_flat, list(user2_values.values())]).astype(np.float32)
                All_features_array = np.array(All_features).reshape(1, -1)  # Reshape to 2D array with one row


                prediction = model.predict(All_features_array)
                today = datetime.now().strftime("%B %d, %Y")

                return render_template('index.html', prediction=prediction[0], today=today, user_values=user_values,
                                        pred_DS_eff=pred_DS_eff, pred_VS_eff=pred_VS_eff, pos=pos, user2_values=user2_values,plot_url=plot_url)
                
        return render_template('index.html', error_message=error_message)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
