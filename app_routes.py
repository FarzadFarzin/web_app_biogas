from flask import Flask, request, render_template, session
from datetime import datetime
import pandas as pd
from model_func import opt_func, load_model
import numpy as np




def configure_routes(app):

    @app.route('/', methods=['GET', 'POST'])
    def home():
        if request.method == 'POST':
            user_values, error_message = validate_and_extract_form_data(request.form)
            if error_message:
                return render_template('index.html', error_message=error_message)
            
            if 'optimize' in request.form:
                return handle_optimization(user_values)
            elif 'predict' in request.form:
                return handle_prediction(user_values)
        
        return render_template('index.html')

    def validate_and_extract_form_data(form):
        required_fields = ['DS_PS_in', 'VS_PS_in', 'DS_WS_in', 'VS_WS_in', 'DS_Digester', 'FA', 'Q_PS_Q_WS', 'Q_PS_in', 'Q_WS_in', 'Q_Total']
        missing_fields = [field for field in required_fields if not form.get(field)]
        
        if missing_fields:
            return None, f"Please fill in all required fields. Missing: {', '.join(missing_fields)}"
        
        user_values = {field: form.get(field, '') for field in required_fields}
        return user_values, None

    def handle_optimization(user_values):
        global pred_DS_eff
        global pred_VS_eff
        global plot_url
        global final_value
        user_values_list = [float(value) for value in user_values.values()]
        pred_DS_eff, pred_VS_eff, pos, final_value, plot_url ,warning_message= opt_func(user_values_list)
        
        # Store predictions in session
        # session['pred_DS_eff'] = pred_DS_eff
        # session['pred_VS_eff'] = pred_VS_eff
        # session['plot_url'] = plot_url

        optimized_values = {
            'ALK_optimized': round(pos[12], 2),
            'pH_optimized': round(pos[13], 2),
            'T_optimized': round(pos[14], 2),
            'Q_eff_optimized': round(pos[15], 2)
        }
        return render_template('index.html', user_values=user_values, pred_DS_eff=pred_DS_eff,
                               pred_VS_eff=pred_VS_eff, plot_url=plot_url,warning_message=warning_message,  **optimized_values)

    def handle_prediction(user_values):
        # pred_DS_eff, pred_VS_eff, plot_url = get_latest_predictions()

        user2_values = {
            'ALK': float(request.form.get('ALK', 0)),
            'pH': float(request.form.get('pH', 0)),
            'T': float(request.form.get('T', 0)),
            'Q_eff': float(request.form.get('Q_eff', 0))
        }
        All_features = np.concatenate([list(user_values.values()), [pred_DS_eff], [pred_VS_eff], list(user2_values.values())]).astype(np.float32)
        
        All_features_array = np.array(All_features).reshape(1, -1)
        _, _, _, _, _, model,_ = load_model()
        prediction = model.predict(All_features_array)
        today = datetime.now().strftime("%B %d, %Y")
        return render_template('index.html', prediction=prediction[0], today=today, user_values=user_values,pred_DS_eff=pred_DS_eff, pred_VS_eff=pred_VS_eff, user2_values=user2_values, plot_url=plot_url,final_value=final_value)

    # def get_latest_predictions():
    #     # Fetch predictions from session
    #     pred_DS_eff = session.get('pred_DS_eff')
    #     pred_VS_eff = session.get('pred_VS_eff')
    #     plot_url = session.get('plot_url')
    #     return pred_DS_eff, pred_VS_eff, plot_url

