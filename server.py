from flask import Flask, render_template, request
from app import opt_func
from waitress import serve
from datetime import datetime

app = Flask(__name__)

@app.route('/')
@app.route('/index')

def index():
    return render_template ("index.html")


@app.route('/welcome')
def welcome():
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
    
    
    return render_template("welcome.html",date = datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


@app.route('/optimization')

def optimize():
    
    x = opt_func(user_values)
    
    
    return render_template('optimization.html')

if __name__ == '__main__':
    serve(app, host= "0.0.0.0", port=8000)

