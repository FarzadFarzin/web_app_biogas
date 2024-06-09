from flask import Flask
from app_routes import configure_routes

app = Flask(__name__)
app.secret_key = 'adadadad'  # Needed for session management

configure_routes(app)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)