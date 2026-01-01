from flask import Flask, jsonify
from postgres_connector import PostgresConnector

app = Flask(__name__)

# Initialize Connector
# Standard default connection. User update if needed.
connector = PostgresConnector()

@app.route('/')
def home():
    return "<h1>Pomegranate Disease Management API</h1><p>Status: Running</p>"

@app.route('/reports', methods=['GET'])
def get_reports():
    """
    Returns the latest 50 scan reports.
    """
    try:
        data = connector.get_latest_reports(limit=50)
        return jsonify({"status": "success", "count": len(data), "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/latest', methods=['GET'])
def get_latest():
    """
    Returns only the single most recent report.
    """
    try:
        data = connector.get_latest_reports(limit=1)
        if data:
            return jsonify({"status": "success", "data": data[0]})
        else:
            return jsonify({"status": "empty", "data": None})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Host 0.0.0.0 allows access from other devices on the network (e.g. Mobile App)
    print("Starting API Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
