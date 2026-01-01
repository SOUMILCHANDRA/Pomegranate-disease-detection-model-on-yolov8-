@echo off
echo Starting Pomegranate Disease API Server...
echo.
echo [INFO] Ensure PostgreSQL is running and you have created the 'pomegranate_db' database.
echo [INFO] Your App should connect to the IP address below:
ipconfig | findstr "IPv4"
echo.
py e:/SIH2/api_server.py
pause
