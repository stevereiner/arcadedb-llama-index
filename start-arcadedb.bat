@echo off
chcp 65001 >nul
echo Starting ArcadeDB with Docker...
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo Docker is running, starting ArcadeDB...
echo.

REM Start ArcadeDB container
docker-compose up -d

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] ArcadeDB is starting up!
    echo.
    echo [INFO] Connection Details:
    echo    Host: localhost
    echo    HTTP Port: 2480
    echo    Binary Port: 2424
    echo    Username: root
    echo    Password: playwithdata
    echo.
    echo [WEB] Web Console: http://localhost:2480
    echo.
    echo [WAIT] Waiting for ArcadeDB to be ready...
    timeout /t 10 /nobreak >nul
    
    REM Test connection
    curl -s http://localhost:2480/api/v1/server >nul 2>&1
    if %errorlevel% equ 0 (
        echo [READY] ArcadeDB is ready!
    ) else (
        echo [WAIT] ArcadeDB is still starting up, please wait a moment...
    )
    
    echo.
    echo [READY] Ready to test the integration!
    echo Run: python test_with_docker.py
) else (
    echo [ERROR] Failed to start ArcadeDB
    echo Check the Docker logs with: docker-compose logs arcadedb
)

pause
