#!/bin/bash

echo "🚀 Starting ArcadeDB with Docker..."
echo

# Check if Docker is running
if ! docker version >/dev/null 2>&1; then
    echo "❌ ERROR: Docker is not running or not installed"
    echo "Please start Docker and try again"
    exit 1
fi

echo "✅ Docker is running, starting ArcadeDB..."
echo

# Start ArcadeDB container
if docker-compose up -d; then
    echo
    echo "✅ ArcadeDB is starting up!"
    echo
    echo "📋 Connection Details:"
    echo "   Host: localhost"
    echo "   HTTP Port: 2480"
    echo "   Binary Port: 2424"
    echo "   Username: root"
    echo "   Password: playwithdata"
    echo
    echo "🌐 Web Console: http://localhost:2480"
    echo
    echo "⏳ Waiting for ArcadeDB to be ready..."
    
    # Wait for ArcadeDB to be ready
    for i in {1..30}; do
        if curl -s http://localhost:2480/api/v1/server >/dev/null 2>&1; then
            echo "✅ ArcadeDB is ready!"
            break
        fi
        sleep 1
        echo -n "."
    done
    
    echo
    echo "🚀 Ready to test the integration!"
    echo "Run: python test_with_docker.py"
else
    echo "❌ Failed to start ArcadeDB"
    echo "Check the Docker logs for more information:"
    echo "docker-compose logs arcadedb"
fi
