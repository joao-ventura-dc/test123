#!/bin/bash
# Script para iniciar Jupyter Notebook

echo "=================================="
echo "  Jupyter Notebook - POC Posturas"
echo "=================================="
echo ""

echo "🚀 A iniciar Jupyter Notebook..."
echo ""
echo "Aceda via browser:"
echo "  http://localhost:8888"
echo ""
echo "Pressione Ctrl+C para parar"
echo ""

docker-compose --profile jupyter up jupyter
