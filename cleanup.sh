#!/bin/bash
# Script de limpeza de outputs e containers

echo "=================================="
echo "  Limpeza de Outputs e Containers"
echo "=================================="
echo ""

read -p "Remover outputs gerados (data/processed, models, reports)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removendo outputs..."
    rm -rf data/processed/*
    rm -rf models/*
    rm -rf reports/eda/*
    rm -rf reports/anomalies/*
    rm -rf reports/predictions/*
    rm -rf output/*
    echo "✅ Outputs removidos"
fi

echo ""
read -p "Remover containers e imagens Docker? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removendo containers e imagens..."
    docker-compose down --rmi all --volumes
    echo "✅ Containers e imagens removidos"
fi

echo ""
echo "✅ Limpeza concluída!"
