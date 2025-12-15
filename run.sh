#!/bin/bash
# Script de arranque rápido do POC

echo "=================================="
echo "  POC: Análise de Posturas com IA"
echo "=================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não está instalado. Por favor instale o Docker primeiro."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose não está instalado. Por favor instale o Docker Compose primeiro."
    exit 1
fi

echo "✅ Docker e Docker Compose encontrados"
echo ""

# Check if biomechanic scores directory exists
if [ ! -d "biomechanic scores" ]; then
    echo "❌ Pasta 'biomechanic scores' não encontrada!"
    echo "   Por favor coloque os ficheiros XLSX na pasta 'biomechanic scores/'"
    exit 1
fi

# Count XLSX files
xlsx_count=$(find "biomechanic scores" -name "*.xlsx" | wc -l)
echo "📁 Ficheiros XLSX encontrados: $xlsx_count"

if [ $xlsx_count -eq 0 ]; then
    echo "⚠️  Nenhum ficheiro XLSX encontrado na pasta 'biomechanic scores/'"
    echo "   O pipeline irá falhar sem dados de entrada."
    read -p "Continuar mesmo assim? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 A iniciar pipeline completo..."
echo ""
echo "O pipeline irá:"
echo "  1. Consolidar dados dos XLSX"
echo "  2. Realizar análise exploratória (EDA)"
echo "  3. Detectar anomalias"
echo "  4. Treinar modelos preditivos"
echo "  5. Gerar relatórios e visualizações"
echo ""

# Build and run
docker-compose up --build

echo ""
echo "=================================="
echo "✅ Pipeline concluído!"
echo "=================================="
echo ""
echo "📁 Verifique os resultados em:"
echo "  - data/processed/"
echo "  - models/"
echo "  - reports/"
echo ""
