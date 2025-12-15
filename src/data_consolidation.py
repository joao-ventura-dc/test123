"""
Script de Consolidação de Dados - POC Análise de Posturas
Consolida todos os ficheiros XLSX numa única base de dados.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_excel_file(file_path):
    """Carrega um ficheiro Excel e adiciona metadados."""
    try:
        df = pd.read_excel(file_path)

        # Extrair informações do nome do ficheiro
        filename = Path(file_path).stem
        parts = filename.split('_')

        # Adicionar colunas de metadados
        df['source_file'] = filename
        df['camera_id'] = parts[1] if len(parts) > 1 else 'unknown'
        df['recording_date'] = parts[0] if len(parts) > 0 else 'unknown'

        return df
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {str(e)}")
        return None


def consolidate_data(data_dir='biomechanic scores', output_dir='data/processed'):
    """Consolida todos os ficheiros XLSX numa única base de dados."""
    print("=" * 80)
    print("CONSOLIDAÇÃO DE DADOS - POC ANÁLISE DE POSTURAS")
    print("=" * 80)

    # Encontrar todos os ficheiros XLSX
    xlsx_files = glob.glob(f"{data_dir}/*.xlsx")
    print(f"\n📁 Ficheiros encontrados: {len(xlsx_files)}")

    # Carregar todos os ficheiros
    all_dataframes = []
    for i, file_path in enumerate(xlsx_files, 1):
        print(f"  [{i}/{len(xlsx_files)}] Carregando: {Path(file_path).name}")
        df = load_excel_file(file_path)
        if df is not None:
            all_dataframes.append(df)

    # Consolidar todos os dataframes
    print("\n🔗 Consolidando dados...")
    consolidated_df = pd.concat(all_dataframes, ignore_index=True)

    # Estatísticas básicas
    print(f"\n📊 ESTATÍSTICAS DA CONSOLIDAÇÃO:")
    print(f"  - Total de registos: {len(consolidated_df):,}")
    print(f"  - Total de colunas: {len(consolidated_df.columns)}")
    print(f"  - Ficheiros processados: {len(all_dataframes)}")
    print(f"  - Período: {consolidated_df['recording_date'].min()} a {consolidated_df['recording_date'].max()}")

    # Informação sobre as colunas
    print(f"\n📋 COLUNAS DISPONÍVEIS:")
    for col in consolidated_df.columns:
        non_null = consolidated_df[col].count()
        null_pct = (1 - non_null / len(consolidated_df)) * 100
        print(f"  - {col}: {non_null:,} valores ({null_pct:.1f}% missing)")

    # Salvar dataset consolidado
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / 'consolidated_data.csv'
    print(f"\n💾 Salvando dataset consolidado em: {csv_path}")
    consolidated_df.to_csv(csv_path, index=False)

    # Salvar também em formato comprimido
    csv_gz_path = output_path / 'consolidated_data.csv.gz'
    consolidated_df.to_csv(csv_gz_path, index=False, compression='gzip')
    print(f"💾 Salvando versão comprimida em: {csv_gz_path}")

    # Salvar metadados
    metadata = {
        'total_records': len(consolidated_df),
        'total_columns': len(consolidated_df.columns),
        'files_processed': len(all_dataframes),
        'date_range': f"{consolidated_df['recording_date'].min()} to {consolidated_df['recording_date'].max()}",
        'columns': list(consolidated_df.columns),
        'processing_date': datetime.now().isoformat()
    }

    metadata_path = output_path / 'metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Metadados salvos em: {metadata_path}")

    print("\n" + "=" * 80)
    print("✅ CONSOLIDAÇÃO COMPLETA!")
    print("=" * 80)

    return consolidated_df


if __name__ == "__main__":
    df = consolidate_data()
    print(f"\n🎯 Dataset consolidado: {len(df)} registos x {len(df.columns)} colunas")
