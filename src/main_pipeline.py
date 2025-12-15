"""
Pipeline Principal - POC Análise de Posturas
Executa todo o pipeline de análise de ponta a ponta.
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports dos módulos
from data_consolidation import consolidate_data
from exploratory_analysis import PostureEDA
from anomaly_detection import AnomalyDetector
from predictive_model import PredictiveModel


def print_header(title):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_full_pipeline():
    """Executa pipeline completo de análise."""
    start_time = datetime.now()

    print_header("🚀 POC: ANÁLISE E PREVISÃO DE POSTURAS DE TRABALHO COM IA")
    print(f"⏰ Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # FASE 1: Consolidação de Dados
        print_header("FASE 1: CONSOLIDAÇÃO DE DADOS")
        df_consolidated = consolidate_data()
        print(f"\n✅ Fase 1 completa: {len(df_consolidated):,} registos consolidados")

        # FASE 2: Análise Exploratória (EDA)
        print_header("FASE 2: ANÁLISE EXPLORATÓRIA (EDA)")
        eda = PostureEDA()
        df_with_features = eda.run_full_analysis()
        print(f"\n✅ Fase 2 completa: {len(df_with_features.columns)} features criadas")

        # FASE 3: Deteção de Anomalias
        print_header("FASE 3: DETEÇÃO DE ANOMALIAS")
        detector = AnomalyDetector()
        df_with_anomalies = detector.run_full_detection()
        n_anomalies = df_with_anomalies['ensemble_anomaly'].sum() if 'ensemble_anomaly' in df_with_anomalies.columns else 0
        print(f"\n✅ Fase 3 completa: {n_anomalies:,} anomalias detetadas")

        # FASE 4: Modelos Preditivos
        print_header("FASE 4: MODELOS PREDITIVOS")
        predictor = PredictiveModel()
        predictor.run_full_pipeline()
        print(f"\n✅ Fase 4 completa: Modelos treinados e salvos")

        # Resumo Final
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print_header("📊 RESUMO FINAL")
        print(f"✅ Pipeline executado com sucesso!\n")
        print(f"📈 Estatísticas:")
        print(f"  - Registos processados: {len(df_with_anomalies):,}")
        print(f"  - Features criadas: {len(df_with_features.columns)}")
        print(f"  - Anomalias detetadas: {n_anomalies:,}")
        print(f"  - Modelos treinados: {len(list(Path('models').glob('*.joblib')))}")
        print(f"\n⏱️ Tempo de execução: {duration:.2f} segundos")
        print(f"⏰ Conclusão: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print("📁 Outputs gerados:")
        print("  - data/processed/consolidated_data.csv")
        print("  - data/processed/data_with_features.csv")
        print("  - data/processed/data_with_anomalies.csv")
        print("  - reports/eda/")
        print("  - reports/anomalies/")
        print("  - reports/predictions/")
        print("  - models/")

        print("\n" + "=" * 80)
        print("🎉 POC COMPLETO - TODOS OS ENTREGÁVEIS GERADOS!")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ ERRO durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_full_pipeline()
    sys.exit(0 if success else 1)
