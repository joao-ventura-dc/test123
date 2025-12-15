"""
Deteção de Anomalias - POC Análise de Posturas
Implementa múltiplos algoritmos para detetar posturas anómalas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Classe para deteção de anomalias em posturas de trabalho."""

    def __init__(self, data_path='data/processed/data_with_features.csv'):
        """Inicializa o detetor de anomalias."""
        print("=" * 80)
        print("DETEÇÃO DE ANOMALIAS - POSTURAS DE TRABALHO")
        print("=" * 80)

        print(f"\n📂 Carregando dados de: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"✅ Dados carregados: {len(self.df):,} registos")

        # Selecionar features numéricas para análise
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Excluir colunas de metadata
        exclude_cols = ['hour', 'day_of_week', 'is_weekend']
        self.feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        print(f"📊 Features para análise: {len(self.feature_cols)}")

        self.output_dir = Path('reports/anomalies')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Preparar dados
        self.X = self.df[self.feature_cols].fillna(self.df[self.feature_cols].median())

        # Normalizar dados
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def isolation_forest_detection(self, contamination=0.1):
        """Deteção de anomalias usando Isolation Forest."""
        print("\n" + "=" * 80)
        print("🌲 ISOLATION FOREST")
        print("=" * 80)

        print(f"\n⚙️ Treinando modelo (contamination={contamination})...")
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        # Treinar e prever
        predictions = model.fit_predict(self.X_scaled)
        scores = model.score_samples(self.X_scaled)

        # -1 para anomalia, 1 para normal
        self.df['iso_forest_anomaly'] = predictions
        self.df['iso_forest_score'] = scores

        n_anomalies = (predictions == -1).sum()
        pct_anomalies = (n_anomalies / len(predictions)) * 100

        print(f"\n📊 Resultados:")
        print(f"  - Anomalias detetadas: {n_anomalies:,} ({pct_anomalies:.2f}%)")
        print(f"  - Registos normais: {(predictions == 1).sum():,}")

        # Análise das anomalias
        anomaly_df = self.df[self.df['iso_forest_anomaly'] == -1]

        if 'risk_category' in self.df.columns and len(anomaly_df) > 0:
            print(f"\n🎯 Distribuição de Risco nas Anomalias:")
            risk_dist = anomaly_df['risk_category'].value_counts()
            for category, count in risk_dist.items():
                pct = (count / len(anomaly_df)) * 100
                print(f"  - {category}: {count} ({pct:.1f}%)")

        return predictions, scores

    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """Deteção de anomalias usando DBSCAN (outliers como noise)."""
        print("\n" + "=" * 80)
        print("🔍 DBSCAN CLUSTERING")
        print("=" * 80)

        print(f"\n⚙️ Treinando modelo (eps={eps}, min_samples={min_samples})...")

        # Reduzir dimensionalidade para melhor performance
        pca = PCA(n_components=min(10, self.X_scaled.shape[1]))
        X_pca = pca.fit_transform(self.X_scaled)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(X_pca)

        self.df['dbscan_cluster'] = clusters
        # -1 indica noise/anomalia no DBSCAN
        self.df['dbscan_anomaly'] = (clusters == -1).astype(int)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = (clusters == -1).sum()
        pct_noise = (n_noise / len(clusters)) * 100

        print(f"\n📊 Resultados:")
        print(f"  - Clusters encontrados: {n_clusters}")
        print(f"  - Noise/Anomalias: {n_noise:,} ({pct_noise:.2f}%)")

        # Distribuição por cluster
        cluster_dist = pd.Series(clusters).value_counts().sort_index()
        print(f"\n📈 Distribuição por Cluster:")
        for cluster, count in cluster_dist.items():
            cluster_name = "Noise/Anomalia" if cluster == -1 else f"Cluster {cluster}"
            print(f"  - {cluster_name}: {count:,}")

        return clusters

    def elliptic_envelope_detection(self, contamination=0.1):
        """Deteção de anomalias usando Elliptic Envelope (Gaussian)."""
        print("\n" + "=" * 80)
        print("🎯 ELLIPTIC ENVELOPE")
        print("=" * 80)

        print(f"\n⚙️ Treinando modelo (contamination={contamination})...")

        # Elliptic Envelope funciona melhor com menos features
        pca = PCA(n_components=min(5, self.X_scaled.shape[1]))
        X_pca = pca.fit_transform(self.X_scaled)

        model = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X_pca)

        self.df['elliptic_anomaly'] = predictions
        n_anomalies = (predictions == -1).sum()
        pct_anomalies = (n_anomalies / len(predictions)) * 100

        print(f"\n📊 Resultados:")
        print(f"  - Anomalias detetadas: {n_anomalies:,} ({pct_anomalies:.2f}%)")
        print(f"  - Registos normais: {(predictions == 1).sum():,}")

        return predictions

    def statistical_anomaly_detection(self, threshold=3):
        """Deteção de anomalias usando método estatístico (Z-score)."""
        print("\n" + "=" * 80)
        print("📐 DETEÇÃO ESTATÍSTICA (Z-Score)")
        print("=" * 80)

        print(f"\n⚙️ Calculando Z-scores (threshold={threshold})...")

        # Calcular Z-score para cada feature
        z_scores = np.abs((self.X - self.X.mean()) / self.X.std())

        # Anomalia se qualquer feature ultrapassar o threshold
        is_anomaly = (z_scores > threshold).any(axis=1)

        self.df['statistical_anomaly'] = is_anomaly.astype(int)
        n_anomalies = is_anomaly.sum()
        pct_anomalies = (n_anomalies / len(is_anomaly)) * 100

        print(f"\n📊 Resultados:")
        print(f"  - Anomalias detetadas: {n_anomalies:,} ({pct_anomalies:.2f}%)")
        print(f"  - Registos normais: {(~is_anomaly).sum():,}")

        return is_anomaly

    def ensemble_anomaly_detection(self):
        """Combina múltiplos métodos para deteção mais robusta."""
        print("\n" + "=" * 80)
        print("🎭 ENSEMBLE DE MÉTODOS")
        print("=" * 80)

        # Contar quantos métodos classificaram como anomalia
        anomaly_cols = ['iso_forest_anomaly', 'dbscan_anomaly', 'elliptic_anomaly', 'statistical_anomaly']

        # Converter para binário (1 = anomalia)
        anomaly_flags = pd.DataFrame()
        for col in anomaly_cols:
            if col in self.df.columns:
                if col == 'iso_forest_anomaly' or col == 'elliptic_anomaly':
                    # -1 significa anomalia nestes métodos
                    anomaly_flags[col] = (self.df[col] == -1).astype(int)
                else:
                    anomaly_flags[col] = self.df[col]

        # Contar votos
        self.df['anomaly_votes'] = anomaly_flags.sum(axis=1)

        # Anomalia se pelo menos 2 métodos concordarem
        self.df['ensemble_anomaly'] = (self.df['anomaly_votes'] >= 2).astype(int)

        n_anomalies = self.df['ensemble_anomaly'].sum()
        pct_anomalies = (n_anomalies / len(self.df)) * 100

        print(f"\n📊 Resultados do Ensemble:")
        print(f"  - Anomalias detetadas: {n_anomalies:,} ({pct_anomalies:.2f}%)")

        # Distribuição de votos
        print(f"\n🗳️ Distribuição de Votos:")
        vote_dist = self.df['anomaly_votes'].value_counts().sort_index()
        for votes, count in vote_dist.items():
            pct = (count / len(self.df)) * 100
            print(f"  - {votes} métodos: {count:,} ({pct:.1f}%)")

        return self.df['ensemble_anomaly']

    def visualize_anomalies(self):
        """Gera visualizações das anomalias detetadas."""
        print("\n" + "=" * 80)
        print("📊 GERANDO VISUALIZAÇÕES")
        print("=" * 80)

        # 1. PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        plt.figure(figsize=(12, 5))

        # Plot 1: Isolation Forest
        plt.subplot(1, 2, 1)
        normal = self.df['iso_forest_anomaly'] == 1
        plt.scatter(X_pca[normal, 0], X_pca[normal, 1],
                   c='blue', alpha=0.5, s=20, label='Normal')
        plt.scatter(X_pca[~normal, 0], X_pca[~normal, 1],
                   c='red', alpha=0.7, s=30, label='Anomalia', marker='x')
        plt.title('Isolation Forest - PCA Visualization')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.legend()

        # Plot 2: Ensemble
        plt.subplot(1, 2, 2)
        normal = self.df['ensemble_anomaly'] == 0
        plt.scatter(X_pca[normal, 0], X_pca[normal, 1],
                   c='blue', alpha=0.5, s=20, label='Normal')
        plt.scatter(X_pca[~normal, 0], X_pca[~normal, 1],
                   c='red', alpha=0.7, s=30, label='Anomalia', marker='x')
        plt.title('Ensemble Method - PCA Visualization')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico: anomaly_visualization.png")

        # 2. Scores distribution
        if 'iso_forest_score' in self.df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(self.df['iso_forest_score'], bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(self.df[self.df['iso_forest_anomaly'] == -1]['iso_forest_score'].max(),
                       color='red', linestyle='--', linewidth=2, label='Threshold')
            plt.title('Distribuição de Anomaly Scores (Isolation Forest)')
            plt.xlabel('Score')
            plt.ylabel('Frequência')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'anomaly_scores.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico: anomaly_scores.png")

        # 3. Comparison of methods
        methods = ['iso_forest_anomaly', 'dbscan_anomaly', 'elliptic_anomaly',
                   'statistical_anomaly', 'ensemble_anomaly']
        method_names = ['Isolation\nForest', 'DBSCAN', 'Elliptic\nEnvelope',
                       'Statistical', 'Ensemble']

        anomaly_counts = []
        for method in methods:
            if method in self.df.columns:
                if method in ['iso_forest_anomaly', 'elliptic_anomaly']:
                    count = (self.df[method] == -1).sum()
                else:
                    count = self.df[method].sum()
                anomaly_counts.append(count)
            else:
                anomaly_counts.append(0)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(method_names[:len(anomaly_counts)], anomaly_counts,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])

        plt.title('Comparação de Métodos de Deteção de Anomalias', fontsize=14, fontweight='bold')
        plt.ylabel('Número de Anomalias Detetadas')
        plt.xticks(rotation=0)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico: method_comparison.png")

        print(f"\n📁 Visualizações salvas em: {self.output_dir}")

    def generate_anomaly_report(self):
        """Gera relatório detalhado sobre anomalias."""
        print("\n" + "=" * 80)
        print("📝 GERANDO RELATÓRIO DE ANOMALIAS")
        print("=" * 80)

        report = []
        report.append("# RELATÓRIO DE DETEÇÃO DE ANOMALIAS")
        report.append("## POC: Análise de Posturas de Trabalho\n")
        report.append("---\n")

        report.append("## 1. RESUMO DOS MÉTODOS\n")

        methods_info = {
            'Isolation Forest': 'iso_forest_anomaly',
            'DBSCAN Clustering': 'dbscan_anomaly',
            'Elliptic Envelope': 'elliptic_anomaly',
            'Statistical (Z-Score)': 'statistical_anomaly',
            'Ensemble': 'ensemble_anomaly'
        }

        for method_name, col in methods_info.items():
            if col in self.df.columns:
                if col in ['iso_forest_anomaly', 'elliptic_anomaly']:
                    n_anomalies = (self.df[col] == -1).sum()
                else:
                    n_anomalies = self.df[col].sum()

                pct = (n_anomalies / len(self.df)) * 100
                report.append(f"### {method_name}")
                report.append(f"- Anomalias: {n_anomalies:,} ({pct:.2f}%)")
                report.append(f"- Registos normais: {len(self.df) - n_anomalies:,}\n")

        report.append("\n## 2. ANÁLISE DAS ANOMALIAS (ENSEMBLE)\n")

        if 'ensemble_anomaly' in self.df.columns:
            anomalies = self.df[self.df['ensemble_anomaly'] == 1]

            report.append(f"**Total de anomalias detetadas**: {len(anomalies):,}\n")

            if 'risk_category' in self.df.columns and len(anomalies) > 0:
                report.append("### Distribuição por Categoria de Risco:\n")
                risk_dist = anomalies['risk_category'].value_counts()
                for category, count in risk_dist.items():
                    pct = (count / len(anomalies)) * 100
                    report.append(f"- **{category}**: {count} ({pct:.1f}%)")
                report.append("")

            # Top features nas anomalias
            score_cols = [col for col in self.df.columns if 'score' in col.lower()]
            if score_cols:
                report.append("### Scores Médios nas Anomalias:\n")
                for col in score_cols:
                    if col in anomalies.columns:
                        avg_score = anomalies[col].mean()
                        report.append(f"- **{col}**: {avg_score:.2f}")
                report.append("")

        report.append("\n## 3. RECOMENDAÇÕES\n")
        report.append("1. **Investigar registos com ensemble_anomaly = 1** - múltiplos métodos concordam")
        report.append("2. **Priorizar anomalias com categoria de risco 'Crítico' ou 'Alto'**")
        report.append("3. **Analisar padrões temporais nas anomalias**")
        report.append("4. **Validar anomalias com especialistas ergonómicos**")
        report.append("")

        report.append("\n## 4. PRÓXIMOS PASSOS\n")
        report.append("- Implementar sistema de alertas em tempo real")
        report.append("- Criar modelo preditivo para antecipar anomalias")
        report.append("- Desenvolver dashboard interativo")
        report.append("- Integrar com sensores para monitorização contínua")

        # Salvar relatório
        report_path = self.output_dir / 'anomaly_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ Relatório salvo em: {report_path}")

        # Salvar dataset com anomalias
        output_path = Path('data/processed/data_with_anomalies.csv')
        self.df.to_csv(output_path, index=False)
        print(f"✅ Dataset com anomalias salvo em: {output_path}")

        # Salvar apenas anomalias
        if 'ensemble_anomaly' in self.df.columns:
            anomalies = self.df[self.df['ensemble_anomaly'] == 1]
            anomalies_path = Path('data/processed/detected_anomalies.csv')
            anomalies.to_csv(anomalies_path, index=False)
            print(f"✅ Anomalias detetadas salvas em: {anomalies_path}")

    def run_full_detection(self):
        """Executa deteção completa de anomalias."""
        self.isolation_forest_detection()
        self.dbscan_clustering()
        self.elliptic_envelope_detection()
        self.statistical_anomaly_detection()
        self.ensemble_anomaly_detection()
        self.visualize_anomalies()
        self.generate_anomaly_report()

        print("\n" + "=" * 80)
        print("✅ DETEÇÃO DE ANOMALIAS COMPLETA!")
        print("=" * 80)

        return self.df


if __name__ == "__main__":
    detector = AnomalyDetector()
    df = detector.run_full_detection()
