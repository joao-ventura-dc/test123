"""
Análise Exploratória de Dados (EDA) - POC Análise de Posturas
Realiza análise exploratória completa e engenharia de features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PostureEDA:
    """Classe para análise exploratória de dados de posturas."""

    def __init__(self, data_path='data/processed/consolidated_data.csv'):
        """Inicializa e carrega os dados."""
        print("=" * 80)
        print("ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
        print("=" * 80)
        print(f"\n📂 Carregando dados de: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"✅ Dados carregados: {len(self.df):,} registos")

        # Identificar colunas de scores e articulações
        self.score_cols = [col for col in self.df.columns if 'score' in col.lower()]
        self.joint_cols = ['neck', 'trunk', 'knee', 'arm', 'forearm', 'hand']
        self.joint_cols = [col for col in self.joint_cols if col in self.df.columns]

        self.output_dir = Path('reports/eda')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def basic_statistics(self):
        """Estatísticas básicas dos dados."""
        print("\n" + "=" * 80)
        print("📊 ESTATÍSTICAS BÁSICAS")
        print("=" * 80)

        print(f"\nShape: {self.df.shape}")
        print(f"\nInfo:")
        self.df.info()

        print("\n📈 Estatísticas Descritivas (Scores):")
        if self.score_cols:
            print(self.df[self.score_cols].describe())

        print("\n📈 Estatísticas Descritivas (Articulações):")
        if self.joint_cols:
            print(self.df[self.joint_cols].describe())

        # Missing values
        print("\n❓ Valores Missing:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))

    def data_cleaning(self):
        """Limpeza e normalização dos dados."""
        print("\n" + "=" * 80)
        print("🧹 LIMPEZA E NORMALIZAÇÃO")
        print("=" * 80)

        initial_rows = len(self.df)

        # Remover duplicados
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"\n🔍 Duplicados encontrados: {duplicates}")
            self.df = self.df.drop_duplicates()
            print(f"✅ Duplicados removidos")

        # Tratar valores missing
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                # Usar mediana para valores numéricos
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)

        final_rows = len(self.df)
        print(f"\n📊 Registos: {initial_rows:,} → {final_rows:,} (removidos: {initial_rows - final_rows:,})")

        return self.df

    def feature_engineering(self):
        """Engenharia de features."""
        print("\n" + "=" * 80)
        print("⚙️ ENGENHARIA DE FEATURES")
        print("=" * 80)

        # Score médio geral
        if self.score_cols:
            self.df['avg_score'] = self.df[self.score_cols].mean(axis=1)
            print(f"✅ Criada feature: avg_score")

        # Score máximo (pior postura)
        if self.score_cols:
            self.df['max_score'] = self.df[self.score_cols].max(axis=1)
            print(f"✅ Criada feature: max_score")

        # Variabilidade postural
        if self.joint_cols:
            self.df['joint_std'] = self.df[self.joint_cols].std(axis=1)
            print(f"✅ Criada feature: joint_std (variabilidade postural)")

        # Categoria de risco baseada em scores
        if 'avg_score' in self.df.columns:
            self.df['risk_category'] = pd.cut(
                self.df['avg_score'],
                bins=[-np.inf, 2, 4, 7, np.inf],
                labels=['Baixo', 'Moderado', 'Alto', 'Crítico']
            )
            print(f"✅ Criada feature: risk_category")

        # Features temporais (se houver timestamp)
        if 'timestamp' in self.df.columns or 'Timestamp' in self.df.columns:
            timestamp_col = 'timestamp' if 'timestamp' in self.df.columns else 'Timestamp'
            try:
                self.df['datetime'] = pd.to_datetime(self.df[timestamp_col])
                self.df['hour'] = self.df['datetime'].dt.hour
                self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
                self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
                print(f"✅ Criadas features temporais: hour, day_of_week, is_weekend")
            except:
                print(f"⚠️ Não foi possível criar features temporais")

        print(f"\n📊 Total de features: {len(self.df.columns)}")

        return self.df

    def visualizations(self):
        """Gerar visualizações exploratórias."""
        print("\n" + "=" * 80)
        print("📊 GERANDO VISUALIZAÇÕES")
        print("=" * 80)

        # 1. Distribuição de Scores
        if self.score_cols:
            fig, axes = plt.subplots(1, len(self.score_cols), figsize=(15, 4))
            if len(self.score_cols) == 1:
                axes = [axes]

            for idx, col in enumerate(self.score_cols):
                axes[idx].hist(self.df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribuição {col}')
                axes[idx].set_xlabel('Score')
                axes[idx].set_ylabel('Frequência')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico: score_distributions.png")

        # 2. Correlação entre variáveis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Matriz de Correlação', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico: correlation_matrix.png")

        # 3. Boxplots de articulações
        if self.joint_cols:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, col in enumerate(self.joint_cols[:6]):
                if col in self.df.columns:
                    axes[idx].boxplot(self.df[col].dropna())
                    axes[idx].set_title(f'{col.upper()}')
                    axes[idx].set_ylabel('Valor')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'joint_boxplots.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico: joint_boxplots.png")

        # 4. Categoria de risco
        if 'risk_category' in self.df.columns:
            plt.figure(figsize=(10, 6))
            risk_counts = self.df['risk_category'].value_counts()
            colors = ['green', 'yellow', 'orange', 'red']
            plt.bar(risk_counts.index, risk_counts.values, color=colors[:len(risk_counts)])
            plt.title('Distribuição de Categorias de Risco', fontsize=16, fontweight='bold')
            plt.xlabel('Categoria')
            plt.ylabel('Número de Registos')
            plt.xticks(rotation=45)

            for i, v in enumerate(risk_counts.values):
                plt.text(i, v + max(risk_counts.values) * 0.01, str(v),
                        ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'risk_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico: risk_distribution.png")

        print(f"\n📁 Visualizações salvas em: {self.output_dir}")

    def pattern_analysis(self):
        """Análise de padrões nos dados."""
        print("\n" + "=" * 80)
        print("🔍 ANÁLISE DE PADRÕES")
        print("=" * 80)

        # Análise por categoria de risco
        if 'risk_category' in self.df.columns:
            print("\n📊 Distribuição por Categoria de Risco:")
            risk_dist = self.df['risk_category'].value_counts()
            for category, count in risk_dist.items():
                pct = (count / len(self.df)) * 100
                print(f"  - {category}: {count:,} ({pct:.1f}%)")

        # Correlações mais fortes
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            print("\n🔗 Top 10 Correlações Mais Fortes:")

            # Extrair correlações únicas
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

            corr_df = pd.DataFrame(correlations)
            corr_df = corr_df.sort_values('correlation', ascending=False, key=abs)

            for idx, row in corr_df.head(10).iterrows():
                print(f"  - {row['var1']} ↔ {row['var2']}: {row['correlation']:.3f}")

        # Estatísticas por câmera
        if 'camera_id' in self.df.columns and self.score_cols:
            print("\n📹 Estatísticas por Câmera:")
            camera_stats = self.df.groupby('camera_id')[self.score_cols].mean()
            print(camera_stats)

    def generate_report(self):
        """Gera relatório completo da análise."""
        print("\n" + "=" * 80)
        print("📝 GERANDO RELATÓRIO")
        print("=" * 80)

        report = []
        report.append("# RELATÓRIO DE ANÁLISE EXPLORATÓRIA")
        report.append("## POC: Análise e Previsão de Posturas de Trabalho com IA\n")
        report.append("---\n")

        report.append("## 1. RESUMO EXECUTIVO\n")
        report.append(f"- **Total de registos**: {len(self.df):,}")
        report.append(f"- **Total de features**: {len(self.df.columns)}")
        report.append(f"- **Período de dados**: {self.df['recording_date'].min()} a {self.df['recording_date'].max()}\n")

        if 'risk_category' in self.df.columns:
            report.append("### Distribuição de Risco:\n")
            risk_dist = self.df['risk_category'].value_counts()
            for category, count in risk_dist.items():
                pct = (count / len(self.df)) * 100
                report.append(f"- **{category}**: {count:,} registos ({pct:.1f}%)")
            report.append("")

        report.append("\n## 2. QUALIDADE DOS DADOS\n")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            report.append("✅ Nenhum valor missing após limpeza\n")
        else:
            report.append("### Valores Missing:\n")
            for col, miss in missing[missing > 0].items():
                pct = (miss / len(self.df)) * 100
                report.append(f"- {col}: {miss} ({pct:.1f}%)")
            report.append("")

        report.append("\n## 3. FEATURES CRIADAS\n")
        new_features = ['avg_score', 'max_score', 'joint_std', 'risk_category']
        for feature in new_features:
            if feature in self.df.columns:
                report.append(f"- ✅ {feature}")
        report.append("")

        report.append("\n## 4. VISUALIZAÇÕES GERADAS\n")
        report.append("- score_distributions.png")
        report.append("- correlation_matrix.png")
        report.append("- joint_boxplots.png")
        report.append("- risk_distribution.png")
        report.append("")

        report.append("\n## 5. PRÓXIMOS PASSOS\n")
        report.append("- Implementar modelos de deteção de anomalias")
        report.append("- Criar modelo preditivo temporal")
        report.append("- Validar previsões de risco")
        report.append("- Desenvolver API e dashboard")

        # Salvar relatório
        report_path = self.output_dir / 'eda_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ Relatório salvo em: {report_path}")

        # Salvar dataset processado
        processed_path = Path('data/processed/data_with_features.csv')
        self.df.to_csv(processed_path, index=False)
        print(f"✅ Dataset com features salvo em: {processed_path}")

    def run_full_analysis(self):
        """Executa análise completa."""
        self.basic_statistics()
        self.data_cleaning()
        self.feature_engineering()
        self.visualizations()
        self.pattern_analysis()
        self.generate_report()

        print("\n" + "=" * 80)
        print("✅ ANÁLISE EXPLORATÓRIA COMPLETA!")
        print("=" * 80)

        return self.df


if __name__ == "__main__":
    eda = PostureEDA()
    df = eda.run_full_analysis()
