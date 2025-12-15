"""
Modelo Preditivo Temporal - POC Análise de Posturas
Cria modelo para prever scores e riscos futuros.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class PredictiveModel:
    """Classe para modelos preditivos de posturas."""

    def __init__(self, data_path='data/processed/data_with_anomalies.csv'):
        """Inicializa o modelo preditivo."""
        print("=" * 80)
        print("MODELO PREDITIVO TEMPORAL - POSTURAS DE TRABALHO")
        print("=" * 80)

        print(f"\n📂 Carregando dados de: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"✅ Dados carregados: {len(self.df):,} registos")

        self.output_dir = Path('reports/predictions')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = Path('models')
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_features(self):
        """Prepara features para modelagem."""
        print("\n" + "=" * 80)
        print("⚙️ PREPARAÇÃO DE FEATURES")
        print("=" * 80)

        # Identificar features disponíveis
        joint_cols = ['neck', 'trunk', 'knee', 'arm', 'forearm', 'hand']
        joint_cols = [col for col in joint_cols if col in self.df.columns]

        # Features adicionais
        additional_features = ['joint_std', 'hour', 'day_of_week', 'is_weekend', 'ensemble_anomaly']
        additional_features = [col for col in additional_features if col in self.df.columns]

        self.feature_cols = joint_cols + additional_features

        print(f"📊 Features selecionadas ({len(self.feature_cols)}):")
        for col in self.feature_cols:
            print(f"  - {col}")

        # Targets para previsão
        self.score_targets = [col for col in self.df.columns if 'score' in col.lower() and 'iso_forest' not in col]
        self.risk_target = 'risk_category' if 'risk_category' in self.df.columns else None

        print(f"\n🎯 Targets de regressão ({len(self.score_targets)}):")
        for col in self.score_targets:
            print(f"  - {col}")

        if self.risk_target:
            print(f"\n🎯 Target de classificação: {self.risk_target}")

    def train_regression_models(self):
        """Treina modelos de regressão para prever scores."""
        print("\n" + "=" * 80)
        print("📈 MODELOS DE REGRESSÃO (PREVISÃO DE SCORES)")
        print("=" * 80)

        results = {}

        for target in self.score_targets:
            print(f"\n🔵 Treinando modelo para: {target}")
            print("-" * 60)

            # Preparar dados
            df_clean = self.df[self.feature_cols + [target]].dropna()
            X = df_clean[self.feature_cols]
            y = df_clean[target]

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Normalizar
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Treinar Random Forest
            print("  🌲 Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)

            # Previsões
            y_pred_train = rf_model.predict(X_train_scaled)
            y_pred_test = rf_model.predict(X_test_scaled)

            # Métricas
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)

            print(f"    ✓ Train R²: {train_r2:.3f}")
            print(f"    ✓ Test R²: {test_r2:.3f}")
            print(f"    ✓ Test RMSE: {test_rmse:.3f}")
            print(f"    ✓ Test MAE: {test_mae:.3f}")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n    📊 Top 5 Features Importantes:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"      - {row['feature']}: {row['importance']:.3f}")

            # Salvar modelo
            model_filename = f"rf_regressor_{target}.joblib"
            joblib.dump(rf_model, self.models_dir / model_filename)
            joblib.dump(scaler, self.models_dir / f"scaler_{target}.joblib")
            print(f"\n    💾 Modelo salvo: {model_filename}")

            # Guardar resultados
            results[target] = {
                'model': rf_model,
                'scaler': scaler,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'feature_importance': feature_importance,
                'y_test': y_test,
                'y_pred': y_pred_test
            }

        self.regression_results = results
        return results

    def train_classification_model(self):
        """Treina modelo de classificação para prever categoria de risco."""
        if not self.risk_target or self.risk_target not in self.df.columns:
            print("\n⚠️ Target de classificação não disponível")
            return None

        print("\n" + "=" * 80)
        print("🎯 MODELO DE CLASSIFICAÇÃO (PREVISÃO DE RISCO)")
        print("=" * 80)

        # Preparar dados
        df_clean = self.df[self.feature_cols + [self.risk_target]].dropna()
        X = df_clean[self.feature_cols]
        y = df_clean[self.risk_target]

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        print(f"\n📊 Distribuição de classes:")
        for i, class_name in enumerate(le.classes_):
            count = (y_encoded == i).sum()
            pct = (count / len(y_encoded)) * 100
            print(f"  - {class_name}: {count} ({pct:.1f}%)")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Treinar Random Forest Classifier
        print("\n🌲 Treinando Random Forest Classifier...")
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf_clf.fit(X_train_scaled, y_train)

        # Previsões
        y_pred_train = rf_clf.predict(X_train_scaled)
        y_pred_test = rf_clf.predict(X_test_scaled)

        # Métricas
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"\n📊 Resultados:")
        print(f"  ✓ Train Accuracy: {train_acc:.3f}")
        print(f"  ✓ Test Accuracy: {test_acc:.3f}")

        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=le.classes_))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': rf_clf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n📊 Top 5 Features Importantes:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.3f}")

        # Salvar modelo
        joblib.dump(rf_clf, self.models_dir / "rf_classifier_risk.joblib")
        joblib.dump(scaler, self.models_dir / "scaler_risk.joblib")
        joblib.dump(le, self.models_dir / "label_encoder_risk.joblib")
        print(f"\n💾 Modelo de classificação salvo")

        self.classification_result = {
            'model': rf_clf,
            'scaler': scaler,
            'label_encoder': le,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred': y_pred_test
        }

        return self.classification_result

    def visualize_predictions(self):
        """Gera visualizações dos modelos preditivos."""
        print("\n" + "=" * 80)
        print("📊 GERANDO VISUALIZAÇÕES")
        print("=" * 80)

        # 1. Regression results
        if hasattr(self, 'regression_results') and self.regression_results:
            n_targets = len(self.regression_results)
            fig, axes = plt.subplots(1, min(n_targets, 3), figsize=(15, 4))
            if n_targets == 1:
                axes = [axes]

            for idx, (target, results) in enumerate(list(self.regression_results.items())[:3]):
                ax = axes[idx]
                ax.scatter(results['y_test'], results['y_pred'], alpha=0.5)
                ax.plot([results['y_test'].min(), results['y_test'].max()],
                       [results['y_test'].min(), results['y_test'].max()],
                       'r--', lw=2)
                ax.set_xlabel('Valor Real')
                ax.set_ylabel('Previsão')
                ax.set_title(f'{target}\nR² = {results["test_r2"]:.3f}')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'regression_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico: regression_predictions.png")

            # Feature importance
            if len(self.regression_results) > 0:
                first_target = list(self.regression_results.keys())[0]
                fi = self.regression_results[first_target]['feature_importance'].head(10)

                plt.figure(figsize=(10, 6))
                plt.barh(fi['feature'], fi['importance'])
                plt.xlabel('Importância')
                plt.title(f'Feature Importance - {first_target}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(self.output_dir / 'feature_importance_regression.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Gráfico: feature_importance_regression.png")

        # 2. Classification results
        if hasattr(self, 'classification_result') and self.classification_result:
            # Confusion Matrix
            cm = self.classification_result['confusion_matrix']
            le = self.classification_result['label_encoder']

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=le.classes_,
                       yticklabels=le.classes_)
            plt.title('Confusion Matrix - Risk Category')
            plt.ylabel('Valor Real')
            plt.xlabel('Previsão')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico: confusion_matrix.png")

            # Feature importance
            fi = self.classification_result['feature_importance'].head(10)

            plt.figure(figsize=(10, 6))
            plt.barh(fi['feature'], fi['importance'])
            plt.xlabel('Importância')
            plt.title('Feature Importance - Risk Classification')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance_classification.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico: feature_importance_classification.png")

        print(f"\n📁 Visualizações salvas em: {self.output_dir}")

    def generate_prediction_report(self):
        """Gera relatório sobre modelos preditivos."""
        print("\n" + "=" * 80)
        print("📝 GERANDO RELATÓRIO DE PREVISÕES")
        print("=" * 80)

        report = []
        report.append("# RELATÓRIO DE MODELOS PREDITIVOS")
        report.append("## POC: Análise de Posturas de Trabalho\n")
        report.append("---\n")

        report.append("## 1. MODELOS DE REGRESSÃO (SCORES)\n")

        if hasattr(self, 'regression_results'):
            for target, results in self.regression_results.items():
                report.append(f"### {target}\n")
                report.append(f"- **R² (Test)**: {results['test_r2']:.3f}")
                report.append(f"- **RMSE**: {results['test_rmse']:.3f}")
                report.append(f"- **MAE**: {results['test_mae']:.3f}")
                report.append(f"\n**Top 3 Features:**")
                for idx, row in results['feature_importance'].head(3).iterrows():
                    report.append(f"- {row['feature']}: {row['importance']:.3f}")
                report.append("")

        report.append("\n## 2. MODELO DE CLASSIFICAÇÃO (RISCO)\n")

        if hasattr(self, 'classification_result'):
            result = self.classification_result
            report.append(f"- **Accuracy (Test)**: {result['test_acc']:.3f}")
            report.append(f"\n**Top 3 Features:**")
            for idx, row in result['feature_importance'].head(3).iterrows():
                report.append(f"- {row['feature']}: {row['importance']:.3f}")
            report.append("")

        report.append("\n## 3. MODELOS SALVOS\n")
        report.append("Os seguintes modelos foram salvos na pasta `models/`:\n")

        model_files = list(Path('models').glob('*.joblib'))
        for model_file in model_files:
            report.append(f"- {model_file.name}")

        report.append("\n## 4. COMO USAR OS MODELOS\n")
        report.append("```python")
        report.append("import joblib")
        report.append("import pandas as pd")
        report.append("")
        report.append("# Carregar modelo")
        report.append("model = joblib.load('models/rf_regressor_scoreA.joblib')")
        report.append("scaler = joblib.load('models/scaler_scoreA.joblib')")
        report.append("")
        report.append("# Preparar dados")
        report.append("X_new = pd.DataFrame(...)")
        report.append("X_scaled = scaler.transform(X_new)")
        report.append("")
        report.append("# Fazer previsão")
        report.append("predictions = model.predict(X_scaled)")
        report.append("```")

        report.append("\n## 5. PRÓXIMOS PASSOS\n")
        report.append("- Implementar validação cruzada mais robusta")
        report.append("- Testar outros algoritmos (XGBoost, Neural Networks)")
        report.append("- Implementar previsões em tempo real via API")
        report.append("- Criar sistema de alertas baseado nas previsões")
        report.append("- Desenvolver interface para visualização das previsões")

        # Salvar relatório
        report_path = self.output_dir / 'prediction_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ Relatório salvo em: {report_path}")

    def run_full_pipeline(self):
        """Executa pipeline completo de modelagem preditiva."""
        self.prepare_features()
        self.train_regression_models()
        self.train_classification_model()
        self.visualize_predictions()
        self.generate_prediction_report()

        print("\n" + "=" * 80)
        print("✅ MODELAGEM PREDITIVA COMPLETA!")
        print("=" * 80)


if __name__ == "__main__":
    model = PredictiveModel()
    model.run_full_pipeline()
