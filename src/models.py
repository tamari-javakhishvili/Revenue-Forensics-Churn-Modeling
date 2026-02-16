import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    RocCurveDisplay, 
    roc_auc_score, 
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Apply professional visual theme
sns.set_theme(style="whitegrid")


class ChurnPredictor:
    """
    Production-grade churn prediction model WITHOUT target leakage.
    
     METHODOLOGICAL NOTE:
    This implementation intentionally EXCLUDES 'Recency' from the feature set
    to avoid target leakage. Since churn is defined as Recency > threshold,
    using Recency as a predictor would allow the model to memorize the target
    definition rather than learn behavioral patterns.
    
    Instead, we rely on:
    - Frequency: Purchase count (engagement level)
    - Monetary: Total spend (customer value)
    - AOV: Average Order Value (spending behavior)
    - Country: Geographic risk factors
    - Engineered flags: Low-value segment indicators
    
    This approach simulates a real-world scenario where we predict FUTURE churn
    based on PAST transactional behavior, not current recency status.
    
    Expected Performance:
    - Realistic AUC: 0.65-0.75 (industry standard for behavioral churn models)
    - Trade-off: Lower accuracy, but genuine predictive power
    
    Attributes:
        base_dir (str): Absolute path to the current script directory.
        data_dir (str): Path to the directory containing analytical datasets.
        image_dir (str): Path to the directory for saving model performance charts.
        config (dict): Global parameters loaded from config.json.
    """
    
    def __init__(self):
        """Initializes the model pipeline and creates necessary directory structures."""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, '..', 'data')
        self.image_dir = os.path.join(self.base_dir, '..', 'images')
        
        config_path = os.path.join(self.base_dir, '..', 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.image_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def prepare_ml_data(
        self, 
        rfm_df: pd.DataFrame, 
        raw_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list, pd.DataFrame, LabelEncoder]:
        """
        Engineers features WITHOUT target leakage for production-grade predictions.
        
        ⚠️ KEY DESIGN DECISION: Recency Exclusion
        ============================================
        Recency is NOT used as a feature because:
        1. Churn target is defined as: is_churn = (Recency > threshold)
        2. Including Recency would create a deterministic relationship:
           If Recency > 90 → Model learns: "Always predict churn = 1"
        3. This is target leakage - the model memorizes the target definition
           rather than learning behavioral patterns that LEAD to churn.
        
        Real-World Analogy:
        -------------------
        We want to predict: "Will customer churn in the NEXT 30 days?"
        We cannot use: "How long since their last purchase TODAY?"
        We CAN use: "How often did they buy? How much? What's their trend?"
        
        Feature Engineering Strategy:
        ------------------------------
        1. Behavioral Metrics:
           - Frequency: Engagement level (more visits = stickier customer)
           - Monetary: Total value (high-value customers less likely to churn)
           - AOV: Spending per transaction (declining AOV = warning sign)
        
        2. Segment Indicators:
           - Low_Frequency_Flag: Bottom 25% of purchasers
           - Low_Monetary_Flag: Bottom 25% of spenders
           - These capture "at-risk" segments indirectly
        
        3. Geographic Risk:
           - Country_encoded: Some markets have higher churn rates
        
        Expected Impact:
        ----------------
        - AUC will drop from ~0.95 (with leakage) to ~0.65-0.75 (realistic)
        - This is NORMAL and EXPECTED for behavioral churn models
        - Model now learns genuine patterns, not memorized rules
        
        Args:
            rfm_df (pd.DataFrame): Calculated RFM metrics (Recency, Frequency, Monetary).
            raw_df (pd.DataFrame): Raw transaction data for country attribution.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, features, ml_data, label_encoder)
        """
        threshold = self.config.get('churn_threshold', 90)
        
        # Ensure CustomerID is available as a column
        if 'CustomerID' not in rfm_df.columns:
            rfm_df = rfm_df.reset_index()
        
        # Target Creation: Binary churn indicator
        rfm_df['is_churn'] = (rfm_df['Recency'] > threshold).astype(int)
        
        # Feature Engineering: Join Country data
        customer_countries = raw_df.groupby('CustomerID')['Country'].first()
        ml_data = rfm_df.set_index('CustomerID').join(customer_countries)
        
        # Safe AOV calculation (handles Frequency=0 edge cases)
        ml_data['AOV'] = ml_data['Monetary'] / ml_data['Frequency'].replace(0, 1)
        
        # Categorical Encoding: Country → Numeric
        le = LabelEncoder()
        ml_data['Country_encoded'] = le.fit_transform(ml_data['Country'].astype(str))
        
        # ✅ ENGINEERED FEATURES: Indirect churn signals
        # Low engagement flags (bottom quartile = higher risk)
        ml_data['Low_Frequency_Flag'] = (
            ml_data['Frequency'] < ml_data['Frequency'].quantile(0.25)
        ).astype(int)
        
        ml_data['Low_Monetary_Flag'] = (
            ml_data['Monetary'] < ml_data['Monetary'].quantile(0.25)
        ).astype(int)
        
        # Value trend indicator
        median_aov = ml_data['AOV'].median()
        ml_data['Below_Median_AOV'] = (ml_data['AOV'] < median_aov).astype(int)
        
        # Frequency-to-Monetary ratio (efficiency metric)
        # High ratio = many cheap purchases; Low ratio = few expensive purchases
        ml_data['FM_Ratio'] = ml_data['Frequency'] / (ml_data['Monetary'] + 1)  # +1 to avoid div/0
        
        # ✅ CRITICAL: Feature set WITHOUT Recency
        features = [
            'Frequency', 
            'Monetary', 
            'AOV',
            'Country_encoded',
            'Low_Frequency_Flag',
            'Low_Monetary_Flag',
            'Below_Median_AOV',
            'FM_Ratio'
        ]
        
        X = ml_data[features]
        y = ml_data['is_churn']
        
        # Stratified Split: Maintains class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_seed', 42),
            stratify=y
        )
        
        print(f"✓ Feature Engineering Complete (Leak-Free)")
        print(f"  Features: {features}")
        print(f"✓ Data Split: {len(X_train)} train, {len(X_test)} test")
        print(f"✓ Churn rate: Train={y_train.mean():.1%}, Test={y_test.mean():.1%}")
        
        return X_train, X_test, y_train, y_test, features, ml_data, le
    
    def train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> RandomForestClassifier:
        """
        Trains a Random Forest classifier with cross-validation.
        
        Model Configuration:
        - Uses class_weight='balanced' to handle churn imbalance
        - Hyperparameters pulled from config.json for reproducibility
        - 5-Fold Cross-Validation for robust performance estimation
        
        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training labels.
        
        Returns:
            RandomForestClassifier: Trained model instance.
        """
        model_config = self.config.get('model', {})
        model = RandomForestClassifier(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 10),
            min_samples_split=model_config.get('min_samples_split', 5),
            random_state=self.config.get('random_seed', 42),
            class_weight='balanced'  # Critical for imbalanced datasets
        )
        
        # Cross-Validation for model stability assessment
        print("\n=== Cross-Validation (5-Fold) ===")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"CV AUC Scores: {cv_scores}")
        print(f"Mean AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        if cv_scores.mean() < 0.60:
            print("⚠ WARNING: Low CV AUC (<0.60). This is expected without Recency.")
            print("   Model is learning behavioral patterns, not memorizing target definition.")
        
        # Final training on full training set
        model.fit(X_train, y_train)
        print("✓ Model training complete")
        
        return model
    
    def evaluate_model(
        self,
        model: RandomForestClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list
    ) -> float:
        """
        Comprehensive model evaluation with multiple metrics and visualizations.
        
        Generates:
        1. ROC Curve with AUC score
        2. Confusion Matrix with business impact interpretation
        3. Classification Report (Precision, Recall, F1)
        4. Feature Importance Plot
        5. Reality check against industry benchmarks
        
        Args:
            model: Trained RandomForestClassifier.
            X_test (pd.DataFrame): Test feature matrix.
            y_test (pd.Series): True test labels.
            feature_names (list): Names of features for visualization.
        
        Returns:
            float: Area Under the ROC Curve (AUC).
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Classification Report
        print("\n=== Classification Report ===")
        print(classification_report(
            y_test, y_pred, 
            target_names=['Active', 'Churned'],
            digits=3
        ))
        
        # Confusion Matrix with Business Context
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n=== Business Impact Analysis ===")
        print(f"✓ True Negatives (Correctly identified active): {tn}")
        print(f"✓ True Positives (Correctly identified churners): {tp}")
        print(f"⚠ False Positives (Wasted retention budget): {fp}")
        print(f"❌ False Negatives (Missed churners - REVENUE AT RISK): {fn}")
        
        # Calculate business metrics
        precision_churned = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_churned = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n Key Metrics:")
        print(f"   Precision (Campaign Efficiency): {precision_churned:.1%}")
        print(f"   Recall (Risk Coverage): {recall_churned:.1%}")
        
        # Confusion Matrix Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=['Active', 'Churned']
        )
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.title("Confusion Matrix: Churn Prediction (No Leakage)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.image_dir, "confusion_matrix.png"), 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.show()
        plt.close()
        
        # ROC Curve
        auc = roc_auc_score(y_test, y_probs)
        self.plot_roc_curve(model, X_test, y_test, auc)
        
        # Feature Importance
        self.plot_feature_importance(model, feature_names)
        
        # Reality Check
        print("\n=== Model Performance Reality Check ===")
        if auc >= 0.90:
            print("⚠ ALERT: AUC ≥ 0.90 suggests potential data leakage!")
            print("   Industry benchmark for behavioral churn: 0.65-0.80")
        elif auc >= 0.75:
            print("✓ EXCELLENT: AUC in upper range of realistic models")
        elif auc >= 0.65:
            print("✓ GOOD: AUC within expected range for behavioral prediction")
        else:
            print("⚠ LOW: AUC < 0.65 may indicate insufficient signal")
            print("   Consider additional feature engineering or data quality checks")
        
        return auc
    
    def plot_roc_curve(
        self, 
        model: RandomForestClassifier, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        auc_score: float
    ):
        """
        Visualizes the model's ROC Curve with benchmark reference.
        
        Args:
            model: Trained classifier instance.
            X_test (pd.DataFrame): Testing feature set.
            y_test (pd.Series): True labels for the test set.
            auc_score (float): Calculated Area Under the Curve.
        """
        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_estimator(
            model, X_test, y_test,
            color="darkorange",
            linewidth=2,
            name=f"Random Forest (AUC = {auc_score:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Baseline (AUC = 0.50)")
        
        # Add industry benchmark zone
        plt.axhspan(0.65, 0.80, alpha=0.1, color='green', label='Industry Benchmark Zone')
        
        plt.title("ROC Curve: Leak-Free Churn Prediction", fontsize=14, fontweight='bold')
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate (Recall)", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.image_dir, "roc_curve_final.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"✓ ROC Curve saved: {save_path}")
    
    def plot_feature_importance(
        self, 
        model: RandomForestClassifier, 
        feature_names: list
    ):
        """
        Visualizes which features the Random Forest considers most important.
        
        Args:
            model: Trained RandomForestClassifier.
            feature_names (list): Names of features used in training.
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance: Behavioral Churn Drivers", fontsize=14, fontweight='bold')
        plt.bar(
            range(len(importances)), 
            importances[indices], 
            color='teal', 
            alpha=0.8
        )
        plt.xticks(
            range(len(importances)), 
            [feature_names[i] for i in indices], 
            rotation=45,
            ha='right'
        )
        plt.ylabel("Importance Score (Mean Decrease Impurity)", fontsize=12)
        plt.xlabel("Features", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.image_dir, "feature_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"✓ Feature Importance saved: {save_path}")
        
        # Print feature ranking
        print("\n=== Feature Importance Ranking ===")
        for i, idx in enumerate(indices, 1):
            print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    def save_model_artifacts(
        self, 
        model: RandomForestClassifier, 
        label_encoder: LabelEncoder
    ):
        """
        Persists trained model and encoders for production deployment.
        
        Args:
            model: Trained model instance.
            label_encoder: Fitted LabelEncoder for categorical features.
        """
        model_path = os.path.join(self.data_dir, "churn_model.pkl")
        encoder_path = os.path.join(self.data_dir, "label_encoder.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(label_encoder, encoder_path)
        
        print(f"\n✓ Model saved: {model_path}")
        print(f"✓ Label encoder saved: {encoder_path}")
    
    def generate_predictions(
        self,
        model: RandomForestClassifier,
        ml_data: pd.DataFrame,
        features: list
    ):
        """
        Generates churn probabilities for the entire dataset (for Tableau integration).
        
        Args:
            model: Trained model.
            ml_data (pd.DataFrame): Full engineered dataset.
            features (list): Feature columns used during training.
        """
        # Generate probabilities
        ml_data['churn_probability'] = model.predict_proba(ml_data[features])[:, 1]
        ml_data['churn_prediction'] = model.predict(ml_data[features])
        
        # Add risk categories for Tableau filtering
        ml_data['risk_category'] = pd.cut(
            ml_data['churn_probability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Export for Tableau
        output_path = os.path.join(self.data_dir, "final_predictions.csv")
        ml_data.to_csv(output_path)
        
        print(f"\n✓ Predictions exported: {output_path}")
        print(f"   Columns: {list(ml_data.columns)}")
        
        # Summary statistics
        print(f"\n=== Prediction Summary ===")
        print(f"Total customers: {len(ml_data)}")
        print(f"Predicted churners (>0.5 prob): {ml_data['churn_prediction'].sum()} ({ml_data['churn_prediction'].mean():.1%})")
        print(f"High risk (>0.6 prob): {(ml_data['churn_probability'] > 0.6).sum()}")
        print(f"Avg churn probability: {ml_data['churn_probability'].mean():.3f}")
        
        # Risk distribution
        print(f"\n=== Risk Distribution ===")
        print(ml_data['risk_category'].value_counts().sort_index())


if __name__ == "__main__":
    """
    Main execution pipeline for leak-free churn prediction model.
    
     EXPECTED OUTCOMES:
    - AUC: 0.65-0.75 (realistic for behavioral models)
    - Model learns from patterns, not target definition
    - Suitable for production deployment
    """
    
    predictor = ChurnPredictor()
    
    # Define file paths
    rfm_path = os.path.join(predictor.data_dir, "final_customer_analytics.csv")
    raw_path = os.path.join(predictor.data_dir, "ecommerce_data.csv")
    
    # Fail-fast validation
    if not os.path.exists(rfm_path):
        raise FileNotFoundError(
            f" RFM file not found: {rfm_path}\n"
            "   Run analysis.py first to generate RFM metrics."
        )
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f" Raw data not found: {raw_path}\n"
            "   Run data_loader.py first to generate the dataset."
        )
    
    print("="*60)
    print("LEAK-FREE CHURN PREDICTION MODEL TRAINING")
    print("="*60)
    print("\nMETHODOLOGICAL NOTE:")
    print("This model intentionally EXCLUDES Recency to avoid target leakage.")
    print("Expected AUC: 0.65-0.75 (realistic for behavioral prediction)")
    print("="*60)
    
    # Step 1: Load Data
    print("\n[1/5] Loading datasets...")
    rfm_df = pd.read_csv(rfm_path)
    raw_df = pd.read_csv(raw_path)
    print(f"✓ RFM data: {rfm_df.shape}")
    print(f"✓ Raw data: {raw_df.shape}")
    
    # Step 2: Feature Engineering
    print("\n[2/5] Preparing ML features (leak-free)...")
    X_train, X_test, y_train, y_test, features, ml_data, le = predictor.prepare_ml_data(
        rfm_df, raw_df
    )
    
    # Step 3: Model Training
    print("\n[3/5] Training Random Forest model...")
    model = predictor.train_model(X_train, y_train)
    
    # Step 4: Evaluation
    print("\n[4/5] Evaluating model performance...")
    auc = predictor.evaluate_model(model, X_test, y_test, features)
    
    # Step 5: Persistence & Export
    print("\n[5/5] Saving artifacts and generating predictions...")
    predictor.save_model_artifacts(model, le)
    predictor.generate_predictions(model, ml_data, features)
    
    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE | Model AUC: {auc:.3f}")
    print("="*60)
    print("\n Next Steps:")
    print("   1. Review visualizations in /images directory")
    print("   2. Import final_predictions.csv into Tableau")
    print("   3. Use churn_model.pkl for production deployment")
    print("\n Portfolio Tip:")
    print("   Mention this model is leak-free (excludes Recency)")
    print("   Realistic AUC shows understanding of ML best practices")