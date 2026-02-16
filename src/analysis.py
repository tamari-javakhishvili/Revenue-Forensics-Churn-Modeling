import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set professional visualization style for reporting
sns.set_theme(style="whitegrid")

class RevenueAnalyzer:
    """
    Forensic analysis module for RFM metrics and market collapse validation.

    This class provides a comprehensive pipeline to calculate customer-centric 
    metrics (RFM), identify optimal market segments through vector-based 
    elbow detection, and statistically validate revenue anomalies.

    Attributes:
        base_dir (str): Absolute path to the directory containing this script.
        data_dir (str): Directory path for storing/loading CSV datasets.
        image_dir (str): Directory path for saving generated visualizations.
        config (dict): Configuration parameters loaded from the project root.
    """

    def __init__(self):
        """
        Initializes the RevenueAnalyzer, configures file paths, and ensures 
        directory readiness for data output and visualization.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, '..', 'data')
        self.image_dir = os.path.join(self.base_dir, '..', 'images')
        self.config = self._load_config()
        
        # Ensure image directory exists for automated chart saving
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def _load_config(self):
        """
        Loads simulation and analysis parameters from the central config.json file.

        Returns:
            dict: Configuration parameters. Defaults to safe values if file is missing.
        """
        config_path = os.path.join(self.base_dir, '..', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {"random_seed": 42, "n_customers": 5000}

    def perform_rfm_analysis(self, df):
        """
        Calculates Recency, Frequency, and Monetary (RFM) metrics per customer.

        Args:
            df (pd.DataFrame): Raw transaction data containing 'CustomerID', 
                               'InvoiceDate', 'InvoiceNo', and 'TotalAmount'.

        Returns:
            pd.DataFrame: RFM metrics indexed by CustomerID.
        """
        ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (ref_date - x.max()).days,
            'InvoiceNo': 'count',
            'TotalAmount': 'sum'
        }).rename(columns={
            'InvoiceDate': 'Recency', 
            'InvoiceNo': 'Frequency', 
            'TotalAmount': 'Monetary'
        })
        return rfm

    def find_optimal_k(self, scaled_data, max_k=10):
        """
        Calculates the optimal number of clusters using Point-to-Line vector distance.

        Args:
            scaled_data (np.ndarray): Standardized features for clustering.
            max_k (int): Maximum number of clusters to evaluate.

        Returns:
            tuple: (optimal_k, ks, inertias) for further plotting and modeling.
        """
        inertias = []
        ks = range(1, max_k + 1)
        for k in ks:
            kmeans = KMeans(
                n_clusters=k, 
                init='k-means++', 
                random_state=self.config.get('random_seed', 42), 
                n_init=10
            )
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Vector geometry to find the "Elbow" point
        p1, p2 = np.array([ks[0], inertias[0]]), np.array([ks[-1], inertias[-1]])
        distances = [
            np.abs(np.cross(p2 - p1, p1 - np.array([ks[i], inertias[i]]))) / np.linalg.norm(p2 - p1) 
            for i in range(len(ks))
        ]
        
        optimal_k = ks[np.argmax(distances)]
        return optimal_k, ks, inertias

    def plot_elbow_method(self, ks, inertias, optimal_k):
        """
        Generates and saves a visual representation of the Elbow Method.

        Args:
            ks (range): Range of clusters tested.
            inertias (list): Sum of squared distances for each K.
            optimal_k (int): The mathematically determined optimal K.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(ks, inertias, 'go--', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K: {optimal_k}')
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia (SSE)', fontsize=12)
        plt.title('Elbow Method: Identifying Optimal Customer Segments', fontsize=14)
        plt.legend()
        
        save_path = os.path.join(self.image_dir, 'elbow_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 
        plt.close()

    def plot_clusters(self, rfm):
        """
        Creates a log-scaled scatter plot of the identified customer segments.

        Args:
            rfm (pd.DataFrame): RFM data with an assigned 'Cluster' column.
        """
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=rfm, x='Frequency', y='Monetary', 
            hue='Cluster', palette='viridis', alpha=0.7, s=60
        )
        plt.title(f'Customer Segmentation Results (K={rfm["Cluster"].nunique()})', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Purchase Frequency (Log Scale)', fontsize=12)
        plt.ylabel('Monetary Value (Log Scale)', fontsize=12)
        
        save_path = os.path.join(self.image_dir, 'customer_clusters.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def run_germany_ttest(self, df):
        """
        Executes Welch's T-test to validate market revenue collapse.

        Args:
            df (pd.DataFrame): Raw transaction data.

        Returns:
            tuple: (T-statistic, P-value).
        """
        g_df = df[df['Country'] == 'Germany'].copy()
        g_24 = g_df[g_df['InvoiceDate'].dt.year == 2024]['TotalAmount']
        g_25 = g_df[g_df['InvoiceDate'].dt.year == 2025]['TotalAmount']
        return stats.ttest_ind(g_24, g_25, equal_var=False)

if __name__ == "__main__":
    # Initialize the analysis pipeline
    analyzer = RevenueAnalyzer()
    
    # Path to simulation data
    data_path = os.path.join(analyzer.data_dir, "ecommerce_data.csv")
    
    if os.path.exists(data_path):
        # Load and process data
        df = pd.read_csv(data_path, parse_dates=['InvoiceDate'])
        rfm = analyzer.perform_rfm_analysis(df)
        
        # Scaling and Segmentation
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # Determine and apply clustering
        best_k, ks, inertias = analyzer.find_optimal_k(scaled_features)
        kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Generate and save all forensic visualizations
        analyzer.plot_elbow_method(ks, inertias, best_k)
        analyzer.plot_clusters(rfm)
        
        # Statistical validation
        t_stat, p_val = analyzer.run_germany_ttest(df)
        print(f"Forensic Audit: Germany T-stat={t_stat:.2f}, P-val={p_val:.4f}")
        
        # Persistence
        rfm.to_csv(os.path.join(analyzer.data_dir, "final_customer_analytics.csv"))
        print(f"Analysis complete. 2 charts saved to {analyzer.image_dir}")
    else:
        print("Error: ecommerce_data.csv not found. Run data_loader.py first.")