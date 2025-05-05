"""
Model analyzer module for running comprehensive analysis on all trained models.
Provides functions to analyze models, generate visualizations and save results.
"""

from pathlib import Path
import os
import sys
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from src.datasets.dataset import get_dataset
from src.analysis_extra import (
    top_misclassifications,
    plot_learning_curves,
    plot_tsne,
    plot_calibration_curve,
    vocabulary_drift,
)
from config.notebook_setup import RESULTS_DIR

def analyze_all_models(config, models_dir=None, results_dir=None):
    """
    Run comprehensive analysis on all trained models.
    
    Args:
        config (dict): Configuration dictionary
        models_dir (str, optional): Directory containing model files
        results_dir (str, optional): Directory to save results
        
    Returns:
        dict: Analysis results paths
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Access config variables
    dataset_split_type = config['dataset']['split_type']
    n_classes = int(config['general']['n_classes'])
    cutoff_year = int(config['dataset']['cutoff_year'])
    run_name = config['general']['run_name']
    
    # Set up paths if not provided
    if models_dir is None:
        models_dir = os.path.join(repo_root, run_name, "models")
    if results_dir is None:
        results_dir = os.path.join(repo_root, run_name, "results")
    
    print(f"Run name: {run_name}")
    print(f"Models directory: {models_dir}")
    print(f"Results directory: {results_dir}")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load data
    print("\nLoading dataset...")
    X_train, y_train, X_test, y_test, classes = get_dataset(
        split_type=dataset_split_type,
        n_classes=n_classes,
        cutoff_year=cutoff_year
    )
    
    print(f"Loaded {len(X_train)} training documents with {n_classes} classes")
    
    # 2. Load all models
    models = {}
    model_dirs = []
    
    print(f"\nLooking for models in: {models_dir}")
    if os.path.exists(models_dir):
        # Get all subdirectories (model types)
        model_dirs = [d for d in Path(models_dir).iterdir() if d.is_dir()]
        print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")
    
    # Load each model
    if model_dirs:
        for model_dir in model_dirs:
            model_name = model_dir.name
            model_path = model_dir / "model.joblib"
            model_path_pkl = model_dir / "model.pkl"
            
            if model_path.exists():
                try:
                    models[model_name] = joblib.load(model_path)
                    print(f"Loaded model: {model_name} from {model_path}")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
            elif model_path_pkl.exists():
                try:
                    with open(model_path_pkl, 'rb') as f:
                        models[model_name] = pickle.load(f)
                    print(f"Loaded model: {model_name} from {model_path_pkl}")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
            else:
                print(f"No model file found in {model_dir}")
                
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
    
    # Create a TF-IDF vectorizer for visualization if no models are loaded
    tfidf_embeddings = None
    if len(models) == 0:
        print("\nNo models loaded. Creating TF-IDF embeddings for visualization...")
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_embeddings = vectorizer.fit_transform(X_train)
    
    # 3. Get embeddings for t-SNE visualization
    print("\nCreating embeddings for visualization...")
    try:
        embeddings = None
        if len(models) > 0:
            first_model = list(models.values())[0]
            if hasattr(first_model, 'named_steps') and 'vectorizer' in first_model.named_steps:
                vectorizer = first_model.named_steps['vectorizer']
                embeddings = vectorizer.transform(X_train)
                print("Created embeddings using model's vectorizer")
            elif hasattr(first_model, 'vectorizer'):
                vectorizer = first_model.vectorizer
                embeddings = vectorizer.transform(X_train)
                print("Created embeddings using model's vectorizer")
            else:
                # Use TF-IDF as fallback
                print("Model doesn't have a vectorizer, using TF-IDF")
                if tfidf_embeddings is None:
                    vectorizer = TfidfVectorizer(max_features=100)
                    embeddings = vectorizer.fit_transform(X_train)
                else:
                    embeddings = tfidf_embeddings
        else:
            # Use TF-IDF embeddings we created earlier
            embeddings = tfidf_embeddings
            print("Using TF-IDF embeddings (no models loaded)")
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        if tfidf_embeddings is None:
            vectorizer = TfidfVectorizer(max_features=100)
            embeddings = vectorizer.fit_transform(X_train)
        else:
            embeddings = tfidf_embeddings
    
    # 4. Run t-SNE visualization (once, independent of models)
    results = {"tsne": None, "vocab_drift": None, "models": {}}
    
    print("\nGenerating t-SNE visualization...")
    label_encoder = LabelEncoder()
    y_train_numeric = label_encoder.fit_transform(y_train)
    
    plt.figure(figsize=(10, 6))
    try:
        plot_tsne(embeddings, y_train_numeric)
        
        # Add a legend with text labels
        class_names = label_encoder.classes_
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.get_cmap('viridis')(i/len(class_names)), 
                            markersize=10) for i in range(len(class_names))]
        plt.legend(handles, class_names, title="Classes", loc="best")
        
        plt.title("t-SNE Visualization of Document Embeddings")
        plt.tight_layout()
        tsne_path = f"{RESULTS_DIR}/tsne_visualization.png"
        plt.savefig(tsne_path)
        plt.close()
        print(f"Saved t-SNE visualization to {tsne_path}")
        results["tsne"] = tsne_path
    except Exception as e:
        print(f"Error in t-SNE visualization: {e}")
    
    # 5. Run vocabulary drift analysis (independent of models)
    print("\nRunning vocabulary drift analysis...")
    try:
        drift_df = vocabulary_drift(X_train, X_test)
        print("\nVocabulary drift (top terms):\n")
        print(drift_df.head(10))
        
        # Save to CSV
        vocab_drift_path = f"{RESULTS_DIR}/vocabulary_drift.csv"
        drift_df.to_csv(vocab_drift_path, index=False)
        print(f"Saved vocabulary drift analysis to {vocab_drift_path}")
        results["vocab_drift"] = vocab_drift_path
    except Exception as e:
        print(f"Error in vocabulary drift analysis: {e}")
    
    # Helper function to fix models if needed
    def fix_model_if_needed(model, model_name):
        # Fix for RagSklearnAdapter where rag is None
        if hasattr(model, 'rag') and model.rag is None:
            print(f"Warning: {model_name}.rag is None. Creating dummy rag component.")
            # Create a placeholder for the missing rag component
            class DummyRag:
                def predict(self, X):
                    return ["dummy"] * len(X)
                def predict_proba(self, X):
                    return np.ones((len(X), len(np.unique(y_train))))
            model.rag = DummyRag()
            
        # Ensure the model has classes_ attribute
        if not hasattr(model, 'classes_') or model.classes_ is None:
            if hasattr(model, 'clf') and hasattr(model.clf, 'classes_'):
                model.classes_ = model.clf.classes_
                print(f"Set classes_ attribute for {model_name} from underlying classifier")
            else:
                # If clf doesn't have classes_, create from unique values in y_train
                model.classes_ = np.unique(y_train)
                print(f"Set classes_ attribute for {model_name} from training data")
        
        return model
    
    # 6. Analyze all models
    if len(models) > 0:
        for model_name, model in models.items():
            print(f"\n{'='*50}\nAnalyzing model: {model_name}\n{'='*50}")
            
            # Create a directory for this model's results
            model_result_dir = os.path.join(results_dir, model_name.replace("/", "_"))
            os.makedirs(model_result_dir, exist_ok=True)
            
            results["models"][model_name] = {
                "top_misclassifications": None,
                "learning_curves": None,
                "calibration_curve": None
            }
            
            # Fix model if needed
            try:
                model = fix_model_if_needed(model, model_name)
                
                # Top misclassifications
                print("\n1. Top misclassifications:")
                try:
                    df_mis = top_misclassifications(model, X_test, y_test, top_n=5)
                    print(df_mis)
                    
                    # Save to CSV
                    misclass_path = f"{model_result_dir}/top_misclassifications.csv"
                    df_mis.to_csv(misclass_path, index=False)
                    print(f"Saved top misclassifications to {misclass_path}")
                    results["models"][model_name]["top_misclassifications"] = misclass_path
                except Exception as e:
                    print(f"Error in top misclassifications: {e}")
                
                # Learning curves
                print("\n2. Learning curves:")
                try:
                    plt.figure(figsize=(10, 6))
                    plot_learning_curves(model, X_train, y_train, cv=3, train_sizes=np.linspace(0.3, 1.0, 5))
                    plt.title(f"Learning Curves - {model_name}")
                    plt.tight_layout()
                    learning_curves_path = f"{model_result_dir}/learning_curves.png"
                    plt.savefig(learning_curves_path)
                    plt.close()
                    print(f"Saved learning curves to {learning_curves_path}")
                    results["models"][model_name]["learning_curves"] = learning_curves_path
                except Exception as e:
                    print(f"Error in learning curves: {e}")
                
                # Calibration curve
                print("\n3. Calibration curve:")
                try:
                    plt.figure(figsize=(10, 6))
                    plot_calibration_curve(model, X_test, y_test)
                    plt.title(f"Calibration Curve - {model_name}")
                    plt.tight_layout()
                    calibration_path = f"{model_result_dir}/calibration_curve.png"
                    plt.savefig(calibration_path)
                    plt.close()
                    print(f"Saved calibration curve to {calibration_path}")
                    results["models"][model_name]["calibration_curve"] = calibration_path
                except Exception as e:
                    print(f"Error in calibration curve: {e}")
                    
            except Exception as e:
                print(f"Error analyzing model {model_name}: {e}")
    else:
        print("\nNo models to analyze. Only general analyses were performed.")
    
    print("\nAnalysis complete! Results saved to:", results_dir)
    return results 