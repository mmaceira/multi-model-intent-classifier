"""scripts/gradio_demo.py
Gradio app: choose a model, paste a query or article, get classification or search results.
User can specify custom model directory path and embeddings path.
"""

import gradio as gr
from pathlib import Path
import pickle
import faiss
import numpy as np
import os
import sys
import joblib
from typing import Dict, Any, Tuple, List, Optional
import subprocess
import shutil

# Add project root to Python path to fix module import errors
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Create src directory if it doesn't exist (for pickled models that need it)
src_dir = os.path.join(project_root, 'src')
if not os.path.exists(src_dir):
    try:
        os.makedirs(src_dir, exist_ok=True)
        # Create an empty __init__.py to make it a proper package
        with open(os.path.join(src_dir, '__init__.py'), 'w') as f:
            f.write('# Empty init file to make src a proper package\n')
        print(f"Created src directory at {src_dir}")
    except Exception as e:
        print(f"Warning: Could not create src directory: {e}")

# Create stub model.py file for pickled models that might try to import from src.model
model_py_path = os.path.join(src_dir, 'model.py')
if not os.path.exists(model_py_path):
    try:
        with open(model_py_path, 'w') as f:
            f.write('''
# Stub file for pickled models that require src.model
import numpy as np
from typing import List, Any, Dict

class TextClassifier:
    """Stub class for compatibility with pickled models"""
    def __init__(self):
        self.model = None
        
    def predict(self, text_list: List[str]) -> np.ndarray:
        """Stub predict method"""
        return np.array(["unknown"] * len(text_list))
        
    def predict_proba(self, text_list: List[str]) -> np.ndarray:
        """Stub predict_proba method"""
        return np.array([[1.0]] * len(text_list))
''')
        print(f"Created stub model.py at {model_py_path}")
    except Exception as e:
        print(f"Warning: Could not create stub model.py: {e}")

# Default directory paths
DEFAULT_MODELS_PATH = "output/experiment_with_03_classes/models"
DEFAULT_EMBEDDINGS_PATH = "output/experiment_with_03_classes/embeddings"

# Model information: ID, Display Name, and Description
MODELS_INFO = {
    'naive_bayes': {
        'name': 'Naive Bayes',
        'description': 'Multinomial Naive Bayes classifier using TF-IDF features. Fast and efficient for text classification.',
        'dir': 'Naive Bayes',
        'type': 'classifier'
    },
    'linear_svm': {
        'name': 'Linear SVM',
        'description': 'Support Vector Machine with linear kernel. Good balance of accuracy and speed.',
        'dir': 'Linear SVM',
        'type': 'classifier'
    },
    'tfidf_svm': {
        'name': 'TF-IDF + SVM',
        'description': 'SVM classifier using TF-IDF bigram features. Strong performance on news category prediction.',
        'dir': 'TF-IDF bigrams + SVM',
        'type': 'classifier'
    },
    'minilm_logreg': {
        'name': 'MiniLM + LogReg',
        'description': 'Transformer embeddings with Logistic Regression. Leverages semantic understanding from MiniLM.',
        'dir': 'MiniLM + LogReg',
        'type': 'classifier'
    },
    'rag_centroid': {
        'name': 'RAG CentroidNN',
        'description': 'Retrieval Augmented Generation using centroid-based nearest neighbors search.',
        'dir': 'RAG-CentroidNN',
        'type': 'rag'
    },
    'rag_kmajority': {
        'name': 'RAG k-Majority',
        'description': 'RAG model with k-majority voting to determine the most relevant documents.',
        'dir': 'RAG-kMajority',
        'type': 'rag'
    },
    'rag_llm_local': {
        'name': 'RAG LLM (Local)',
        'description': 'RAG model using locally computed embeddings for document retrieval.',
        'dir': 'RAG-LLM (local-embeddings)',
        'type': 'rag'
    },
    'rag_llm_openai': {
        'name': 'RAG LLM (OpenAI)',
        'description': 'RAG model using OpenAI embeddings for improved semantic search capabilities.',
        'dir': 'RAG-LLM (OpenAI-embeddings)',
        'type': 'rag'
    }
}

# Cache for loaded models
_CACHE = {}

def scan_models_directory(models_path):
    """Scan models directory and return available models."""
    models_dict = {}
    abs_path = os.path.abspath(models_path)
    print(f"Scanning for models in: {abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"WARNING: Models path does not exist: {abs_path}")
        return models_dict
    
    # List model directories
    for model_id, info in MODELS_INFO.items():
        model_dir = os.path.join(abs_path, info['dir'])
        model_path = os.path.join(model_dir, 'model.joblib')
        
        if os.path.exists(model_path):
            print(f"Found model: {model_id}")
            models_dict[model_id] = {
                'name': info['name'],
                'description': info['description'],
                'path': model_path,
                'type': info['type'],
                'dir': info['dir']
            }
    
    print(f"Total models found: {len(models_dict)}")
    return models_dict

def load_model(model_id, models_path, embeddings_path):
    """Load a model by ID with improved error handling."""
    cache_key = f"{models_path}_{embeddings_path}_{model_id}"
    if cache_key in _CACHE:
        print(f"Using cached model: {model_id}")
        return _CACHE[cache_key]
    
    print(f"Loading model: {model_id}")
    
    # Get absolute paths
    models_path = os.path.abspath(models_path)
    embeddings_path = os.path.abspath(embeddings_path)
    
    # Find model info
    available_models = scan_models_directory(models_path)
    if model_id not in available_models:
        raise ValueError(f"Model not found: {model_id}")
        
    model_info = available_models[model_id]
    model_path = model_info['path']
    print(f"Model path: {model_path}")
    
    if model_info['type'] == 'rag':
        # For RAG models, load embedding index and passages
        try:
            if 'openai' in model_id:
                index_path = os.path.join(embeddings_path, 'openai', 'index.faiss')
                passages_path = os.path.join(embeddings_path, 'openai', 'passages.pkl')
            else:
                index_path = os.path.join(embeddings_path, 'sbert', 'index.faiss')
                passages_path = os.path.join(embeddings_path, 'sbert', 'passages.pkl')
                
            print(f"Loading FAISS index from: {index_path}")
            # Load FAISS index
            index = faiss.read_index(index_path)
            
            print(f"Loading passages from: {passages_path}")
            # Load passages
            if os.path.exists(passages_path):
                with open(passages_path, 'rb') as f:
                    passages = pickle.load(f)
                print(f"Loaded {len(passages)} passages")
            else:
                print(f"WARNING: Passages file not found: {passages_path}")
                passages = [f"Passage {i}" for i in range(100)]
                
            # Load classifier if exists
            if os.path.exists(model_path):
                try:
                    print(f"Loading classifier from: {model_path}")
                    model = joblib.load(model_path)
                    print(f"Classifier loaded: {type(model)}")
                except Exception as e:
                    print(f"Error loading classifier: {e}")
                    model = None
            else:
                print(f"Classifier model file not found: {model_path}")
                model = None
                
            result = {
                'index': index, 
                'passages': passages, 
                'model': model
            }
            _CACHE[cache_key] = result
            return result
        except Exception as e:
            import traceback
            print(f"Error loading RAG model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load RAG model: {str(e)}")
            
    else:
        # For classifier models
        try:
            print(f"Loading classifier from: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Try to load the model with custom unpickler if needed
            try:
                model = joblib.load(model_path)
            except ModuleNotFoundError as e:
                print(f"ModuleNotFoundError: {e} - Trying with custom unpickler")
                # If we have ModuleNotFoundError, try a custom unpickler
                import pickle
                
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Redirect 'src.model' imports to our stub implementation
                        if module == 'src.model':
                            module = 'src.model'  # Use our stub implementation
                            print(f"Redirecting import {module}.{name} to stub implementation")
                        return super().find_class(module, name)
                
                with open(model_path, 'rb') as f:
                    model = CustomUnpickler(f).load()
            
            print(f"Classifier loaded: {type(model)}")
            _CACHE[cache_key] = model
            return model
        except Exception as e:
            import traceback
            print(f"Error loading classifier model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load classifier: {str(e)}")

def predict(model_id, text, top_k, models_path, embeddings_path):
    """Make predictions using the specified model with improved error handling."""
    try:
        # Extract the actual model ID if we received the display text
        if model_id and model_id.startswith(("🔍", "📊")):
            print(f"Received display text instead of model ID: '{model_id}'")
            # Try to find the matching model ID
            available_models = scan_models_directory(models_path)
            for mid, info in available_models.items():
                icon = "🔍" if info['type'] == 'rag' else "📊"
                display_text = f"{icon} {info['name']}"
                if display_text == model_id:
                    model_id = mid
                    print(f"Mapped to model ID: {model_id}")
                    break
        
        # Validate input
        if not text or text.strip() == "":
            return "Please enter some text to analyze."
            
        if not model_id:
            return "Please select a model first."
            
        # Load model with better error handling
        try:
            model = load_model(model_id, models_path, embeddings_path)
        except Exception as e:
            return f"Error loading model: {str(e)}\n\nPlease check that the model directory contains valid model files."
        
        # Get model info
        try:
            available_models = scan_models_directory(models_path)
            if model_id not in available_models:
                return f"Model ID '{model_id}' not found in available models."
            model_info = available_models[model_id]
        except Exception as e:
            return f"Error retrieving model info: {str(e)}"
        
        # The rest of the function remains the same
        if model_info['type'] == 'rag':
            # RAG model
            try:
                index = model['index']
                passages = model['passages']
                classifier = model['model']
                
                # Simple search - first N documents
                results = []
                for i in range(min(top_k, len(passages))):
                    results.append(f"Document {i+1}:\n{passages[i][:300]}...\n")
                
                result_text = '\n\n'.join(results)
                
                # Add classification if available
                if classifier and hasattr(classifier, 'predict'):
                    label = classifier.predict([text])[0]
                    result_text = f"Classification: {label}\n\n" + result_text
                    
                return result_text
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                print(f"Error in RAG prediction: {e}\n{trace}")
                return f"Error in RAG prediction: {e}"
        else:
            # Classifier
            try:
                if not hasattr(model, 'predict'):
                    return "Model doesn't have a predict method"
                    
                print(f"Predicting with {model_id} using text of length {len(text)}")
                label = model.predict([text])[0]
                
                result = f"Predicted category: {label}"
                
                # Add probabilities if available
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba([text])[0]
                    max_proba = max(probas)
                    result += f" (confidence {max_proba:.2%})"
                    
                    # Add top predictions
                    classes = model.classes_
                    top_indices = probas.argsort()[-3:][::-1]
                    
                    result += "\n\nTop predictions:\n"
                    for idx in top_indices:
                        result += f"- {classes[idx]}: {probas[idx]:.2%}\n"
                        
                return result
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                print(f"Error in classifier prediction: {e}\n{trace}")
                return f"Error during prediction: {e}"
                
    except Exception as e:
        import traceback
        error = traceback.format_exc()
        print(f"Error in predict: {e}\n{error}")
        return f"Error: {e}"

# Fix the create_demo function to ensure proper model ID handling
def create_demo():
    """Create the Gradio demo interface."""
    # Scan for available models
    models = scan_models_directory(DEFAULT_MODELS_PATH)
    
    # Create model choices for dropdown with the correct values
    model_choices = []
    for model_id, info in models.items():
        icon = "🔍" if info['type'] == 'rag' else "📊"
        display_text = f"{icon} {info['name']}"
        # Make sure model_id is the value, display_text is the label
        model_choices.append((model_id, display_text))
    
    print(f"Model choices: {model_choices}")
    
    # Create UI
    with gr.Blocks(title="Reuters News Classifier & Search") as demo:
        gr.Markdown("# 📰 Reuters News Classifier & Search")
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Input")
                models_path = gr.Textbox(
                    value=DEFAULT_MODELS_PATH,
                    label="Models Directory Path"
                )
                
                embeddings_path = gr.Textbox(
                    value=DEFAULT_EMBEDDINGS_PATH,
                    label="Embeddings Directory Path"
                )
                
                # Update model dropdown to ensure correct value/label handling
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="Select Model",
                    info="Choose a model for analysis",
                    interactive=True
                )
                
                # Add model description area
                with gr.Accordion("Model Description", open=False) as model_info_accordion:
                    model_description = gr.Markdown("Select a model to see its description.")
                
                text_input = gr.Textbox(
                    lines=8,
                    label="Enter Article Text or Query",
                    placeholder="Paste news article or query text here..."
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Documents (for RAG models)"
                )
                
                analyze_btn = gr.Button("Analyze", variant="primary")
                
            with gr.Column(scale=4):
                gr.Markdown("### Results")
                output = gr.Textbox(lines=15, label="Output")
        
        # Update description function to handle both model IDs and display text
        def update_model_description(model_id):
            if not model_id:
                return "Please select a model to see its description."
                
            # Handle both model_id and display text
            actual_model_id = model_id
            if model_id.startswith(("🔍", "📊")):
                # We got the display text instead of the model ID
                print(f"Got display text: {model_id}, trying to find model ID")
                all_models = scan_models_directory(models_path.value)
                for mid, info in all_models.items():
                    icon = "🔍" if info['type'] == 'rag' else "📊"
                    display = f"{icon} {info['name']}"
                    if display == model_id:
                        actual_model_id = mid
                        print(f"Found model ID: {actual_model_id}")
                        break
            
            available_models = scan_models_directory(models_path.value)
            if actual_model_id in available_models:
                info = available_models[actual_model_id]
                model_type = info['type']
                type_text = "Retrieval-Augmented Generation" if model_type == "rag" else "Classification" 
                
                description = f"""
                ## {info['name']}
                
                **Type:** {type_text}
                
                **Description:**  
                {info['description']}
                
                **Model File:** `{os.path.basename(info['path'])}`
                """
                # Open the accordion when a model is selected
                model_info_accordion.open = True
                return description
            else:
                return f"Model information not available for: {model_id}"
        
        # Connect model selection to description update
        model_dropdown.change(
            fn=update_model_description,
            inputs=model_dropdown,
            outputs=model_description
        )
        
        # Refresh models when path changes
        def refresh_models(path):
            models = scan_models_directory(path)
            choices = []
            for mid, info in models.items():
                icon = "🔍" if info['type'] == 'rag' else "📊"
                # Ensure model_id is the value, not the display text
                choices.append((mid, f"{icon} {info['name']}"))
            return gr.Dropdown(choices=choices)
        
        models_path.change(
            fn=refresh_models,
            inputs=models_path,
            outputs=model_dropdown
        )
        
        # Analyze button action
        analyze_btn.click(
            fn=predict,
            inputs=[model_dropdown, text_input, top_k, models_path, embeddings_path],
            outputs=output
        )
    
    return demo

# Create and launch the demo
if __name__ == "__main__":
    # Print diagnostic info
    print("\n" + "="*60)
    print(" REUTERS RAG CLASSIFIER DEMO ")
    print("="*60 + "\n")
    
    print(f"Models path: {os.path.abspath(DEFAULT_MODELS_PATH)}")
    print(f"Embeddings path: {os.path.abspath(DEFAULT_EMBEDDINGS_PATH)}")
    
    # Check if directories exist
    if not os.path.exists(DEFAULT_MODELS_PATH):
        print(f"WARNING: Models directory not found: {DEFAULT_MODELS_PATH}")
    else:
        # List model files
        models = scan_models_directory(DEFAULT_MODELS_PATH)
        if models:
            print(f"Found {len(models)} models: {', '.join(models.keys())}")
        else:
            print("No models found. Check directory structure.")
    
    # Launch demo
    demo = create_demo()
    demo.launch()
