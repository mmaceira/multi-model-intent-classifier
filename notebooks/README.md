# Reuters RAG Classifier Notebooks

This directory contains Jupyter notebooks that demonstrate the complete workflow of the Reuters RAG Classifier project, from data loading to model evaluation.

## Notebook Overview

The notebooks are designed to be run in sequence, as each builds upon the work of the previous ones:

1. **00_Build_Embeddings_llms.ipynb**
   - Purpose: Build and test embeddings and LLM components
   - Key tasks:
     - Initialize embedding models
     - Test embedding quality
     - Set up LLM connections
     - Validate model responses

2. **01_Data_Loading.ipynb**
   - Purpose: Load and preprocess the Reuters-21578 dataset
   - Key tasks:
     - Dataset loading
     - Text preprocessing
     - Data cleaning
     - Train/test split
     - Data validation

3. **02_Exploratory_Analysis.ipynb**
   - Purpose: Analyze the dataset and understand its characteristics
   - Key tasks:
     - Topic distribution analysis
     - Text length analysis
     - Word frequency analysis
     - Topic correlation analysis
     - Data visualization

4. **03_Model_Training.ipynb**
   - Purpose: Train and compare different classification models
   - Key tasks:
     - Feature engineering
     - Model training
     - Hyperparameter tuning
     - Model comparison
     - Performance metrics

5. **04_Model_Evaluation.ipynb**
   - Purpose: Evaluate model performance and analyze results
   - Key tasks:
     - Cross-validation
     - Error analysis
     - Confusion matrix
     - ROC curves
     - Feature importance


## Running the Notebooks

1. Ensure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Start Jupyter Lab or Notebook:
   ```bash
   jupyter lab  # or jupyter notebook
   ```

3. Run the notebooks in order, as each notebook depends on the outputs of previous ones.

## Notebook Dependencies

Each notebook requires specific Python packages. The main dependencies are:
- pandas
- numpy
- scikit-learn
- transformers
- faiss-cpu
- openai
- matplotlib
- seaborn
- jupyter

## Output Files

The notebooks generate various output files that are stored in the `output/` directory:
- `embeddings/`: Generated embeddings
- `models/`: Trained model files
- `results/`: Evaluation results
- `visualizations/`: Generated plots and charts

## Tips for Running

1. **Memory Management**
   - Some notebooks, especially those dealing with embeddings, require significant memory
   - Consider using a machine with at least 16GB RAM
   - Use batch processing for large datasets

2. **GPU Usage**
   - Notebooks 00 and 03 benefit from GPU acceleration
   - Set `CUDA_VISIBLE_DEVICES` if using multiple GPUs

3. **API Keys**
   - For OpenAI API usage, ensure your API key is set in the `.env` file
   - Keep your API keys secure and never commit them to version control

4. **Saving Progress**
   - Save intermediate results to avoid recomputing expensive operations
   - Use the provided utility functions for saving and loading

## Troubleshooting

Common issues and solutions:

1. **Memory Errors**
   - Reduce batch size
   - Use smaller models
   - Clear memory between cells

2. **API Rate Limits**
   - Implement rate limiting
   - Use caching
   - Consider using local models

3. **Model Loading Issues**
   - Check internet connection
   - Verify model paths
   - Clear cache and retry

## Contributing

When adding new notebooks:
1. Follow the naming convention: `XX_Descriptive_Name.ipynb`
2. Include clear markdown documentation
3. Add proper error handling
4. Include visualization where appropriate
5. Update this README with the new notebook's description 