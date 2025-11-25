# Performance

This document covers performance metrics, resource requirements, and scaling guidance.

## Performance Metrics

### Model Performance on CLINC150

| Model | Architecture | Accuracy | Macro-F1 | Training Time | Inference Speed | Resource Usage |
|-------|--------------|----------|----------|---------------|-----------------|----------------|
| Naive Bayes | TF-IDF + Naive Bayes | - | - | ~2 min | 60k docs/s | < 2GB RAM |
| Linear SVM | TF-IDF + SVM | - | - | 7-8 min | 12-15k docs/s | < 5GB RAM |
| MiniLM + LogReg | Transformer + Logistic Regression | - | - | ~45 min | 1k docs/s | 12GB GPU |
| RAG-CentroidNN | FAISS + Nearest Neighbors | - | - | ~25 min | 200 QPS | 16GB RAM |
| RAG-LLM | FAISS + LLM (Ollama/OpenAI) | - | - | ~30 min | 150-180 QPS | 16GB RAM |

*Note: Actual accuracy and F1 scores depend on hyperparameters and dataset configuration. Run evaluations to get specific metrics for your setup.*

## Model Strengths

- **Naive Bayes**: Fastest inference, suitable for real-time applications
- **Linear SVM**: Best balance of speed and accuracy
- **MiniLM + LogReg**: Highest accuracy, best for precision-critical tasks
- **RAG Models**: Best for semantic understanding and context-aware classification

## Resource Requirements

| Model | CPU | RAM | GPU | Storage |
|-------|-----|-----|-----|---------|
| Naive Bayes | ✓ | 2GB | - | 500MB |
| Linear SVM | ✓ | 5GB | - | 1GB |
| MiniLM + LogReg | - | 8GB | 12GB | 2GB |
| RAG Models | ✓ | 16GB | - | 5GB |

**Note for Full Dataset Runs:**
- **RAM Peak**: ~16GB without RAG-LLM, ~24GB+ with RAG-LLM (8B model)
- **Disk**: ~5GB for embeddings and models (vectors are NOT stored in metadata JSONL)
- **CPU**: Multi-core recommended; BLAS threads should be limited to prevent oversubscription
- **RAG-LLM**: Disabled by default for full runs; enable only for small configs or use smaller models (0.5B-1B)

## Scaling Guidance

| Method | Data Scaling | Guidance |
|--------|-------------|----------|
| Naive Bayes/SVM | Linear cost | Train on full corpus |
| Transformer + LR | Diminishing returns after ~300k docs | Cap training set |
| RAG FAISS | Linear indexing | Index everything, cap generation |
| Cross-Encoder | Linear infer-time cost | Re-rank only top-k passages |

## Performance Optimization

### Batch Processing
Process documents in batches for higher throughput:
- Embeddings: Process in batches of 32-128
- Predictions: Use batch prediction methods
- RAG retrieval: Batch queries when possible

### Caching
Implement results caching for frequent queries:
- Cache embeddings for repeated documents
- Cache model predictions for identical inputs
- Use Redis or similar for distributed caching

### Quantization
Use quantized models for lower memory footprint:
- Quantize transformer models (8-bit or 4-bit)
- Use smaller embedding models when possible
- Consider model distillation for deployment

### Hybrid Approach
Use lightweight models for first-pass filtering:
1. Use Naive Bayes or Linear SVM for initial filtering
2. Apply RAG-LLM only to uncertain cases
3. Combine predictions with ensemble methods

## Inference Speed Optimization

### For Real-Time Applications
- Use Naive Bayes or Linear SVM
- Pre-compute embeddings for known documents
- Use FAISS for fast similarity search
- Consider model quantization

### For Batch Processing
- Process in large batches
- Use GPU acceleration when available
- Parallelize across multiple workers
- Use async processing for I/O-bound operations

## Memory Optimization

### Reduce Memory Usage
- Use smaller embedding models (e.g., MiniLM-L6 instead of larger models)
- Limit batch sizes during training
- Use gradient checkpointing for large models
- Clear caches between operations

### For Resource-Constrained Environments
- Use CPU-only models (Naive Bayes, Linear SVM)
- Use API-based embeddings (OpenAI) instead of local models
- Limit dataset size for development
- Use smaller RAG top_k values

## Training Time Optimization

### Speed Up Training
- Use smaller datasets for hyperparameter tuning
- Parallelize hyperparameter search
- Use early stopping when applicable
- Cache preprocessed data

### For Large Datasets
- Use incremental learning when possible
- Sample datasets for initial experiments
- Use distributed training for transformer models
- Consider transfer learning from pre-trained models

## Production Deployment Considerations

### Latency Requirements
- **Real-time (< 100ms)**: Use Naive Bayes or Linear SVM
- **Near real-time (< 1s)**: Use Linear SVM or RAG with small top_k
- **Batch processing**: Any model, optimize for throughput

### Throughput Requirements
- **High throughput**: Use Naive Bayes, batch processing, parallel workers
- **Moderate throughput**: Use Linear SVM, RAG with optimized FAISS
- **Low throughput**: Use transformer models, RAG-LLM

### Cost Optimization
- Use local models (Ollama) when possible
- Cache expensive operations (embeddings, LLM calls)
- Use smaller models for less critical tasks
- Monitor API usage for cloud-based models

## Benchmarking

To benchmark your setup:

```bash
# Run full pipeline and check execution times
python scripts/pipeline/run_all.py

# Check model execution times
cat output/{run_name}/models/*/execution_time.txt

# Profile specific operations
python -m cProfile -s cumulative scripts/pipeline/03_model_training.py
```

## Monitoring

### Key Metrics to Monitor
- **Training time**: Track per model and per dataset size
- **Inference latency**: P50, P95, P99 percentiles
- **Memory usage**: Peak and average during training/inference
- **API costs**: Track for cloud-based models
- **Accuracy metrics**: F1, accuracy, per-class performance

### Logging
The pipeline includes comprehensive logging:
- Training progress and timing
- Model performance metrics
- Resource usage statistics
- Error and warning messages

Check logs in:
- `output/{run_name}/models/*/execution_time.txt`
- Console output during pipeline execution
- Evaluation reports in `output/{run_name}/results/`
