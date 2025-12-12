# LLM Providers

The project uses `litellm` which supports multiple LLM providers. By default, it uses Ollama for local, cost-free inference, but it's very easy to switch to OpenAI or other providers.

## Overview

The RAG-LLM model supports multiple LLM providers through the `litellm` library. This allows you to:
- Use local models (Ollama) for cost-free, private inference
- Use cloud APIs (OpenAI, Anthropic) for high-quality results
- Switch between providers easily via configuration

## Using Ollama (Default)

Ollama is the default provider, offering local, cost-free inference.

### Setup

#### 1. Install Ollama

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# On Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows: Download installer from https://ollama.ai/download
```

#### 2. Start Ollama Server

```bash
# Start Ollama server (usually runs automatically, but you can start it manually)
ollama serve
```

#### 3. Pull the LLM Model

```bash
# Pull the default model used by RAG-LLM
ollama pull llama3.1:8b

# Verify the model is available
ollama list
```

#### 4. Build Embeddings (Required for RAG Models)

**Important**: Embeddings are separate from LLM models and need to be built separately:

```bash
# Run the embedding build script (this downloads SBERT model automatically)
python scripts/pipeline/02_build_embeddings.py
```

**Note about embeddings**:
- **SBERT embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) are downloaded automatically from HuggingFace when you first run the embedding script
- No manual download needed - the model downloads automatically (~80MB)
- Embeddings are built from your dataset and stored locally in `output/{run_name}/embeddings/`
- This is a one-time process per experiment

### Configuration

Default configuration in config files (e.g., `config/dataset/clinc150/tiny.yaml`):
```yaml
model:
  llm_model: "ollama/llama3.1:8b"  # Default LLM for RAG-LLM models
```

No API key needed! Ollama runs locally, and SBERT embeddings download automatically.

#### 5. Verify Installation

```bash
# Check Ollama is running and model is available
ollama list

# Test Ollama with a simple query
ollama run llama3.1:8b "Hello, world!"

# Verify embeddings will be built (run the pipeline)
python scripts/pipeline/02_build_embeddings.py --help
```

## Using OpenAI Models

To use OpenAI models instead of Ollama, you need to set up an API key and update your configuration.

### Setup

1. **Get OpenAI API Key**: Sign up at https://platform.openai.com and get your API key

2. **Set Environment Variable**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Update Configuration**:

   **Option 1: Update `config/models_config.yaml`**
   ```yaml
   rag_llm:
     enabled: true
     name: "RAG-LLM"
     class: "RagSklearnAdapter"
     params:
       method: "llm"
       top_k: 25
       model: "gpt-4o-mini"  # Change from "${model.llm_model}" to OpenAI model
       use_openai: false  # Set to true to use OpenAI embeddings instead of SBERT
   ```

   **Option 2: Update your main config file** (e.g., `config/dataset/clinc150/tiny.yaml`)
   ```yaml
   model:
     llm_model: "gpt-4o-mini"  # Change from "ollama/llama3.1:8b" to OpenAI model
   ```

   Then in `config/models_config.yaml`, set:
   ```yaml
   rag_llm:
     params:
       model: "${model.llm_model}"  # Will use the OpenAI model from the main config
       use_openai: false  # Set to true for OpenAI embeddings, false for SBERT
   ```

## Supported Models

The project supports any model that `litellm` supports, including:

### Ollama Models
- `ollama/llama3.1:8b` (default)
- `ollama/llama3.2:3b`
- `ollama/mistral`
- `ollama/codellama`
- And many more - see [Ollama library](https://ollama.ai/library)

### OpenAI Models
- `gpt-4o` - Latest GPT-4 model
- `gpt-4o-mini` - Cost-effective GPT-4 variant
- `gpt-4-turbo` - Previous GPT-4 version
- `gpt-3.5-turbo` - Fast and cost-effective

### Anthropic Models
- `claude-3-5-sonnet` - Latest Claude model
- `claude-3-opus` - Most capable Claude model
- `claude-3-haiku` - Fastest Claude model

### Other Providers
See [litellm documentation](https://docs.litellm.ai/) for the full list of supported providers.

## Configuration Tips

### For Local Development
- Use Ollama (default) - no API keys needed
- Fast iteration without API costs
- Privacy-friendly (data stays local)

### For Production
- Consider OpenAI's `gpt-4o-mini` for cost-effective, high-quality results
- Use `gpt-4o` for best accuracy
- Monitor API usage and costs

### For Best Accuracy
- Use `gpt-4o` or `claude-3-5-sonnet`
- These models provide the highest quality classifications

### For Speed
- Use `gpt-4o-mini` or `gpt-3.5-turbo`
- Or use Ollama with smaller models like `llama3.2:3b`

## Embedding Backend Selection

RAG models can use different embedding backends:

### SBERT (Default)
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- No API key needed
- Fast and privacy-friendly

### OpenAI Embeddings
- API-based embeddings using `text-embedding-3-small`
- Requires `OPENAI_API_KEY`
- Set `use_openai: true` in model configuration

**Example Configuration**:
```yaml
rag_llm:
  params:
    method: "llm"
    top_k: 25
    model: "gpt-4o-mini"  # LLM provider
    use_openai: true      # Use OpenAI embeddings (requires OPENAI_API_KEY)
```

## Switching Between Providers

### Quick Switch Example

1. **From Ollama to OpenAI**:
   ```bash
   export OPENAI_API_KEY="your-key"
   # Update your main config file: llm_model: "gpt-4o-mini"
   python scripts/pipeline/run_all.py
   ```

2. **From OpenAI to Ollama**:
   ```bash
   # Update config/config.yaml: llm_model: "ollama/llama3.1:8b"
   # Unset OPENAI_API_KEY if you want to ensure local-only
   python scripts/pipeline/run_all.py
   ```

## Cost Considerations

### Ollama
- **Cost**: Free (runs locally)
- **Limitations**: Requires local compute resources
- **Best for**: Development, privacy-sensitive applications

### OpenAI
- **Cost**: Pay per API call
- **Pricing**: Varies by model (see OpenAI pricing page)
- **Best for**: Production, when you need highest quality

### Recommendations
- **Development**: Use Ollama (free, local)
- **Production**: Use `gpt-4o-mini` (cost-effective, high quality)
- **High-stakes**: Use `gpt-4o` or `claude-3-5-sonnet` (best accuracy)

## Troubleshooting

### Ollama Issues
- **Model not found**: Run `ollama pull <model-name>`
- **Connection refused**: Make sure `ollama serve` is running
- **Slow inference**: Try a smaller model like `llama3.2:3b`

### Embedding Issues
- **SBERT model download fails**: Check internet connection - the model downloads from HuggingFace automatically
- **Embeddings not found**: Run `python scripts/pipeline/02_build_embeddings.py` to build them
- **Slow embedding generation**: This is normal for large datasets - embeddings are cached after first build
- **Out of memory**: SBERT model (~80MB) should fit in most systems; if issues persist, use a smaller batch size: `SBERT_BATCH=16 python scripts/pipeline/02_build_embeddings.py`

### OpenAI Issues
- **API key not found**: Set `OPENAI_API_KEY` environment variable
- **Rate limits**: Implement retry logic or use a different model
- **Cost concerns**: Use `gpt-4o-mini` instead of `gpt-4o`

## Best Practices

1. **Use Ollama for Development**: Fast iteration without costs
2. **Test with Multiple Providers**: Compare results before production
3. **Monitor API Usage**: Track costs when using cloud APIs
4. **Cache Embeddings**: Reuse embeddings when possible to reduce API calls
5. **Fallback Strategy**: Consider having a fallback to Ollama if API fails
