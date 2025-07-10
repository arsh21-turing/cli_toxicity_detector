# Smart CLI Toxicity Detector – Python CLI Application

## Project Objective
A lightweight command-line interface tool that classifies sentences as Non-Toxic or Toxic and, for toxic content, identifies the specific category—insult, hate, obscene, threat, sexual, or self-harm. The system blends multilingual transformer embeddings with an optional Groq fallback for borderline cases, supports per-file scoring and adjustable thresholds, and culminates in a user-friendly Streamlit interface.

It supports multiple platforms and includes:
- Multilingual transformer embeddings for accurate toxicity detection
- Configurable sigmoid thresholds for fine-tuned classification
- Groq API fallback for ambiguous cases with deterministic tie-breaking
- Batch processing with detailed per-file scoring and reporting
- Threshold tuning and evaluation suite with performance metrics
- Interactive Streamlit interface with real-time threshold adjustment
- Comprehensive documentation and sample datasets

---
## Code Execution Screenshots
### Conversation 1: CLI Foundation and Configuration
Conversation 1 Execution -> https://drive.google.com/file/d/1_AbOgGJ2v2cUX9HDOYSyuSXCazmGDodn/view?usp=drive_link

Conversation 1 Execution -> https://drive.google.com/file/d/1oCvk5p3g0buzbBBoUFtSTqMkYgJgsPwP/view?usp=drive_link

Conversation 1 Execution -> https://drive.google.com/file/d/11qGC1jKlHeXGMU9pAeWPSAiLebjBwN10/view?usp=drive_link

### Conversation 2: Core Inference Engine
Conversation 2 Execution -> https://drive.google.com/file/d/1A1GUPW5SRCirrXgLZsWJ_dZLNjgaxz8q/view?usp=drive_link

Conversation 2 Execution -> https://drive.google.com/file/d/1ME-EcFODK2QHxNSEu_cIrc4r_vTLQGNG/view?usp=drive_link

Conversation 2 Execution -> https://drive.google.com/file/d/15GAZ8UL5w56jS3T2z7EN6kPY-s68XsoJ/view?usp=drive_link

Conversation 2 Execution -> https://drive.google.com/file/d/14cf6dlNyVGbw4OFcUdi7K5ZhnSidM85D/view?usp=drive_link

### Conversation 3: Taxonomy Mapping and Threshold Control
Conversation 3 Execution -> https://drive.google.com/file/d/1c1m1Qo4CjU8Lq4XceWujYTmM62puKpSn/view?usp=drive_link

Conversation 3 Execution -> https://drive.google.com/file/d/1awuQXpx6R7mtT17sWiSwG5TTNCtmqkJ2/view?usp=drive_link

Conversation 3 Execution -> https://drive.google.com/file/d/1hXWPkLnxysGQ4vorUzPiBuw2Ua5EwNZA/view?usp=drive_link

Conversation 3 Execution -> https://drive.google.com/file/d/1fjXpfH8F1ABoKzOLyEtTmT3_mCQpR13D/view?usp=drive_link

Conversation 3 Execution -> https://drive.google.com/file/d/1waq4c-kIXf6Jn2COB5TWb6HIFWxKXwJM/view?usp=drive_link

Conversation 3 Execution -> https://drive.google.com/file/d/1cyrQpQfOcpGH7PZgMhj1f45bQMVUmYCx/view?usp=drive_link

### Conversation 4: Groq Fallback for Ambiguous Sentences
Conversation 4 Execution -> https://drive.google.com/file/d/1WxAjFadLMA963BZLRNVEyPcb3f8ddKZw/view?usp=drive_link

Conversation 4 Execution -> https://drive.google.com/file/d/1sK6yk9ZpdQcxjDWCl2SpTjT9Wl8lnpV1/view?usp=drive_link

Conversation 4 Execution -> https://drive.google.com/file/d/1Twc8n1CMZhQH0P1RT33CFGdLrlJRRxrQ/view?usp=drive_link

Conversation 4 Execution -> https://drive.google.com/file/d/16hU7BEf2gUf10nu-jF_WZGfzrMjxQIaG/view?usp=drive_link

### Conversation 5: Batch Processing and File-Level Scoring
Conversation 5 Execution -> https://drive.google.com/file/d/1NNkpDZI0ejsHgcZZOJqXd6ac12dY0sfg/view?usp=drive_link

Conversation 5 Execution -> https://drive.google.com/file/d/1_hGXGDjDPvkG6vZ5QCIpNgY8pFhWvbpk/view?usp=drive_link

Conversation 5 Execution -> https://drive.google.com/file/d/1FFkwFh2FvTzlsaEvbVIYW_9VgIkGxfRl/view?usp=drive_link

Conversation 5 Execution -> https://drive.google.com/file/d/1sgjFFxzqWU-kmB-gCsCubs-ypD3H02lh/view?usp=drive_link

Conversation 5 Execution -> https://drive.google.com/file/d/1gPXrIY6pn2hRmpjZgM3kFUBMDBFiX5DN/view?usp=drive_link

Conversation 5 Execution -> https://drive.google.com/file/d/1QDizCpqTFN4D9PKFqG_E1_JSCgn2Cbbl/view?usp=drive_link


### Conversation 6: Threshold Tuning and Evaluation Suite
Conversation 6 Execution -> https://drive.google.com/file/d/1fTXOswYeUkmng5ZXkeOithoSUaLXjHTu/view?usp=drive_link

Conversation 6 Execution -> https://drive.google.com/file/d/10y8NlUsdkbS5xkhAyEiwNJArXUiyHJ6W/view?usp=drive_link

Conversation 6 Execution -> https://drive.google.com/file/d/1-M6WS8c0G0Rcr27rInfycl0p7TJloKDa/view?usp=drive_link

Conversation 6 Execution -> https://drive.google.com/file/d/1nREX3qi11F83MJ5utENNlEylnhdUSkyN/view?usp=drive_link

### Conversation 7: Streamlit Interactive Interface
Conversation 7 Execution -> https://drive.google.com/file/d/1zczCtCE9sHjmHybFBn78WJ9aWSs2S_bZ/view?usp=drive_link

### Conversation 8: Advanced Features and System Integration
Conversation 8 Execution -> https://drive.google.com/file/d/1scKDKn5jLUzS4J6NmLRQ72BjOtgyjufO/view?usp=drive_link

Conversation 8 Execution -> https://drive.google.com/file/d/1qILaBaQJSp6_Ms_KlorUSuQE4yjjeGqP/view?usp=drive_link

Conversation 8 Execution -> https://drive.google.com/file/d/1R9UlF4BHTj86e7QeYmQWXHv71IzPhPUk/view?usp=drive_link

Conversation 8 Execution -> https://drive.google.com/file/d/1PuVrqfNBr4xgoUaW0xFP-pa1hzVyFl69/view?usp=drive_link

---
## Project Features Mapped to Components
- **CLI Interface**: Command-line argument parsing, single sentence and file processing
- **Inference Engine**: Multilingual transformer model with probability scoring
- **Taxonomy System**: Category mapping (insult, hate, obscene, threat, sexual, self-harm)
- **Threshold Control**: Configurable sigmoid thresholds with CLI overrides
- **Groq Integration**: API fallback for ambiguous cases with deterministic tie-breaking
- **Batch Processing**: Large file handling with progress tracking and detailed reporting
- **Evaluation Suite**: Threshold tuning with precision, recall, and F1 metrics
- **Streamlit Interface**: Interactive web interface with real-time threshold adjustment

---
## Project Features Mapped to Conversations
- **Conversation 1**: CLI Foundation and Configuration - Create the initial command-line interface with argument parsing and basic settings system. Implement a detect command that accepts either a single sentence or a .txt file, treats each line as a sentence, and returns a placeholder Non-Toxic classification with colored terminal feedback, confirming that the overall interface, help text, and configuration workflow operate correctly. Implement a basic Sentence Transformers based model handler and config manager as well.
- **Conversation 2**: Core Inference Engine - Load a compact toxicity model, add text-preprocessing steps for consistent inputs, and build an inference pipeline that produces raw probability scores for every native label. Connect this engine to the CLI so users can request verbose output displaying full probability maps, establishing the computational backbone for later stages.
- **Conversation 3**: Taxonomy Mapping and Threshold Control - Define the public taxonomy of insult, hate, obscene, threat, sexual, self-harm, and Non-Toxic, map model outputs to these categories, and introduce configurable sigmoid thresholds stored in the settings file and overridable on the command line. Compute the top label and overall toxicity flag, update CLI coloring, and allow users to fine-tune thresholds to match their tolerance levels.
- **Conversation 4**: Groq Fallback for Ambiguous Sentences - Identify gray-zone sentences whose highest model probability sits within a user-defined confidence band, forward these sentences to Groq for a second opinion in the same taxonomy, and merge local and Groq results with a deterministic tie-breaking policy while logging any disagreements. Add a CLI switch to enable or disable this fallback, giving users full control over external calls.
- **Conversation 5**: Batch Processing and File-Level Scoring - Introduce a batch command that ingests large .txt or delimited files, processes sentences in manageable chunks, outputs a detailed report listing per-sentence classifications and probabilities, records Groq overrides where used, and computes an overall toxicity score for the entire file while displaying progress bars and summary statistics in the terminal.
- **Conversation 6**: Threshold Tuning and Evaluation Suite - Add an evaluation workflow that accepts a labeled validation set, sweeps threshold values across reasonable ranges, and reports precision, recall, and F1 for each category. Suggest new threshold settings that maximize macro-F1, allow users to save these settings back to the configuration file, and generate performance curves to visualize the impact of different threshold choices.
- **Conversation 7**: Streamlit Interactive Interface - Develop a Streamlit application featuring a single-sentence text box that displays an immediate toxicity verdict with colored badges, a sidebar housing sliders for real-time threshold adjustment and a toggle for Groq fallback, and a file-upload section that reuses the batch engine to score .txt or CSV inputs, visualize category distributions, and provide a downloadable report while preserving user threshold preferences across sessions.
- **Conversation 8**: Advanced Features and System Integration - Implement comprehensive system enhancements including real-time metrics dashboard with live throughput and memory usage monitoring, system health tracking with CPU and disk I/O metrics, Groq API key management with secure input and test connection functionality, performance tracking with accuracy metrics and confidence calibration, collapsible sidebar sections for improved navigation, comprehensive export functionality with multi-format reports, color-coded toxicity levels throughout the app, and robust data serialization handling for JSON export compatibility.

---

### Usage:
Run the main application:
```bash
# Basic CLI usage
python main.py --text "Hello world" --confidence-filter 0.8

# Batch file processing
python main.py --batch test_sample.txt --verbose

# Stream processing mode
echo -e "Hello world\nThis is great" | python main.py --stream --confidence-filter 0.999 --no-color

# Model evaluation
python main.py --evaluate example_validation_data.csv --optimize-thresholds --generate-pdf

# Web interface
streamlit run streamlit_app.py --server.port 8503
```

### Command Line Features
- `--text` - Analyze single text input
- `--file` - Process text file line-by-line
- `--batch` - Batch process files or directories
- `--stream` - Real-time streaming analysis from stdin
- `--evaluate` - Model evaluation on validation datasets
- `--confidence-filter` - Filter results by confidence threshold
- `--allow-groq-fallback` - Enable Groq API fallback for uncertain predictions
- `--json` - Output results in JSON format
- `--verbose` - Show detailed analysis information

# Project Setup

A lightweight command-line interface tool that classifies sentences as Non-Toxic or Toxic and identifies specific toxicity categories using multilingual transformer embeddings with optional Groq fallback.

## Getting Started

### Clone the Repository
```bash
# Clone the repository
git clone https://github.com/arsh21-turing/cli_toxicity_detector.git
```

### Setting Up the Environment

1. Create a Virtual Environment
```bash
# For Python 3 (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

2. Install Requirements
```bash
# Install all dependencies
pip install -r requirements.txt
```

3. Set up API Keys (Optional)
Create a .env file in the project root or set environment variables:

```bash
# Add your Groq API key to .env file (optional)
echo "GROQ_API_KEY=your_groq_api_key" > .env
```

## Running the Application

### Command Line Interface (CLI)

#### Basic Text Analysis
```bash
# Analyze a single text
python main.py --text "Hello world" --confidence-filter 0.8

# Analyze with detailed probabilities
python main.py --text "This is a test sentence." --probabilities --quiet
```

#### File Processing
```bash
# Process a text file line by line
python main.py --file test_sample.txt --verbose

# Batch process multiple files
python main.py --batch test_sample.txt --json --output results.json
```

#### Stream Processing
```bash
# Real-time streaming analysis
echo -e "Hello world\nThis is great" | python main.py --stream --confidence-filter 0.999 --no-color

# Interactive streaming mode
python main.py --stream --confidence-explain
```

#### Model Evaluation
```bash
# Evaluate model on validation dataset
python main.py --evaluate example_validation_data.csv --optimize-thresholds --generate-pdf

# Compare multiple models
python main.py --compare-models --model-paths "model1,model2" --evaluate validation_data.csv
```

### Web Interface (Streamlit)

#### Launch the Web App
```bash
# Run the Streamlit web interface
streamlit run streamlit_app.py --server.port 8503

# Access the web interface at http://localhost:8503
```

#### Web App Features
- **Single Text Analysis**: Real-time toxicity analysis with interactive visualizations
- **Batch File Upload**: Upload TXT or CSV files for comprehensive analysis
- **Interactive Dashboard**: Real-time metrics, performance tracking, and system health monitoring
- **Export Functionality**: Download results in JSON, CSV, PDF, or HTML formats
- **Configuration Management**: Adjustable thresholds and profile management
- **Groq API Integration**: Enhanced accuracy with API fallback options

## Key Features

### CLI Features
- **Real-time Analysis**: Instant toxicity detection with confidence scoring
- **Batch Processing**: Efficient processing of large files with progress tracking
- **Stream Mode**: Continuous analysis of text input streams
- **Confidence Filtering**: Filter results based on confidence thresholds
- **Multiple Output Formats**: JSON, colored text, and detailed probability outputs
- **Model Evaluation**: Comprehensive evaluation with threshold optimization
- **Groq Integration**: API fallback for enhanced accuracy on uncertain predictions

### Web Interface Features
- **Interactive Analysis**: Real-time text analysis with immediate feedback
- **File Upload**: Support for TXT and CSV files with automatic processing
- **Visual Analytics**: Interactive charts and graphs for result visualization
- **Performance Monitoring**: Real-time metrics dashboard and system health tracking
- **Export System**: Multi-format export with detailed reporting
- **Configuration Wizard**: Easy setup and customization of analysis parameters
- **Profile Management**: Save and load different analysis configurations

### Advanced Features
- **Threshold Optimization**: Automatic optimization of detection thresholds
- **Model Comparison**: Compare multiple models on the same dataset
- **Performance Tracking**: Monitor processing speed and resource usage
- **System Health**: Real-time monitoring of CPU, memory, and disk usage
- **Comprehensive Logging**: Detailed logging with filtering and export options
- **Error Handling**: Robust error handling with graceful degradation

## Configuration

### Configuration Wizard
```bash
# Launch interactive configuration wizard
python main.py --setup-config

# Use specific use case profile
python main.py --setup-config --wizard-defaults high_precision
```

### Custom Configuration
Create a `config.yaml` file:
```yaml
model:
  name: "unitary/toxic-bert"
  threshold: 0.6

categories:
  toxicity: 0.5
  severe_toxicity: 0.3
  identity_attack: 0.4
  insult: 0.5
  obscene: 0.4
  threat: 0.3
  sexual_explicit: 0.4

groq:
  enabled: false
  api_key: ""
  tie_policy: "prefer-groq"
  lower_bound: 0.4
  upper_bound: 0.6
```

## Troubleshooting

### Common Issues

**Model Loading Errors:**
- Ensure you have sufficient disk space for model downloads
- Check internet connection for model downloads
- Try clearing model cache: `rm -rf ~/.cache/huggingface/`

**Memory Issues:**
- Reduce batch size in configuration
- Use smaller model variants
- Enable GPU acceleration if available

**API Integration Issues:**
- Verify Groq API key is correctly set
- Check API rate limits and quotas
- Ensure network connectivity for API calls

**Web Interface Issues:**
- Clear browser cache and cookies
- Check if port 8503 is available
- Restart Streamlit server if interface becomes unresponsive

### Performance Optimization

**For Large Files:**
- Use batch processing mode
- Increase batch size in configuration
- Enable parallel processing where available

**For Real-time Analysis:**
- Use stream mode for continuous input
- Enable confidence filtering to reduce noise
- Use Groq API for enhanced accuracy
