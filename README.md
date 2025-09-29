# MCP Server - Gemini Image Generator- AUTO UPDATED
#CRONTAB IS WORKING 
A **Model Context Protocol (MCP) server** that enables AI-driven image generation using Google's **Gemini API**. This server provides seamless integration with MCP-compatible clients for generating high-quality images from text prompts.

## ✨ Features

- 🖼️ **Gemini-powered image generation** with advanced AI models
- 🔌 **MCP protocol compatibility** for easy integration
- 🛠️ **Configurable prompts** and generation parameters
- 📁 **Static file serving** for generated images
- 🧪 **Built-in testing** and validation tools
- 🔒 **Environment-based configuration** for secure API key management

## 📂 Project Structure

```
mcp-server-gemini-image-generator/
├── .venv/                                    # Virtual environment
├── examples/                                 # Usage examples
├── src/mcp_server_gemini_image_generator/
│   ├── __init__.py                          # Package initialization
│   ├── server.py                            # Main MCP server
│   ├── prompts.py                           # Prompt management
│   ├── utils.py                             # Utility functions
│   └── test.py                              # Test suite
├── static/                                  # Generated images & assets
├── .env                                     # Environment configuration
├── .gitignore                               # Git ignore rules
├── pyproject.toml                           # Project metadata & dependencies
├── uv.lock                                  # Dependency lock file
├── smithery.yaml                            # Deployment configuration
├── LICENSE                                  # MIT License
└── README.md                                # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Gemini API key** (get one from [Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/mcp-server-gemini-image-generator.git
   cd mcp-server-gemini-image-generator
   ```

2. **Set up virtual environment**
   ```bash
   # Using Python venv
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   
   # Or using uv (recommended)
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Using pip
   pip install -e .
   
   # Or using uv (faster)
   uv sync
   ```

4. **Configure environment**
   
   Create a `.env` file in the project root:
   ```env
   API_KEY=your_gemini_api_key_here
   PORT=8002
   HOST=127.0.0.1
   ```

5. **Start the server**
   ```bash
   python src/mcp_server_gemini_image_generator/server.py
   ```

The MCP server will be available at: `http://127.0.0.1:8002/mcp/`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_KEY` | Your Gemini API key | - | ✅ |
| `PORT` | Server port | `8002` | ❌ |
| `HOST` | Server host | `127.0.0.1` | ❌ |

### MCP Client Integration

Add this server to your MCP client configuration:

```json
{
  "mcpServers": {
    "gemini-image-generator": {
      "command": "python",
      "args": ["path/to/src/mcp_server_gemini_image_generator/server.py"],
      "env": {
        "API_KEY": "your_gemini_api_key"
      }
    }
  }
}
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python src/mcp_server_gemini_image_generator/test.py
```

## 📖 Usage Examples

### Basic Image Generation

```python
from mcp_server_gemini_image_generator import generate_image

# Generate an image from a text prompt
result = generate_image("A serene mountain landscape at sunset")
print(f"Generated image: {result['image_url']}")
```

### Custom Prompts

```python
# Use the prompt utilities for enhanced generation
from mcp_server_gemini_image_generator.prompts import enhance_prompt

enhanced = enhance_prompt("cat", style="photorealistic", mood="playful")
result = generate_image(enhanced)
```

## 🛠️ Development

### Project Setup for Development

```bash
# Clone and setup
git clone https://github.com/your-username/mcp-server-gemini-image-generator.git
cd mcp-server-gemini-image-generator

# Install in development mode
uv sync --dev

# Run tests
python -m pytest src/mcp_server_gemini_image_generator/test.py
```

### Adding New Features

1. Implement your feature in the appropriate module (`server.py`, `prompts.py`, or `utils.py`)
2. Add tests in `test.py`
3. Update documentation
4. Submit a pull request

## 📋 API Reference

### MCP Tools

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `generate_image` | Generate image from text prompt | `prompt: str`, `style?: str` |
| `enhance_prompt` | Improve prompt for better results | `prompt: str`, `options?: dict` |

### Response Format

```json
{
  "success": true,
  "image_url": "http://127.0.0.1:8002/static/generated_image_123.png",
  "prompt_used": "Enhanced prompt that was actually used",
  "generation_time": 2.34,
  "metadata": {
    "model": "gemini-pro-vision",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```


**Made with ❤️ for the MCP community**
