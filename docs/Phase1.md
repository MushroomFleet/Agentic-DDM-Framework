# Phase 1: Foundation & Environment Setup

## Phase Overview

**Goal:** Establish a production-ready project foundation with proper dependency management, configuration, logging, and directory structure  
**Prerequisites:** 
- Python 3.10 or higher installed
- Access to OpenRouter API (free tier available at https://openrouter.ai)
- Terminal/command line access
- Text editor or IDE (VS Code, PyCharm recommended)

**Estimated Duration:** 2-3 hours  

**Key Deliverables:**
- ✅ Virtual environment with all dependencies
- ✅ Project directory structure
- ✅ Configuration management system
- ✅ Environment variable handling
- ✅ Logging infrastructure
- ✅ API key validation
- ✅ Basic smoke tests

---

## Step-by-Step Implementation

### Step 1: Create Project Directory Structure

**Purpose:** Organize code, tests, documentation, and outputs systematically  
**Duration:** 5 minutes

#### Instructions

1. Create the main project directory:
```bash
mkdir addm-framework
cd addm-framework
```

2. Create the complete directory structure:
```bash
# Core source code
mkdir -p src/addm_framework/{models,ddm,llm,agent,utils}

# Test directories
mkdir -p tests/{unit,integration,fixtures}

# Documentation
mkdir -p docs/{phases,guides,api}

# Configuration and data
mkdir -p config
mkdir -p data/{inputs,outputs,visualizations}

# Deployment
mkdir -p deployment/{docker,scripts}

# Create __init__.py files for Python packages
touch src/addm_framework/__init__.py
touch src/addm_framework/models/__init__.py
touch src/addm_framework/ddm/__init__.py
touch src/addm_framework/llm/__init__.py
touch src/addm_framework/agent/__init__.py
touch src/addm_framework/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
```

3. Verify the structure:
```bash
tree -L 3
```

#### Expected Directory Structure
```
addm-framework/
├── src/
│   └── addm_framework/
│       ├── __init__.py
│       ├── models/          # Pydantic data models
│       │   └── __init__.py
│       ├── ddm/             # DDM simulation engine
│       │   └── __init__.py
│       ├── llm/             # LLM client layer
│       │   └── __init__.py
│       ├── agent/           # Agent orchestration
│       │   └── __init__.py
│       └── utils/           # Helper functions
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── unit/                # Unit tests
│   │   └── __init__.py
│   ├── integration/         # Integration tests
│   │   └── __init__.py
│   └── fixtures/            # Test data
├── docs/
│   ├── phases/              # Phase documents
│   ├── guides/              # User guides
│   └── api/                 # API documentation
├── config/                  # Configuration files
├── data/
│   ├── inputs/              # Input data
│   ├── outputs/             # Generated outputs
│   └── visualizations/      # DDM plots
└── deployment/
    ├── docker/              # Dockerfiles
    └── scripts/             # Deployment scripts
```

#### Verification
- [ ] All directories created
- [ ] All `__init__.py` files present
- [ ] `tree` command shows correct structure

---

### Step 2: Set Up Virtual Environment

**Purpose:** Isolate project dependencies from system Python  
**Duration:** 5 minutes

#### Instructions

1. Create virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

3. Verify activation (prompt should show `(venv)`):
```bash
which python  # Linux/macOS
where python  # Windows
```

4. Upgrade pip to latest version:
```bash
pip install --upgrade pip
```

#### Verification
- [ ] Virtual environment activated (prompt shows `(venv)`)
- [ ] `which python` points to `venv/bin/python`
- [ ] `pip --version` shows latest version (23.0+)

#### Troubleshooting

**Issue:** `python3: command not found`
- **Solution:** Install Python 3.10+ from https://python.org or use system package manager
  ```bash
  # Ubuntu/Debian
  sudo apt update && sudo apt install python3.10 python3.10-venv
  
  # macOS (with Homebrew)
  brew install python@3.10
  ```

**Issue:** Virtual environment won't activate
- **Solution:** Check permissions, ensure you're in the correct directory
  ```bash
  ls -la venv/bin/activate  # Should exist and be executable
  ```

---

### Step 3: Install Dependencies

**Purpose:** Install all required Python packages  
**Duration:** 5-10 minutes

#### Instructions

1. Create `requirements.txt`:
```bash
cat > requirements.txt << 'EOF'
# Core dependencies
numpy==1.26.4
matplotlib==3.8.3
pydantic==2.6.3
requests==2.31.0
aiohttp==3.9.3
retry==0.9.2

# Development dependencies
pytest==8.1.1
pytest-cov==4.1.0
pytest-asyncio==0.23.5
black==24.2.0
mypy==1.8.0
pylint==3.1.0

# Optional: API server
fastapi==0.110.0
uvicorn[standard]==0.27.1

# Optional: Enhanced logging
python-json-logger==2.0.7
EOF
```

2. Install all dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installations:
```bash
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"
python -c "import requests; print(f'Requests: {requests.__version__}')"
python -c "import aiohttp; print(f'aiohttp: {aiohttp.__version__}')"
```

#### Expected Output
```
NumPy: 1.26.4
Pydantic: 2.6.3
Requests: 2.31.0
aiohttp: 3.9.3
```

#### Verification
- [ ] All packages installed without errors
- [ ] Import tests pass for core dependencies
- [ ] `pip list` shows all required packages

#### Troubleshooting

**Issue:** `pip install` fails with compilation errors
- **Solution:** Install system dependencies (C compiler, development headers)
  ```bash
  # Ubuntu/Debian
  sudo apt install build-essential python3-dev
  
  # macOS
  xcode-select --install
  ```

**Issue:** Conflicting dependencies
- **Solution:** Clear pip cache and reinstall
  ```bash
  pip cache purge
  pip install --no-cache-dir -r requirements.txt
  ```

---

### Step 4: Configuration Management System

**Purpose:** Centralize configuration with environment variables and config files  
**Duration:** 15 minutes

#### Instructions

1. Create configuration module:
```bash
cat > src/addm_framework/utils/config.py << 'EOF'
"""Configuration management for ADDM Framework."""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class DDMConfig:
    """DDM simulation hyperparameters."""
    base_drift: float = 1.0
    threshold: float = 1.0
    noise_sigma: float = 1.0
    dt: float = 0.01
    non_decision_time: float = 0.2
    starting_bias: float = 0.0
    max_time: float = 5.0
    n_trials: int = 100


@dataclass
class APIConfig:
    """API client configuration."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "x-ai/grok-4-fast"
    timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your key at https://openrouter.ai"
            )
        
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", cls.base_url),
            model=os.getenv("OPENROUTER_MODEL", cls.model),
            timeout=int(os.getenv("API_TIMEOUT", cls.timeout)),
            max_retries=int(os.getenv("API_MAX_RETRIES", cls.max_retries))
        )


@dataclass
class AppConfig:
    """Application-wide configuration."""
    project_root: Path
    data_dir: Path
    output_dir: Path
    log_dir: Path
    enable_traces: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load application configuration."""
        project_root = Path(__file__).parent.parent.parent.parent
        
        return cls(
            project_root=project_root,
            data_dir=project_root / "data",
            output_dir=project_root / "data" / "outputs",
            log_dir=project_root / "logs",
            enable_traces=os.getenv("ENABLE_TRACES", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self):
        self.app = AppConfig.from_env()
        self.api = APIConfig.from_env()
        self.ddm = DDMConfig()
        
        # Ensure directories exist
        self.app.data_dir.mkdir(parents=True, exist_ok=True)
        self.app.output_dir.mkdir(parents=True, exist_ok=True)
        self.app.log_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration."""
        checks = [
            (self.api.api_key, "API key is set"),
            (self.app.project_root.exists(), "Project root exists"),
            (self.app.data_dir.exists(), "Data directory exists"),
            (0 < self.ddm.threshold <= 10, "DDM threshold valid"),
            (0 < self.ddm.noise_sigma <= 3, "DDM noise valid"),
        ]
        
        all_valid = True
        for condition, message in checks:
            if not condition:
                print(f"❌ {message}")
                all_valid = False
            else:
                print(f"✅ {message}")
        
        return all_valid


# Singleton instance
_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get or create configuration instance."""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config
EOF
```

2. Create `.env.example` template:
```bash
cat > .env.example << 'EOF'
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=x-ai/grok-4-fast
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions

# API Settings
API_TIMEOUT=30
API_MAX_RETRIES=3

# Application Settings
ENABLE_TRACES=true
LOG_LEVEL=INFO

# DDM Configuration (optional overrides)
# DDM_BASE_DRIFT=1.0
# DDM_THRESHOLD=1.0
# DDM_NOISE_SIGMA=1.0
EOF
```

3. Create your actual `.env` file:
```bash
cp .env.example .env
# Edit .env and add your actual API key
nano .env  # or vim, code, etc.
```

4. Add `.env` to `.gitignore`:
```bash
cat > .gitignore << 'EOF'
# Environment and secrets
.env
.env.local
*.key

# Virtual environment
venv/
env/
.venv/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# Data (optional: remove if you want to track data)
data/outputs/
data/visualizations/

# OS
.DS_Store
Thumbs.db
EOF
```

#### Verification Script

Create and run a validation script:

```bash
cat > scripts/validate_config.py << 'EOF'
#!/usr/bin/env python3
"""Validate configuration setup."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from addm_framework.utils.config import get_config

def main():
    print("Validating ADDM Framework Configuration...\n")
    
    try:
        config = get_config()
        
        print("Configuration loaded successfully!")
        print(f"\nProject Root: {config.app.project_root}")
        print(f"Data Directory: {config.app.data_dir}")
        print(f"Output Directory: {config.app.output_dir}")
        print(f"Log Directory: {config.app.log_dir}")
        print(f"\nAPI Model: {config.api.model}")
        print(f"API Timeout: {config.api.timeout}s")
        print(f"Max Retries: {config.api.max_retries}")
        print(f"\nDDM Base Drift: {config.ddm.base_drift}")
        print(f"DDM Threshold: {config.ddm.threshold}")
        print(f"DDM Trials: {config.ddm.n_trials}")
        
        print("\nRunning validation checks...\n")
        if config.validate():
            print("\n✅ All configuration checks passed!")
            return 0
        else:
            print("\n❌ Some configuration checks failed!")
            return 1
    
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x scripts/validate_config.py
```

Run validation:
```bash
mkdir -p scripts
python scripts/validate_config.py
```

#### Verification
- [ ] `.env` file created with valid API key
- [ ] `.env` in `.gitignore`
- [ ] Config module imports without errors
- [ ] Validation script passes all checks
- [ ] All directories created automatically

---

### Step 5: Logging Infrastructure

**Purpose:** Set up comprehensive logging for debugging and monitoring  
**Duration:** 15 minutes

#### Instructions

1. Create logging utility:
```bash
cat > src/addm_framework/utils/logging.py << 'EOF'
"""Logging infrastructure for ADDM Framework."""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored console output for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    Set up application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        enable_console: Enable console output
        enable_file: Enable file output
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("addm_framework")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(
            console_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"addm_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        file_format = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(
            file_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"addm_framework.{name}")
EOF
```

2. Create test script:
```bash
cat > scripts/test_logging.py << 'EOF'
#!/usr/bin/env python3
"""Test logging infrastructure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from addm_framework.utils.config import get_config
from addm_framework.utils.logging import setup_logging, get_logger

def main():
    # Load config
    config = get_config()
    
    # Setup logging
    setup_logging(
        log_level=config.app.log_level,
        log_dir=config.app.log_dir,
        enable_console=True,
        enable_file=True
    )
    
    # Get module-specific logger
    logger = get_logger("test")
    
    # Test all log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    print(f"\n✅ Logging test complete. Check logs in: {config.app.log_dir}")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/test_logging.py
```

3. Run logging test:
```bash
python scripts/test_logging.py
```

4. Verify log file created:
```bash
ls -lh logs/
cat logs/addm_*.log | head -20
```

#### Verification
- [ ] Console output shows colored log levels
- [ ] Log file created in `logs/` directory
- [ ] All log levels working (DEBUG through CRITICAL)
- [ ] Timestamps formatted correctly
- [ ] Module names included in logs

---

### Step 6: Create Package Initialization

**Purpose:** Set up package metadata and version management  
**Duration:** 10 minutes

#### Instructions

1. Create main package `__init__.py`:
```bash
cat > src/addm_framework/__init__.py << 'EOF'
"""
ADDM Framework: Agentic Drift-Diffusion Model for Decision Making

A production-ready cognitive decision-making system combining DDM principles
with LLM agents for transparent, evidence-based autonomous decisions.
"""

__version__ = "0.1.0"
__author__ = "ADDM Framework Contributors"
__license__ = "MIT"

# Import main components (will be added in later phases)
# from .agent import ADDM_Agent
# from .ddm import MultiAlternativeDDM
# from .llm import OpenRouterClient
# from .models import ActionCandidate, DDMOutcome, AgentResponse

__all__ = [
    "__version__",
    # "ADDM_Agent",
    # "MultiAlternativeDDM", 
    # "OpenRouterClient",
    # "ActionCandidate",
    # "DDMOutcome",
    # "AgentResponse",
]


def get_version() -> str:
    """Return the current version."""
    return __version__
EOF
```

2. Create `setup.py` for package installation:
```bash
cat > setup.py << 'EOF'
"""Setup script for ADDM Framework."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="addm-framework",
    version="0.1.0",
    description="Agentic Drift-Diffusion Model for Decision Making",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ADDM Framework Contributors",
    url="https://github.com/your-org/addm-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=4.0",
            "black>=24.0",
            "mypy>=1.8",
            "pylint>=3.0",
        ],
        "api": [
            "fastapi>=0.110",
            "uvicorn[standard]>=0.27",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
EOF
```

3. Install package in development mode:
```bash
pip install -e .
```

4. Test package import:
```bash
python -c "import addm_framework; print(f'ADDM Framework v{addm_framework.get_version()}')"
```

#### Verification
- [ ] Package installed successfully
- [ ] Import works without errors
- [ ] Version number accessible
- [ ] Development mode allows code changes without reinstall

---

### Step 7: Create Basic README

**Purpose:** Document project setup and usage  
**Duration:** 10 minutes

#### Instructions

```bash
cat > README.md << 'EOF'
# ADDM Framework

Agentic Drift-Diffusion Model for Evidence-Based Decision Making

## Overview

The ADDM Framework combines cognitive neuroscience (Drift-Diffusion Models) with Large Language Models to create transparent, evidence-based decision-making agents.

**Key Features:**
- 🧠 Multi-alternative DDM with racing accumulators
- 🤖 OpenRouter/Grok integration for evidence generation
- 📊 Built-in A/B testing (DDM vs argmax)
- 🔍 Comprehensive trace logging
- ⚡ Parallel API calls for low latency
- 🎨 Trajectory visualizations

## Quick Start

### Prerequisites
- Python 3.10+
- OpenRouter API key (get at https://openrouter.ai)

### Installation

1. Clone repository:
```bash
git clone <your-repo-url>
cd addm-framework
```

2. Set up environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

3. Configure API key:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

4. Validate setup:
```bash
python scripts/validate_config.py
```

### Usage

```python
from addm_framework.agent import ADDM_Agent

# Initialize agent
agent = ADDM_Agent(api_key="your_key")

# Make a decision
response = agent.decide_and_act(
    user_input="Recommend a healthy breakfast",
    task_type="nutrition",
    mode="ddm"
)

print(response.decision)
print(f"Confidence: {response.metrics['confidence']:.2%}")
```

## Project Status

**Current Phase:** Phase 1 - Foundation Setup ✅  
**Next Phase:** Phase 2 - Data Models

See `docs/phases/Phase0.md` for complete roadmap.

## Development

Run tests:
```bash
pytest tests/ -v --cov=src/addm_framework
```

Format code:
```bash
black src/ tests/
```

Type checking:
```bash
mypy src/
```

## Documentation

- [Phase 0: Project Overview](docs/phases/Phase0.md)
- [Phase 1: Foundation Setup](docs/phases/Phase1.md)
- API Documentation (coming soon)
- Usage Examples (coming soon)

## License

MIT License - see LICENSE file

## Citation

```bibtex
@software{addm_framework_2024,
  title={Agentic Drift-Diffusion Model Framework},
  year={2024},
  url={https://github.com/your-org/addm-framework}
}
```

## Contact

For questions or issues, please open a GitHub issue.
EOF
```

#### Verification
- [ ] README renders correctly in markdown viewer
- [ ] Quick start instructions are clear
- [ ] Links work (if repository exists)

---

### Step 8: API Key Setup & Validation

**Purpose:** Ensure API credentials are properly configured  
**Duration:** 10 minutes

#### Instructions

1. Get your OpenRouter API key:
   - Visit https://openrouter.ai
   - Sign up / Log in
   - Go to Keys section
   - Create a new API key
   - Copy the key

2. Add to `.env` file:
```bash
nano .env
```

Add:
```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
```

3. Test API connection:
```bash
cat > scripts/test_api.py << 'EOF'
#!/usr/bin/env python3
"""Test OpenRouter API connection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests
from addm_framework.utils.config import get_config
from addm_framework.utils.logging import setup_logging, get_logger

def test_api_connection():
    """Test basic API connectivity."""
    config = get_config()
    logger = get_logger("api_test")
    
    setup_logging(log_level="INFO", log_dir=config.app.log_dir)
    
    logger.info("Testing OpenRouter API connection...")
    
    headers = {
        "Authorization": f"Bearer {config.api.api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.api.model,
        "messages": [
            {"role": "user", "content": "Say 'API test successful' if you receive this."}
        ],
        "max_tokens": 20
    }
    
    try:
        response = requests.post(
            config.api.base_url,
            headers=headers,
            json=payload,
            timeout=config.api.timeout
        )
        
        response.raise_for_status()
        data = response.json()
        
        message = data['choices'][0]['message']['content']
        logger.info(f"✅ API Response: {message}")
        logger.info(f"✅ Model: {config.api.model}")
        logger.info(f"✅ Connection successful!")
        
        return True
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            logger.error("❌ Invalid API key!")
            logger.error("Check your .env file and ensure OPENROUTER_API_KEY is correct")
        else:
            logger.error(f"❌ HTTP Error: {e}")
        return False
    
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_api_connection()
    sys.exit(0 if success else 1)
EOF

chmod +x scripts/test_api.py
python scripts/test_api.py
```

#### Expected Output
```
2025-10-27 14:30:45 | INFO | addm_framework.api_test | Testing OpenRouter API connection...
2025-10-27 14:30:46 | INFO | addm_framework.api_test | ✅ API Response: API test successful
2025-10-27 14:30:46 | INFO | addm_framework.api_test | ✅ Model: x-ai/grok-4-fast
2025-10-27 14:30:46 | INFO | addm_framework.api_test | ✅ Connection successful!
```

#### Verification
- [ ] API key in `.env` file
- [ ] API test script runs successfully
- [ ] Response received from OpenRouter
- [ ] No authentication errors

#### Troubleshooting

**Error: "401 Unauthorized"**
- Check API key is correct in `.env`
- Ensure no extra spaces in key
- Verify key is active in OpenRouter dashboard

**Error: "429 Rate Limit"**
- You're on free tier with limits
- Wait a moment and retry
- Consider upgrading plan if needed

**Error: "Connection timeout"**
- Check internet connection
- Try increasing timeout in config
- Verify OpenRouter is not down (check status page)

---

### Step 9: Create Smoke Test

**Purpose:** Ensure all components can import and initialize  
**Duration:** 10 minutes

#### Instructions

```bash
cat > scripts/smoke_test.py << 'EOF'
#!/usr/bin/env python3
"""Smoke test for Phase 1 completion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_smoke_tests():
    """Run all smoke tests."""
    print("🔍 Running Phase 1 Smoke Tests...\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import core utils
    try:
        from addm_framework.utils import config, logging
        print("✅ Test 1: Core utils import")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1: Core utils import failed: {e}")
        tests_failed += 1
    
    # Test 2: Load configuration
    try:
        from addm_framework.utils.config import get_config
        cfg = get_config()
        assert cfg.api.api_key, "API key not set"
        print("✅ Test 2: Configuration loaded")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2: Configuration failed: {e}")
        tests_failed += 1
    
    # Test 3: Setup logging
    try:
        from addm_framework.utils.logging import setup_logging, get_logger
        setup_logging(log_level="INFO", log_dir=cfg.app.log_dir)
        logger = get_logger("smoke_test")
        logger.info("Logging system operational")
        print("✅ Test 3: Logging initialized")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3: Logging failed: {e}")
        tests_failed += 1
    
    # Test 4: Check directories
    try:
        assert cfg.app.project_root.exists(), "Project root not found"
        assert cfg.app.data_dir.exists(), "Data directory not found"
        assert cfg.app.output_dir.exists(), "Output directory not found"
        assert cfg.app.log_dir.exists(), "Log directory not found"
        print("✅ Test 4: Directory structure verified")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4: Directories failed: {e}")
        tests_failed += 1
    
    # Test 5: Dependency imports
    try:
        import numpy
        import matplotlib
        import pydantic
        import requests
        import aiohttp
        print("✅ Test 5: Dependencies available")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 5: Dependencies failed: {e}")
        tests_failed += 1
    
    # Test 6: Configuration validation
    try:
        is_valid = cfg.validate()
        assert is_valid, "Configuration validation failed"
        print("✅ Test 6: Configuration validated")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 6: Validation failed: {e}")
        tests_failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"{'='*50}")
    
    if tests_failed == 0:
        print("\n🎉 All smoke tests passed! Phase 1 complete.")
        print("Next step: Proceed to Phase 2 (Data Models)")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_smoke_tests())
EOF

chmod +x scripts/smoke_test.py
python scripts/smoke_test.py
```

#### Expected Output
```
🔍 Running Phase 1 Smoke Tests...

✅ Test 1: Core utils import
✅ Test 2: Configuration loaded
✅ Test 3: Logging initialized
✅ Test 4: Directory structure verified
✅ Test 5: Dependencies available
✅ Test 6: Configuration validated

==================================================
Tests Passed: 6
Tests Failed: 0
==================================================

🎉 All smoke tests passed! Phase 1 complete.
Next step: Proceed to Phase 2 (Data Models)
```

#### Verification
- [ ] All 6 smoke tests pass
- [ ] No import errors
- [ ] Configuration valid
- [ ] Directories exist

---

## Testing Procedures

### Manual Verification Checklist

Run through this checklist to ensure Phase 1 is complete:

```bash
# 1. Virtual environment
which python  # Should show venv/bin/python
pip list | grep -E "(numpy|pydantic|requests)"  # Should show versions

# 2. Directory structure
ls -la src/addm_framework/  # All subdirs present
ls -la tests/  # Test directories present
ls -la docs/phases/  # Phase docs directory

# 3. Configuration
python -c "from addm_framework.utils.config import get_config; get_config().validate()"
# Should output ✅ checkmarks

# 4. Logging
python scripts/test_logging.py
ls logs/  # Should show log file

# 5. API connection
python scripts/test_api.py  # Should succeed

# 6. Smoke test
python scripts/smoke_test.py  # All tests pass
```

### Automated Test Suite (pytest)

Create initial test file:

```bash
cat > tests/unit/test_phase1.py << 'EOF'
"""Unit tests for Phase 1 components."""
import pytest
from pathlib import Path
import os


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_loads(self):
        """Test that configuration loads successfully."""
        from addm_framework.utils.config import get_config
        config = get_config()
        assert config is not None
        assert config.api.api_key
    
    def test_ddm_config_defaults(self):
        """Test DDM config has valid defaults."""
        from addm_framework.utils.config import DDMConfig
        ddm = DDMConfig()
        assert 0 < ddm.base_drift <= 5
        assert 0 < ddm.threshold <= 10
        assert 0 < ddm.noise_sigma <= 3
    
    def test_api_config_validation(self):
        """Test API config validation."""
        from addm_framework.utils.config import APIConfig
        
        # Should fail without API key
        os.environ.pop("OPENROUTER_API_KEY", None)
        with pytest.raises(ValueError):
            APIConfig.from_env()


class TestLogging:
    """Test logging infrastructure."""
    
    def test_logger_creation(self):
        """Test logger can be created."""
        from addm_framework.utils.logging import get_logger
        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "addm_framework.test"
    
    def test_logging_setup(self, tmp_path):
        """Test logging setup."""
        from addm_framework.utils.logging import setup_logging
        
        logger = setup_logging(
            log_level="INFO",
            log_dir=tmp_path,
            enable_console=True,
            enable_file=True
        )
        
        assert logger is not None
        assert len(logger.handlers) >= 1


class TestPackage:
    """Test package structure."""
    
    def test_version(self):
        """Test version accessible."""
        import addm_framework
        assert addm_framework.__version__
        assert isinstance(addm_framework.__version__, str)
    
    def test_imports(self):
        """Test key modules can be imported."""
        from addm_framework.utils import config, logging
        assert config is not None
        assert logging is not None


def test_directories_exist():
    """Test required directories exist."""
    from addm_framework.utils.config import get_config
    config = get_config()
    
    assert config.app.project_root.exists()
    assert config.app.data_dir.exists()
    assert config.app.output_dir.exists()
    assert config.app.log_dir.exists()
EOF
```

Run tests:
```bash
pytest tests/unit/test_phase1.py -v --cov=src/addm_framework/utils
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError
**Symptom:** `ModuleNotFoundError: No module named 'addm_framework'`

**Solutions:**
```bash
# Ensure virtual environment active
source venv/bin/activate

# Install package in dev mode
pip install -e .

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Import Errors
**Symptom:** `ImportError: attempted relative import with no known parent package`

**Solution:** Always run scripts from project root, not from subdirectories:
```bash
# Correct
python scripts/smoke_test.py

# Incorrect
cd scripts && python smoke_test.py
```

#### 3. Missing Dependencies
**Symptom:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
pip install -r requirements.txt
pip list  # Verify installations
```

#### 4. Permission Errors
**Symptom:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Fix script permissions
chmod +x scripts/*.py

# Fix directory permissions
chmod -R u+rw logs/ data/
```

#### 5. API Key Not Found
**Symptom:** `ValueError: OPENROUTER_API_KEY environment variable not set`

**Solution:**
```bash
# Check .env exists
ls -la .env

# Load environment variables
set -a
source .env
set +a

# Verify
echo $OPENROUTER_API_KEY
```

---

## Next Steps

### Immediate Actions
1. **Verify Completion**: Run `python scripts/smoke_test.py`
2. **Commit Progress**: Git commit with message "Complete Phase 1: Foundation"
3. **Review Phase 2**: Read `docs/phases/Phase2.md` (Data Models)

### Phase 2 Preview
Next phase will implement:
- Pydantic data models
- Evidence schemas
- Validation logic
- Type-safe data flow

**Estimated Duration:** 3-4 hours  
**Prerequisites:** Phase 1 complete (this phase)

---

## Summary

### What Was Accomplished
✅ Project directory structure  
✅ Virtual environment with dependencies  
✅ Configuration management system  
✅ Environment variable handling  
✅ Logging infrastructure  
✅ API key validation  
✅ Package installation  
✅ Smoke tests passing  

### Key Deliverables
- `src/addm_framework/` package structure
- `requirements.txt` with all dependencies
- `.env` file with API credentials
- `config.py` for centralized configuration
- `logging.py` for structured logging
- Validation scripts
- Initial test suite

### Verification Commands
```bash
# Quick verification
python scripts/smoke_test.py

# Full verification
python scripts/validate_config.py
python scripts/test_logging.py
python scripts/test_api.py
pytest tests/unit/test_phase1.py -v
```

---

**Phase 1 Status:** ✅ COMPLETE  
**Ready for Phase 2:** YES  
**Next Phase Document:** `docs/phases/Phase2.md`

