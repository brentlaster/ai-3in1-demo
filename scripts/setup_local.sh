#!/usr/bin/env bash
# setup.sh — Setup for ai-3in1-demo on macOS

set -e
echo "🛠️  Setting up Python environment for ai-3in1-demo..."

# 1. Check Python 3
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 is not installed. Please install it from https://www.python.org"
  exit 1
fi

# 2. Check if pip is available
if ! command -v pip3 &>/dev/null; then
  echo "📦 pip3 not found. Attempting to install with ensurepip..."
  python3 -m ensurepip --upgrade
fi

# 3. Create virtual environment
python3 -m venv .venv
echo "✅ Created virtual environment: .venv/"
# shellcheck disable=SC1091
source .venv/bin/activate
echo "✅ Activated virtual environment."

# 4. Upgrade pip
pip install --upgrade pip

# 5. Write requirements directly into temp file (or adapt path)
cat > /tmp/ai3in1_requirements.txt <<EOF
chromadb==1.0.15
fastmcp==2.10.2
openai==1.93.0
pdfplumber==0.11.7
requests==2.32.4
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
tiktoken==0.9.0
EOF

# 6. Install base dependencies
pip install -r /tmp/ai3in1_requirements.txt

# 7. Pin numpy to <2 in case it's pulled in
pip install "numpy<2"

# 8. Done
echo -e "\n✅ Setup complete. To activate your environment:"
echo "   source .venv/bin/activate"
