#!/bin/bash
# NeuralLayers Quick Setup Script
#
# This script automates the setup process for NeuralLayers

set -e  # Exit on error

echo "=========================================================================="
echo "🚀 NeuralLayers Quick Setup"
echo "=========================================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Python version
echo ""
print_info "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.8"
if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_success "Python version is compatible (>= 3.8)"
else
    print_error "Python 3.8 or higher is required!"
    exit 1
fi

# Check if virtual environment exists
echo ""
print_info "Checking for virtual environment..."
if [ -d "venv" ]; then
    print_success "Virtual environment found"
else
    print_info "Creating virtual environment..."
    python -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate || {
    print_error "Failed to activate virtual environment"
    print_info "Please activate manually: source venv/bin/activate"
    exit 1
}
print_success "Virtual environment activated"

# Upgrade pip
echo ""
print_info "Upgrading pip..."
pip install --upgrade pip -q
print_success "pip upgraded"

# Install dependencies
echo ""
print_info "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Install NeuralLayers in development mode
echo ""
print_info "Installing NeuralLayers in development mode..."
pip install -e . -q
print_success "NeuralLayers installed"

# Run health check
echo ""
echo "=========================================================================="
echo "🏥 Running Health Check"
echo "=========================================================================="
python health_check.py

# Print next steps
echo ""
echo "=========================================================================="
echo "✨ Setup Complete!"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Try the examples:"
echo "     python examples/simple_network.py"
echo "     python examples/basic_training.py"
echo ""
echo "  3. Run benchmarks:"
echo "     cd benchmarks && python benchmark_inference.py"
echo ""
echo "  4. Start Jupyter:"
echo "     jupyter lab notebooks/"
echo ""
echo "  5. Run tests:"
echo "     pytest tests/ -v"
echo ""
echo "  6. Launch demo:"
echo "     python demo_app.py --mode streamlit"
echo ""
echo "=========================================================================="
echo "📚 Documentation: README.md"
echo "🐛 Issues: https://github.com/biblicalandr0id/NeuralLayers/issues"
echo "=========================================================================="
