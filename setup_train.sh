#!/bin/bash
# Setup and Training Startup Script for XAI-Enhanced Vein GAN

echo "=========================================="
echo "XAI-Enhanced Vein GAN - Setup & Training"
echo "=========================================="
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
python3 --version
echo ""

# Step 2: Check if virtual environment exists
echo "Step 2: Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi
echo ""

# Step 3: Activate virtual environment
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Step 4: Install dependencies
echo "Step 4: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 5: Check data structure
echo "Step 5: Checking data structure..."
if [ -d "dorsalhandveins" ]; then
    echo "✓ Data directory found: dorsalhandveins/"
    
    # Check for subdirectories
    if [ -d "data/DorsalHandVeins_DB1" ]; then
        echo "✓ Database 1 found"
        DB1_COUNT=$(find data/DorsalHandVeins_DB1 -name "*.tif" | wc -l)
        echo "  - Images: $DB1_COUNT"
    else
        echo "⚠ Database 1 not found at data/DorsalHandVeins_DB1"
    fi
    
    if [ -d "data/DorsalHandVeins_DB2" ]; then
        echo "✓ Database 2 found"
        DB2_COUNT=$(find data/DorsalHandVeins_DB2 -name "*.tif" | wc -l)
        echo "  - Images: $DB2_COUNT"
    else
        echo "⚠ Database 2 not found at data/DorsalHandVeins_DB2"
    fi
else
    echo "⚠ Data directory 'dorsalhandveins' not found"
    echo "Please organize your data as follows:"
    echo "  data/"
    echo "  ├── DorsalHandVeins_DB1/"
    echo "  │   └── person_*.tif files"
    echo "  └── DorsalHandVeins_DB2/"
    echo "      └── person_*.tif files"
fi
echo ""

# Step 6: Create necessary directories
echo "Step 6: Creating necessary directories..."
mkdir -p checkpoints
mkdir -p results/{generated,xai/{gradcam,shap,lime},metrics,fairness,comparisons,training_progress}
mkdir -p logs
echo "✓ Directories created"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Organize your data (if not done)"
echo "2. Run: python main.py --mode train"
echo ""
