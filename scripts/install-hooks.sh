#!/usr/bin/env bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Git Hooks Installer${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo ""

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo -e "${RED}✗ Not in a git repository${NC}"
    exit 1
}

cd "$REPO_ROOT"

# Determine hooks directory (works with worktrees)
HOOKS_DIR=$(git rev-parse --git-path hooks)

echo "Repository root: $REPO_ROOT"
echo "Hooks directory: $HOOKS_DIR"
echo ""

# Check if hooks directory exists
if [ ! -d "$HOOKS_DIR" ]; then
    echo -e "${YELLOW}Creating hooks directory...${NC}"
    mkdir -p "$HOOKS_DIR"
fi

# Install pre-commit hook
echo "Installing pre-commit hook..."

if [ -f "$HOOKS_DIR/pre-commit" ]; then
    echo -e "${YELLOW}⚠ Pre-commit hook already exists${NC}"

    # Check if it's our hook
    if grep -q "Performance Validation" "$HOOKS_DIR/pre-commit" 2>/dev/null; then
        echo -e "${YELLOW}Existing hook appears to be ours, replacing...${NC}"
    else
        # Backup existing hook
        BACKUP="$HOOKS_DIR/pre-commit.backup.$(date +%s)"
        echo -e "${YELLOW}Backing up existing hook to: $BACKUP${NC}"
        cp "$HOOKS_DIR/pre-commit" "$BACKUP"
    fi
fi

# Copy our hook
cp "$REPO_ROOT/scripts/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo -e "${GREEN}✓ Pre-commit hook installed${NC}"
echo ""

# Verify installation
if [ -x "$HOOKS_DIR/pre-commit" ]; then
    echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Installation successful!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
    echo ""
    echo "The pre-commit hook will now:"
    echo "  1. Run all tests (cargo test)"
    echo "  2. Compare benchmarks with HEAD~1"
    echo "  3. Block commits if tests fail or performance regresses"
    echo ""
    echo "To bypass the hook in emergencies:"
    echo "  git commit --no-verify"
    echo ""
    echo "Note: The first commit will skip benchmark comparison"
    echo "      (no previous commit to compare against)"
    echo ""
else
    echo -e "${RED}✗ Installation failed - hook is not executable${NC}"
    exit 1
fi
