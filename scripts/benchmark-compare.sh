#!/usr/bin/env bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Performance Regression Check${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if we have a previous commit
if ! git rev-parse HEAD~1 >/dev/null 2>&1; then
    echo -e "${GREEN}✓ First commit - skipping benchmark comparison${NC}"
    exit 0
fi

# Check if we have uncommitted changes
if git diff-index --quiet HEAD --; then
    HAVE_CHANGES=false
else
    HAVE_CHANGES=true
fi

# Create a temporary directory for baseline results
BASELINE_DIR=$(mktemp -d)
trap "rm -rf $BASELINE_DIR" EXIT

echo "📊 Running baseline benchmarks (HEAD~1)..."
echo ""

# Save current state
if [ "$HAVE_CHANGES" = true ]; then
    # We have uncommitted changes, need to stash them
    STASH_NAME="pre-commit-benchmark-$$"
    git stash push -q -m "$STASH_NAME" || {
        echo -e "${RED}✗ Failed to stash changes${NC}"
        exit 1
    }
    STASHED=true
else
    STASHED=false
fi

# Checkout previous commit
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git checkout -q HEAD~1 || {
    [ "$STASHED" = true ] && git stash pop -q
    echo -e "${RED}✗ Failed to checkout HEAD~1${NC}"
    exit 1
}

# Check if benches directory exists in previous commit
if [ ! -d "benches" ]; then
    echo -e "${GREEN}✓ Previous commit has no benchmarks - skipping comparison${NC}"
    git checkout -q "$CURRENT_BRANCH"
    [ "$STASHED" = true ] && git stash pop -q
    exit 0
fi

# Run baseline benchmarks
if ! cargo bench --no-default-features --quiet 2>&1 | tail -n 20; then
    git checkout -q "$CURRENT_BRANCH"
    [ "$STASHED" = true ] && git stash pop -q
    echo -e "${YELLOW}⚠ Baseline benchmarks failed - skipping comparison${NC}"
    exit 0
fi

echo ""
echo "📈 Running current benchmarks..."
echo ""

# Return to current state
git checkout -q "$CURRENT_BRANCH" || {
    echo -e "${RED}✗ Failed to checkout $CURRENT_BRANCH${NC}"
    exit 1
}

if [ "$STASHED" = true ]; then
    git stash pop -q || {
        echo -e "${RED}✗ Failed to restore stashed changes${NC}"
        exit 1
    }
fi

# Run current benchmarks and compare
BENCH_OUTPUT=$(mktemp)
trap "rm -rf $BASELINE_DIR $BENCH_OUTPUT" EXIT

if ! cargo bench --no-default-features --quiet -- --baseline previous > "$BENCH_OUTPUT" 2>&1; then
    echo -e "${RED}✗ Current benchmarks failed to run${NC}"
    cat "$BENCH_OUTPUT"
    exit 1
fi

# Show the benchmark output
tail -n 30 "$BENCH_OUTPUT"

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "📊 Analyzing Results..."
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Parse results looking for regressions
# Divan outputs comparison results with patterns like:
# - "faster" or "slower" in comparison lines
# - We need to check if any benchmark got slower

REGRESSIONS=0
IMPROVEMENTS=0
UNCHANGED=0

# Look for comparison indicators in the output
while IFS= read -r line; do
    if echo "$line" | grep -q "slower"; then
        REGRESSIONS=$((REGRESSIONS + 1))
        echo -e "${RED}⬆ Regression detected: $line${NC}"
    elif echo "$line" | grep -q "faster"; then
        IMPROVEMENTS=$((IMPROVEMENTS + 1))
        echo -e "${GREEN}⬇ Improvement: $line${NC}"
    elif echo "$line" | grep -qE "^\s*[0-9]+\.[0-9]+"; then
        # Benchmark result line without comparison - might mean no change
        :
    fi
done < "$BENCH_OUTPUT"

echo ""
echo "Summary:"
echo "  Improvements: $IMPROVEMENTS"
echo "  Regressions:  $REGRESSIONS"
echo "  Unchanged:    $UNCHANGED"
echo ""

if [ $REGRESSIONS -gt 0 ]; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ PERFORMANCE REGRESSION DETECTED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Your changes have made the code slower."
    echo "Please optimize before committing."
    echo ""
    echo "To bypass this check (not recommended):"
    echo "  git commit --no-verify"
    echo ""
    exit 1
elif [ $IMPROVEMENTS -eq 0 ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}⚠ NO PERFORMANCE IMPROVEMENT${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Benchmarks show no measurable improvement."
    echo "This commit must make the code faster."
    echo ""
    echo "To bypass this check (not recommended):"
    echo "  git commit --no-verify"
    echo ""
    exit 1
else
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ PERFORMANCE IMPROVED${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "All benchmarks show improvement. Good work! 🚀"
    echo ""
    exit 0
fi
