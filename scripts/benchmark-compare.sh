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

# Run baseline benchmarks and save output
BASELINE_OUTPUT=$(mktemp)
trap "rm -rf $BASELINE_DIR $BASELINE_OUTPUT" EXIT

if ! cargo bench --no-default-features --quiet > "$BASELINE_OUTPUT" 2>&1; then
    git checkout -q "$CURRENT_BRANCH"
    [ "$STASHED" = true ] && git stash pop -q
    echo -e "${YELLOW}⚠ Baseline benchmarks failed - skipping comparison${NC}"
    exit 0
fi

tail -n 20 "$BASELINE_OUTPUT"

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

# Run current benchmarks
CURRENT_OUTPUT=$(mktemp)
trap "rm -rf $BASELINE_DIR $BASELINE_OUTPUT $CURRENT_OUTPUT" EXIT

if ! cargo bench --no-default-features --quiet > "$CURRENT_OUTPUT" 2>&1; then
    echo -e "${RED}✗ Current benchmarks failed to run${NC}"
    cat "$CURRENT_OUTPUT"
    exit 1
fi

tail -n 20 "$CURRENT_OUTPUT"

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "📊 Analyzing Results..."
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Helper function to convert time to microseconds
convert_to_us() {
    local value=$1
    local unit=$2

    case "$unit" in
        "ps") echo "$value / 1000000" | bc -l ;;
        "ns") echo "$value / 1000" | bc -l ;;
        "µs"|"us") echo "$value" ;;
        "ms") echo "$value * 1000" | bc -l ;;
        "s") echo "$value * 1000000" | bc -l ;;
        *) echo "$value" ;;
    esac
}

# Parse divan output format
# Example line: │  ├─ 1000                        710.8 µs      │ 803.9 µs      │ 723.3 µs      │ 730.2 µs      │ 100     │ 100
parse_benchmarks() {
    local file=$1
    local prefix=""

    grep -E "^(├─|│|╰─)" "$file" | while IFS= read -r line; do
        # Extract benchmark name (remove tree characters and trim)
        local name=$(echo "$line" | sed 's/[├─│╰└]//g' | awk '{print $1}')

        # Skip empty names
        [ -z "$name" ] && continue

        # Skip if this is a parent category (no timing data)
        if ! echo "$line" | grep -qE "[0-9]+\.[0-9]+ (ps|ns|µs|us|ms|s)"; then
            prefix="$name"
            continue
        fi

        # Extract all time values
        local times=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+ (ps|ns|µs|us|ms|s)")

        # Get the 3rd time value (median) - skip if not enough values
        local median=$(echo "$times" | sed -n '3p')

        if [ -n "$median" ]; then
            local value=$(echo "$median" | awk '{print $1}')
            local unit=$(echo "$median" | awk '{print $2}')

            # Skip if value is empty or invalid
            if [ -z "$value" ] || ! echo "$value" | grep -qE "^[0-9]+\.?[0-9]*$"; then
                continue
            fi

            local value_us=$(convert_to_us "$value" "$unit")

            # Output: benchmark_name:value_in_microseconds
            if [ -n "$prefix" ]; then
                echo "${prefix}_${name}:${value_us}"
            else
                echo "${name}:${value_us}"
            fi
        fi
    done
}

# Parse both outputs to temp files
BASELINE_PARSED=$(mktemp)
CURRENT_PARSED=$(mktemp)
RESULTS=$(mktemp)
trap "rm -rf $BASELINE_DIR $BASELINE_OUTPUT $CURRENT_OUTPUT $BASELINE_PARSED $CURRENT_PARSED $RESULTS" EXIT

parse_benchmarks "$BASELINE_OUTPUT" > "$BASELINE_PARSED"
parse_benchmarks "$CURRENT_OUTPUT" > "$CURRENT_PARSED"

# Compare benchmarks - write results to file to avoid subshell variable issues
: > "$RESULTS"  # Clear results file

# Read baseline benchmarks and compare with current
while IFS=: read -r bench baseline_time; do
    # Skip if bench name or baseline_time is empty
    [ -z "$bench" ] && continue
    [ -z "$baseline_time" ] && continue

    # Find matching benchmark in current output
    current_time=$(grep "^${bench}:" "$CURRENT_PARSED" | cut -d: -f2)

    if [ -n "$current_time" ] && [ "$current_time" != "0" ] && [ "$baseline_time" != "0" ]; then
        # Calculate percentage change (with error handling)
        change=$(echo "scale=2; (($current_time - $baseline_time) / $baseline_time) * 100" | bc -l 2>/dev/null)

        # Skip if bc failed
        [ -z "$change" ] && continue

        # Check if change is significant (>5%)
        abs_change=$(echo "$change" | tr -d '-')
        is_significant=$(echo "$abs_change > 5" | bc -l 2>/dev/null)

        # Default to unchanged if bc failed
        [ -z "$is_significant" ] && is_significant=0

        if [ "$is_significant" -eq 1 ]; then
            is_regression=$(echo "$change > 0" | bc -l 2>/dev/null)

            if [ "$is_regression" -eq 1 ]; then
                speedup=$(echo "scale=2; $current_time / $baseline_time" | bc -l 2>/dev/null)
                echo "REGRESSION:$bench:${speedup}:${change}" >> "$RESULTS"
                echo -e "${RED}⬆ Regression: $bench is ${speedup}× slower (${change}% slower)${NC}"
            else
                speedup=$(echo "scale=2; $baseline_time / $current_time" | bc -l 2>/dev/null)
                abs_change_display=$(echo "$change" | tr -d '-')
                echo "IMPROVEMENT:$bench:${speedup}:${abs_change_display}" >> "$RESULTS"
                echo -e "${GREEN}⬇ Improvement: $bench is ${speedup}× faster (${abs_change_display}% faster)${NC}"
            fi
        else
            echo "UNCHANGED:$bench" >> "$RESULTS"
        fi
    fi
done < "$BASELINE_PARSED"

# Count results
REGRESSIONS=$(grep -c "^REGRESSION:" "$RESULTS" 2>/dev/null || echo "0")
IMPROVEMENTS=$(grep -c "^IMPROVEMENT:" "$RESULTS" 2>/dev/null || echo "0")
UNCHANGED=$(grep -c "^UNCHANGED:" "$RESULTS" 2>/dev/null || echo "0")

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
