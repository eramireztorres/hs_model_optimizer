#!/bin/bash
# Script to run tests with various configurations

set -e

echo "=================================="
echo "HS Model Optimizer Test Runner"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if python is installed
if ! command -v venv/bin/python &> /dev/null; then
    print_error "Python not found in venv. Please run setup."
    exit 1
fi

PYTEST_CMD="venv/bin/python -m pytest"

# Parse command line arguments
MODE=${1:-"all"}

case $MODE in
    "all")
        print_status "Running all tests..."
        $PYTEST_CMD tests/ -v
        ;;

    "unit")
        print_status "Running unit tests only..."
        $PYTEST_CMD tests/unit/ -v
        ;;

    "integration")
        print_status "Running integration tests only..."
        $PYTEST_CMD tests/integration/ -v
        ;;

    "fast")
        print_status "Running fast tests only (excluding slow tests)..."
        $PYTEST_CMD tests/ -v -m "not slow"
        ;;

    "coverage")
        print_status "Running tests with coverage report..."
        $PYTEST_CMD tests/ --cov=src --cov-report=html --cov-report=term
        print_status "Coverage report generated in htmlcov/index.html"
        ;;

    "coverage-unit")
        print_status "Running unit tests with coverage..."
        $PYTEST_CMD tests/unit/ --cov=src --cov-report=html --cov-report=term
        print_status "Coverage report generated in htmlcov/index.html"
        ;;

    "verbose")
        print_status "Running tests in verbose mode with output..."
        $PYTEST_CMD tests/ -vv -s
        ;;

    "failed")
        print_status "Re-running last failed tests..."
        $PYTEST_CMD tests/ --lf -v
        ;;

    "specific")
        if [ -z "$2" ]; then
            print_error "Please specify a test file or pattern"
            echo "Usage: ./run_tests.sh specific <test_file_or_pattern>"
            exit 1
        fi
        print_status "Running specific tests: $2"
        $PYTEST_CMD tests/ -v -k "$2"
        ;;

    "help")
        echo "Usage: ./run_tests.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all           - Run all tests (default)"
        echo "  unit          - Run only unit tests"
        echo "  integration   - Run only integration tests"
        echo "  fast          - Run fast tests only (skip slow tests)"
        echo "  coverage      - Run tests with coverage report"
        echo "  coverage-unit - Run unit tests with coverage"
        echo "  verbose       - Run tests with verbose output"
        echo "  failed        - Re-run only failed tests"
        echo "  specific      - Run specific test (requires pattern)"
        echo "  help          - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh all"
        echo "  ./run_tests.sh unit"
        echo "  ./run_tests.sh coverage"
        echo "  ./run_tests.sh specific test_config"
        exit 0
        ;;

    *)
        print_error "Unknown mode: $MODE"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    print_status "Tests completed successfully!"
else
    print_error "Tests failed with exit code $exit_code"
fi

exit $exit_code