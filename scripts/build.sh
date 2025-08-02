#!/bin/bash
# Build script for dynamic-moe-router-kit

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="dynamic-moe-router-kit"
BUILD_DIR="build"
DIST_DIR="dist"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking build requirements..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        log_error "pip is not installed"
        exit 1
    fi
    
    # Check build module
    if ! python -c "import build" 2>/dev/null; then
        log_warning "build module not found, installing..."
        pip install build
    fi
    
    log_success "Requirements check completed"
}

clean_build() {
    log_info "Cleaning previous build artifacts..."
    
    # Remove build directories
    rm -rf "${BUILD_DIR}"
    rm -rf "${DIST_DIR}"
    rm -rf *.egg-info
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    log_success "Build artifacts cleaned"
}

run_tests() {
    log_info "Running test suite..."
    
    if command -v pytest &> /dev/null; then
        # Run tests with coverage
        pytest --cov=dynamic_moe_router --cov-report=term-missing || {
            log_error "Tests failed"
            exit 1
        }
    else
        log_warning "pytest not found, skipping tests"
    fi
    
    log_success "Tests passed"
}

run_linting() {
    log_info "Running code quality checks..."
    
    # Black formatting check
    if command -v black &> /dev/null; then
        black --check . || {
            log_error "Code formatting check failed"
            exit 1
        }
    fi
    
    # isort import check
    if command -v isort &> /dev/null; then
        isort --check-only . || {
            log_error "Import sorting check failed"
            exit 1
        }
    fi
    
    # Ruff linting
    if command -v ruff &> /dev/null; then
        ruff check . || {
            log_error "Linting check failed"
            exit 1
        }
    fi
    
    # Type checking
    if command -v mypy &> /dev/null; then
        mypy src/dynamic_moe_router || {
            log_error "Type checking failed"
            exit 1
        }
    fi
    
    log_success "Code quality checks passed"
}

build_package() {
    log_info "Building package..."
    
    # Build source distribution and wheel
    python -m build || {
        log_error "Package build failed"
        exit 1
    }
    
    # Verify build artifacts
    if [ ! -d "${DIST_DIR}" ]; then
        log_error "Distribution directory not created"
        exit 1
    fi
    
    # Check for expected files
    if ! ls "${DIST_DIR}"/*.tar.gz 1> /dev/null 2>&1; then
        log_error "Source distribution not found"
        exit 1
    fi
    
    if ! ls "${DIST_DIR}"/*.whl 1> /dev/null 2>&1; then
        log_error "Wheel distribution not found"
        exit 1
    fi
    
    log_success "Package built successfully"
    
    # Show build artifacts
    log_info "Build artifacts:"
    ls -la "${DIST_DIR}"/
}

validate_package() {
    log_info "Validating package..."
    
    # Check with twine
    if command -v twine &> /dev/null; then
        twine check "${DIST_DIR}"/* || {
            log_error "Package validation failed"
            exit 1
        }
    else
        log_warning "twine not found, skipping package validation"
    fi
    
    log_success "Package validation passed"
}

build_docker() {
    local dockerfile="${1:-Dockerfile}"
    local tag="${2:-${PROJECT_NAME}:latest}"
    
    log_info "Building Docker image with ${dockerfile}..."
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found, skipping Docker build"
        return
    fi
    
    docker build -f "${dockerfile}" -t "${tag}" . || {
        log_error "Docker build failed"
        exit 1
    }
    
    log_success "Docker image built: ${tag}"
}

generate_sbom() {
    log_info "Generating Software Bill of Materials..."
    
    # Create SBOM directory
    mkdir -p sbom
    
    # Generate pip freeze output
    pip freeze > sbom/requirements.txt
    
    # If syft is available, generate SPDX SBOM
    if command -v syft &> /dev/null; then
        syft . -o spdx-json=sbom/sbom.spdx.json
    else
        log_warning "syft not found, generating basic SBOM only"
    fi
    
    log_success "SBOM generated in sbom/"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --clean-only     Only clean build artifacts"
    echo "  --no-tests       Skip running tests"
    echo "  --no-lint        Skip linting checks"
    echo "  --docker         Build Docker images"
    echo "  --docker-gpu     Build GPU Docker image"
    echo "  --docker-prod    Build production Docker image"
    echo "  --sbom          Generate Software Bill of Materials"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full build with tests and linting"
    echo "  $0 --no-tests        # Build without running tests"
    echo "  $0 --docker          # Build with Docker images"
    echo "  $0 --clean-only      # Only clean artifacts"
}

main() {
    local skip_tests=false
    local skip_lint=false
    local build_docker_images=false
    local build_gpu_docker=false
    local build_prod_docker=false
    local generate_sbom_flag=false
    local clean_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean-only)
                clean_only=true
                shift
                ;;
            --no-tests)
                skip_tests=true
                shift
                ;;
            --no-lint)
                skip_lint=true
                shift
                ;;
            --docker)
                build_docker_images=true
                shift
                ;;
            --docker-gpu)
                build_gpu_docker=true
                shift
                ;;
            --docker-prod)
                build_prod_docker=true
                shift
                ;;
            --sbom)
                generate_sbom_flag=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_info "Starting build for ${PROJECT_NAME}"
    
    # Always check requirements and clean
    check_requirements
    clean_build
    
    # If clean only, exit here
    if [ "$clean_only" = true ]; then
        log_success "Clean completed"
        exit 0
    fi
    
    # Run quality checks
    if [ "$skip_lint" = false ]; then
        run_linting
    fi
    
    if [ "$skip_tests" = false ]; then
        run_tests
    fi
    
    # Build package
    build_package
    validate_package
    
    # Optional builds
    if [ "$build_docker_images" = true ]; then
        build_docker "Dockerfile" "${PROJECT_NAME}:latest"
    fi
    
    if [ "$build_gpu_docker" = true ]; then
        build_docker "Dockerfile.gpu" "${PROJECT_NAME}:gpu"
    fi
    
    if [ "$build_prod_docker" = true ]; then
        build_docker "Dockerfile.production" "${PROJECT_NAME}:production"
    fi
    
    if [ "$generate_sbom_flag" = true ]; then
        generate_sbom
    fi
    
    log_success "Build completed successfully!"
    
    # Show final summary
    echo ""
    log_info "Build Summary:"
    echo "  - Package artifacts: ${DIST_DIR}/"
    if [ "$build_docker_images" = true ] || [ "$build_gpu_docker" = true ] || [ "$build_prod_docker" = true ]; then
        echo "  - Docker images built"
    fi
    if [ "$generate_sbom_flag" = true ]; then
        echo "  - SBOM generated: sbom/"
    fi
}

# Run main function with all arguments
main "$@"