<#
.SYNOPSIS
    Run ObjectLens setup pipeline (PowerShell version)

.DESCRIPTION
    Cross-platform setup: downloads datasets, preprocesses data, and configures database and index.
    This script is the PowerShell analogue of run_pipeline.sh, designed for Windows users.

.NOTES
    Ensure Python and all dependencies are installed and on your PATH.
    Activate your virtual environment before running if required.
#>

function Write-Colored {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    $colors = @{
        "Green"  = "DarkGreen"
        "Yellow" = "Yellow"
        "Red"    = "Red"
        "Reset"  = "White"
    }
    Write-Host $Text -ForegroundColor $colors[$Color]
}

function Run-Step {
    param (
        [string]$Message,
        [string]$Command,
        [int]$StepNumber,
        [int]$TotalSteps
    )
    $yellowStep = "[{0}/{1}]" -f $StepNumber, $TotalSteps
    Write-Colored "$yellowStep $Message" "Yellow"
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Colored "Failed at step $StepNumber: $Message" "Red"
        exit 1
    }
}

Write-Host "==========================================" 
Write-Host "  ObjectLens Setup Pipeline"
Write-Host "=========================================="

$TotalSteps = 8

# Step 1: Download ImageNet
Run-Step "Downloading ImageNet Winter21..." "python scripts/dataset/imagenet_01_download_winter21.py" 1 $TotalSteps

# Step 2: Verify
Run-Step "Verifying download..." "python scripts/dataset/imagenet_02_verify_download.py" 2 $TotalSteps

# Step 3: Build YOLO dataset
Run-Step "Building YOLO dataset..." "python scripts/dataset/imagenet_03_build_yolo_dataset.py" 3 $TotalSteps

# Step 4: Precompute features
Run-Step "Precomputing features..." "python scripts/preprocessing/imagenet_04_precompute_features.py" 4 $TotalSteps

# Step 5: Download pottery dataset
Run-Step "Downloading pottery dataset..." "python scripts/dataset/pottery_01_download.py" 5 $TotalSteps

# Step 6: Build pottery catalog
Run-Step "Building pottery catalog..." "python scripts/dataset/pottery_02_build_catalog.py" 6 $TotalSteps

# Step 7: Split pottery catalog
Run-Step "Splitting pottery catalog..." "python scripts/dataset/pottery_03_split_catalog.py" 7 $TotalSteps

# Step 8: Setup DB and index
Write-Colored "[8/8] Setting up database and index..." "Yellow"
# Try to find 'bash' in path, otherwise warn and skip
$bashPath = Get-Command bash -ErrorAction SilentlyContinue
if ($null -ne $bashPath) {
    bash scripts/indexing/setup_db_and_index.sh
    if ($LASTEXITCODE -ne 0) {
        Write-Colored "Failed at step 8: Setting up database and index" "Red"
        exit 1
    }
} else {
    Write-Colored "Warning: bash is not installed. Please run scripts/indexing/setup_db_and_index.sh manually or in WSL." "Red"
    exit 1
}

Write-Colored "✔ Pipeline complete!" "Green"

# Additional pottery steps (not in main step count)
Write-Colored "Initializing MongoDB for pottery catalog..." "Yellow"
python scripts/dataset/pottery_04_init_mongodb.py
if ($LASTEXITCODE -ne 0) {
    Write-Colored "Failed initializing MongoDB" "Red"
    exit 1
}

Write-Colored "Indexing pottery catalog..." "Yellow"
python scripts/dataset/pottery_05_index_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Colored "Failed indexing pottery catalog" "Red"
    exit 1
}

# For optional evaluation, uncomment:
# Write-Colored "Evaluating retrieval..." "Yellow"
# python scripts/dataset/pottery_06_evaluate_retrieval.py

Write-Colored "✔ All steps completed successfully!" "Green"
