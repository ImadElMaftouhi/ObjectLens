<#
.SYNOPSIS
    Run ObjectLens setup pipeline (PowerShell version)

.DESCRIPTION
    Equivalent to run_pipeline.sh for Windows/PowerShell users.
    Executes dataset download, preprocessing, and database/index setup tasks.

.NOTES
    Make sure Python and all requirements are installed and available in PATH.
    If using a virtual environment, activate it before running this script.
#>

# Color constants
$GREEN  = "`e[0;32m"
$YELLOW = "`e[1;33m"
$RED    = "`e[0;31m"
$NC     = "`e[0m"

Write-Host "==========================================" 
Write-Host "  ObjectLens Setup Pipeline"
Write-Host "=========================================="

# Dataset scripts
Write-Host "$YELLOW[1/8]$NC Downloading ImageNet Winter21..."
python scripts/dataset/imagenet_01_download_winter21.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed at step 1 $NC"; exit 1 }

Write-Host "$YELLOW[2/8]$NC Verifying download..."
python scripts/dataset/imagenet_02_verify_download.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed at step 2 $NC"; exit 1 }

Write-Host "$YELLOW[3/8]$NC Building YOLO dataset..."
python scripts/dataset/imagenet_03_build_yolo_dataset.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed at step 3 $NC"; exit 1 }

# Preprocessing
Write-Host "$YELLOW[4/8]$NC Precomputing features..."
python scripts/preprocessing/imagenet_04_precompute_features.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed at step 4 $NC"; exit 1 }

# Pottery scripts
Write-Host "$YELLOW[5/8]$NC Downloading pottery dataset..."
python scripts/dataset/pottery_01_download.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed at step 5 $NC"; exit 1 }

Write-Host "$YELLOW[6/8]$NC Building pottery catalog..."
python scripts/dataset/pottery_02_build_catalog.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed at step 6 $NC"; exit 1 }

Write-Host "$YELLOW[7/8]$NC Splitting pottery catalog..."
python scripts/dataset/pottery_03_split_catalog.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed at step 7 $NC"; exit 1 }

# Database setup
Write-Host "$YELLOW[8/8]$NC Setting up database and index..."
# On Windows, bash scripts generally can't be executed without an appropriate shell.
# If 'bash' is installed (WSL/git-bash/MinGW), remove the comment below.
# Otherwise, convert setup_db_and_index.sh to PowerShell or run manually.
bash scripts/indexing/setup_db_and_index.sh
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed at step 8 $NC"; exit 1 }

Write-Host "$GREEN`t{u2714} Pipeline complete!$NC"

# indexing pottery catalog
Write-Host "$YELLOW[7/8]$NC Initializing MongoDB..."
python scripts/dataset/pottery_04_init_mongodb.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed initializing MongoDB $NC"; exit 1 }

Write-Host "$YELLOW[7/8]$NC Indexing pottery catalog..."
python scripts/dataset/pottery_05_index_models.py
if ($LASTEXITCODE -ne 0) { Write-Host "$RED Failed indexing pottery catalog $NC"; exit 1 }

# Write-Host "$YELLOW[7/8]$NC Evaluate retrieval..."
# python scripts/dataset/pottery_06_evaluate_retrieval.py

Write-Host "$GREEN`t{u2714} Pipeline complete!$NC"
