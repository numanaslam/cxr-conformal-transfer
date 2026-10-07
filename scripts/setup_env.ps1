<#
One-shot environment setup for OV-CBMR on a Windows CUDA GPU box (PowerShell + conda/Miniforge).

  From the repo root (folder with pyproject.toml), in a Miniforge / Anaconda PowerShell prompt:
      powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Cuda cu121
  Cuda tag: cu121 (default) | cu118 | cu124 | cpu  — match `nvidia-smi`.
#>
param([string]$Cuda = "cu121")
$ErrorActionPreference = "Stop"
$EnvName = "ovcbmr"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  Write-Error "conda not found on PATH. Open the 'Miniforge Prompt', or run:  conda init powershell  then reopen."
  exit 1
}

# Load conda's PowerShell hook so `conda activate` works inside this script.
(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression

$exists = (conda env list | ForEach-Object { ($_ -split '\s+')[0] }) -contains $EnvName
if (-not $exists) {
  Write-Host "[setup] creating conda env '$EnvName' (python 3.10)"
  conda create -y -n $EnvName python=3.10
} else {
  Write-Host "[setup] conda env '$EnvName' already exists"
}
conda activate $EnvName

python -m pip install --upgrade pip

Write-Host "[setup] torch (CUDA tag: $Cuda)"
if ($Cuda -eq "cpu") {
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
} else {
  pip install torch==2.4.1 torchvision==0.19.1 --index-url "https://download.pytorch.org/whl/$Cuda"
}

Write-Host "[setup] project + requirements"
pip install -r requirements.txt
pip install -e .

Write-Host "[setup] verify"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "from ovcbmr.models.encoder import load_biomedclip; print('encoder import: ok')"
python tests\test_smoke.py

Write-Host "[setup] DONE. Next: powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1  (see SERVER_SETUP.md)"
