$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment not found: $Python"
}

& $Python -u (Join-Path $ProjectRoot "src\dp_dcgan\train_dp_dcgan.py") `
    --data-root (Join-Path $ProjectRoot "data") `
    --output-dir (Join-Path $ProjectRoot "runs\required") `
    --epochs 10 `
    --batch-size 64 `
    --image-size 32 `
    --latent-size 64 `
    --generator-features 32 `
    --discriminator-features 32 `
    --target-digit 8 `
    --noise-multiplier 1.0 `
    --max-grad-norm 1.0 `
    --delta 1e-5 `
    --seed 2026 `
    --workers 0 `
    --device cuda

if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
