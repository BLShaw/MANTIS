@echo off
REM MANTIS Setup Script
REM Downloads required files: KoboldCPP (No-AVX) and Qwen 0.5B Model

echo ============================================================
echo   MANTIS: Downloading Required Files
echo ============================================================
echo.

REM Check if files already exist
if exist "server.exe" (
    echo [SKIP] server.exe already exists
) else (
    echo [1/2] Downloading KoboldCPP (No-AVX build)...
    echo       This may take a few minutes (~410 MB)
    curl -L -o "server.exe" "https://github.com/LostRuins/koboldcpp/releases/download/v1.107/koboldcpp-oldpc.exe"
    if errorlevel 1 (
        echo [ERROR] Failed to download KoboldCPP
    ) else (
        echo [OK] server.exe downloaded
    )
)

echo.

if exist "qwen2.5-0.5b-instruct-q4_k_m.gguf" (
    echo [SKIP] Model file already exists
) else (
    echo [2/2] Downloading Qwen 2.5 0.5B Model...
    echo       This may take a few minutes (~463 MB)
    curl -L -o "qwen2.5-0.5b-instruct-q4_k_m.gguf" "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"
    if errorlevel 1 (
        echo [ERROR] Failed to download model
    ) else (
        echo [OK] Model downloaded
    )
)

echo.
echo ============================================================
echo   Download Complete! 
echo   Next: Run 'python src/ingest.py' to index your PDFs
echo ============================================================
pause
