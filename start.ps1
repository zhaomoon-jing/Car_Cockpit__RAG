# 汽车座舱RAG系统启动脚本 (PowerShell版本)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "汽车座舱RAG系统启动脚本" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "错误: 未找到Python，请先安装Python 3.8+" -ForegroundColor Red
    Write-Host "按任意键退出..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# 检查依赖文件
if (-not (Test-Path "requirements.txt")) {
    Write-Host "错误: 未找到requirements.txt文件" -ForegroundColor Red
    Write-Host "按任意键退出..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host ""
Write-Host "1. 检查依赖包..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: 依赖安装失败，请手动安装" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "2. 检查数据目录结构..." -ForegroundColor Yellow

# 创建必要的目录
$directories = @(
    "data\raw",
    "data\chunks", 
    "data\qa_train",
    "vector_store\faiss_index"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "✓ 创建目录: $dir" -ForegroundColor Green
    } else {
        Write-Host "✓ 目录已存在: $dir" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "3. 启动Gradio Web界面..." -ForegroundColor Yellow
Write-Host "请访问: http://localhost:7860" -ForegroundColor Cyan
Write-Host ""

# 启动应用
python main.py

Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")