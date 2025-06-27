# 🎯 ML Optimization Framework - Tutorial Launcher
# Easy way to start learning Optuna

Write-Host "🎯 ML Optimization Framework - Tutorial Launcher" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

Write-Host "`n📚 Choose your learning path:" -ForegroundColor Yellow
Write-Host "1. 🚀 Start Interactive Dashboard (Recommended)" -ForegroundColor Green
Write-Host "2. 📖 Run Colleague Tutorial Example" -ForegroundColor Blue
Write-Host "3. ⚡ Quick 30-second Demo" -ForegroundColor Magenta
Write-Host "4. 📋 View Documentation" -ForegroundColor White

$choice = Read-Host "`nEnter your choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host "`n🚀 Starting Interactive Dashboard..." -ForegroundColor Green
        Write-Host "This will:" -ForegroundColor Gray
        Write-Host "  ✅ Create 6 optimization studies" -ForegroundColor Gray
        Write-Host "  ✅ Start Optuna Dashboard" -ForegroundColor Gray
        Write-Host "  ✅ Open browser to http://localhost:8080" -ForegroundColor Gray
        Write-Host "`nStarting Docker container..." -ForegroundColor Yellow
        
        docker-compose up -d --build
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Container started successfully!" -ForegroundColor Green
            Write-Host "⏳ Waiting for demos to complete (2-3 minutes)..." -ForegroundColor Yellow
            
            Start-Sleep 10
            Write-Host "🌐 Opening dashboard..." -ForegroundColor Cyan
            Start-Process "http://localhost:8080"
            
            Write-Host "`n🎉 Dashboard is ready!" -ForegroundColor Green
            Write-Host "📍 URL: http://localhost:8080" -ForegroundColor Cyan
            Write-Host "📊 Explore 6 different optimization studies" -ForegroundColor White
        } else {
            Write-Host "❌ Failed to start container" -ForegroundColor Red
        }
    }
    
    "2" {
        Write-Host "`n📖 Running Colleague Tutorial Example..." -ForegroundColor Blue
        Write-Host "This is a complete Python example your colleagues can run" -ForegroundColor Gray
        Write-Host "`nStarting tutorial..." -ForegroundColor Yellow
        
        python optuna_colleague_example.py
    }
    
    "3" {
        Write-Host "`n⚡ Running Quick 30-second Demo..." -ForegroundColor Magenta
        Write-Host "Perfect for a quick taste of Optuna" -ForegroundColor Gray
        
        python optuna_colleague_example.py quick
    }
    
    "4" {
        Write-Host "`n📋 Opening Documentation..." -ForegroundColor White
        Write-Host "Available documentation:" -ForegroundColor Gray
        Write-Host "  📖 docs/tutorial.md - Complete Optuna tutorial" -ForegroundColor Gray
        Write-Host "  📋 docs/setup.md - Setup instructions" -ForegroundColor Gray
        Write-Host "  📊 docs/usage.md - Dashboard usage guide" -ForegroundColor Gray
        Write-Host "  🔬 docs/studies.md - Study explanations" -ForegroundColor Gray
        Write-Host "  🔧 docs/api.md - Technical details" -ForegroundColor Gray
        
        $docChoice = Read-Host "`nWhich document to open? (tutorial/setup/usage/studies/api)"
        
        switch ($docChoice) {
            "tutorial" { Start-Process "docs/tutorial.md" }
            "setup" { Start-Process "docs/setup.md" }
            "usage" { Start-Process "docs/usage.md" }
            "studies" { Start-Process "docs/studies.md" }
            "api" { Start-Process "docs/api.md" }
            default { 
                Write-Host "Opening tutorial by default..." -ForegroundColor Yellow
                Start-Process "docs/tutorial.md" 
            }
        }
    }
    
    default {
        Write-Host "`n🚀 Starting Interactive Dashboard (default choice)..." -ForegroundColor Green
        docker-compose up -d --build
        Start-Sleep 10
        Start-Process "http://localhost:8080"
    }
}

Write-Host "`n🎓 Learning Resources:" -ForegroundColor Yellow
Write-Host "📖 Complete Tutorial: docs/tutorial.md" -ForegroundColor White
Write-Host "🎯 Colleague Example: optuna_colleague_example.py" -ForegroundColor White
Write-Host "📊 Interactive Dashboard: http://localhost:8080" -ForegroundColor White
Write-Host "📚 Official Docs: https://optuna.readthedocs.io/" -ForegroundColor White

Write-Host "`n✅ Happy learning with Optuna! 🎯" -ForegroundColor Green
