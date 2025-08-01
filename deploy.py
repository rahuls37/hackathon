#!/usr/bin/env python3
"""
Railway Deployment Script for HackRx 6.0 RAG API
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command and return result"""
    print(f"🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {result.stderr}")
        sys.exit(1)
    return result

def check_railway_cli():
    """Check if Railway CLI is installed"""
    try:
        result = run_command("railway --version", check=False)
        if result.returncode == 0:
            print("✅ Railway CLI is installed")
            return True
    except:
        pass
    
    print("❌ Railway CLI not found. Installing...")
    
    # Install Railway CLI based on OS
    if os.name == 'nt':  # Windows
        print("Please install Railway CLI manually:")
        print("1. Visit: https://railway.app/cli")
        print("2. Download and install Railway CLI")
        print("3. Run: railway login")
        sys.exit(1)
    else:  # Unix-like
        run_command("curl -fsSL https://railway.app/install.sh | sh")
    
    return True

def setup_git():
    """Initialize git repository if not exists"""
    if not Path('.git').exists():
        print("📁 Initializing git repository...")
        run_command("git init")
        run_command("git add .")
        run_command('git commit -m "Initial commit for HackRx 6.0 RAG API"')
    else:
        print("✅ Git repository already exists")

def deploy_to_railway():
    """Deploy to Railway"""
    print("🚀 Deploying to Railway...")
    
    # Login check
    result = run_command("railway whoami", check=False)
    if result.returncode != 0:
        print("🔐 Please login to Railway first:")
        run_command("railway login")
    
    # Create new project or link existing
    print("📦 Setting up Railway project...")
    result = run_command("railway project", check=False)
    if result.returncode != 0:
        print("Creating new Railway project...")
        run_command("railway login")
        run_command("railway init")
    
    # Set environment variables
    print("🔧 Setting environment variables...")
    env_vars = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here"),
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "DATABASE_PATH": "hackrx.db",
        "CHROMA_PERSIST_DIRECTORY": "./chroma_db",
        "LOG_LEVEL": "INFO"
    }
    
    for key, value in env_vars.items():
        run_command(f'railway variables set {key}="{value}"')
    
    # Deploy
    print("🚀 Deploying application...")
    run_command("railway up --detach")
    
    # Get domain
    print("🌐 Getting deployment URL...")
    result = run_command("railway domain", check=False)
    if result.returncode == 0:
        print(f"✅ Deployment successful!")
        print(f"🌐 Your API is available at: {result.stdout.strip()}")
        print(f"🔗 Health check: {result.stdout.strip()}/health")
        print(f"📋 API endpoint: {result.stdout.strip()}/hackrx/run")
    else:
        print("⚠️ Domain not set. Creating domain...")
        run_command("railway domain generate")
        result = run_command("railway domain")
        if result.returncode == 0:
            print(f"✅ Deployment successful!")
            print(f"🌐 Your API is available at: {result.stdout.strip()}")

def main():
    """Main deployment function"""
    print("🚀 HackRx 6.0 RAG API Deployment Script")
    print("=" * 50)
    
    # Check prerequisites
    print("1. Checking prerequisites...")
    check_railway_cli()
    
    # Setup git
    print("2. Setting up git repository...")
    setup_git()
    
    # Deploy
    print("3. Deploying to Railway...")
    deploy_to_railway()
    
    print("\n🎉 Deployment complete!")
    print("📋 Next steps:")
    print("   1. Test your API using the provided URL")
    print("   2. Submit the webhook URL to HackRx platform")
    print("   3. Monitor logs: railway logs")

if __name__ == "__main__":
    main()
