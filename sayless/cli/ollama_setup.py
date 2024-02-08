import platform
import subprocess
import sys
import shutil
import os
import requests
import time
from rich.console import Console

console = Console()

class OllamaSetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.ollama_url = "http://localhost:11434"

    def get_install_command(self):
        """Get the installation command based on the OS"""
        if self.system == "darwin":  # macOS
            return "brew install ollama"
        elif self.system == "linux":
            if shutil.which("apt-get"):  # Debian/Ubuntu
                return "curl -fsSL https://ollama.ai/install.sh | sh"
            elif shutil.which("dnf"):  # Fedora
                return "curl -fsSL https://ollama.ai/install.sh | sh"
            elif shutil.which("pacman"):  # Arch
                return "curl -fsSL https://ollama.ai/install.sh | sh"
        elif self.system == "windows":
            return "Please download Ollama from https://ollama.ai/download"
        
        return None

    def is_ollama_installed(self):
        """Check if Ollama is installed"""
        return shutil.which('ollama') is not None

    def install_ollama(self):
        """Install Ollama based on the OS"""
        install_cmd = self.get_install_command()
        
        if not install_cmd:
            console.print("[red]Error: Unsupported operating system[/red]")
            console.print("[yellow]Please visit https://ollama.ai/download for manual installation[/yellow]")
            return False

        if self.system == "windows":
            console.print("[yellow]Please install Ollama manually:[/yellow]")
            console.print(f"[blue]{install_cmd}[/blue]")
            return False

        try:
            console.print(f"[yellow]Installing Ollama using command: {install_cmd}[/yellow]")
            if "curl" in install_cmd:
                # For curl-based installations
                subprocess.run(install_cmd, shell=True, check=True)
            else:
                # For package manager installations
                subprocess.run(install_cmd.split(), check=True)
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error installing Ollama: {str(e)}[/red]")
            return False

    def is_ollama_running(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_ollama(self):
        """Start Ollama service"""
        if self.system == "windows":
            console.print("[yellow]Please start Ollama manually using the Windows application[/yellow]")
            return False

        try:
            # Start ollama serve in the background
            if self.system == "darwin" or self.system == "linux":
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               start_new_session=True)
                return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error starting Ollama: {str(e)}[/red]")
            return False

    def wait_for_ollama(self, timeout=30):
        """Wait for Ollama to be ready"""
        console.print("[yellow]Waiting for Ollama to start...[/yellow]")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ollama_running():
                console.print("[green]Ollama is running![/green]")
                return True
            time.sleep(1)
        return False

    def ensure_model_exists(self, model_name):
        """Ensure the specified model is pulled"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            tags = response.json().get('models', [])
            
            if not any(tag.get('name') == model_name for tag in tags):
                console.print(f"[yellow]Pulling model {model_name}...[/yellow]")
                subprocess.run(['ollama', 'pull', model_name], check=True)
                console.print(f"[green]Successfully pulled {model_name}[/green]")
            return True
        except (requests.exceptions.RequestException, subprocess.CalledProcessError) as e:
            console.print(f"[red]Error ensuring model exists: {str(e)}[/red]")
            return False

def ensure_ollama_ready(model_name="llama2"):
    """Main function to ensure Ollama is installed, running and model is ready"""
    setup = OllamaSetup()

    # Check if Ollama is installed
    if not setup.is_ollama_installed():
        console.print("[yellow]Ollama is not installed. Installing now...[/yellow]")
        if not setup.install_ollama():
            console.print("[red]Failed to install Ollama[/red]")
            sys.exit(1)

    # Check if Ollama is running
    if not setup.is_ollama_running():
        console.print("[yellow]Ollama is not running. Starting now...[/yellow]")
        if not setup.start_ollama():
            console.print("[red]Failed to start Ollama[/red]")
            console.print("[yellow]Please start Ollama manually and try again[/yellow]")
            sys.exit(1)

        # Wait for Ollama to be ready
        if not setup.wait_for_ollama():
            console.print("[red]Ollama failed to start in time[/red]")
            sys.exit(1)

    # Ensure model exists
    if not setup.ensure_model_exists(model_name):
        console.print(f"[red]Failed to ensure model {model_name} exists[/red]")
        sys.exit(1)

    return True 