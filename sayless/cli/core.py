#!/usr/bin/env python3

import subprocess
import sys
import typer
from rich.console import Console
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from typing import Optional, Tuple
import os
import time
from .config import Config
from .ai_providers import OllamaProvider, OpenAIProvider
import datetime
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
import asyncio
from .embeddings import CommitEmbeddings
from .git_ops import create_branch, list_branches, run_git_command
from .github_ops import create_pr, list_prs

app = typer.Typer(help="AI Git Copilot / Autopilot")
console = Console()
settings = Config()

VALID_PROVIDERS = ["openai", "ollama"]
MAX_COMMIT_MESSAGE_LENGTH = 72
DEBUG_MODE = os.getenv('SAYLESS_DEBUG', 'false').lower() == 'true'

def truncate_commit_message(message: str) -> str:
    """Truncate commit message to max length if needed"""
    if len(message) <= MAX_COMMIT_MESSAGE_LENGTH:
        return message
    return message[:MAX_COMMIT_MESSAGE_LENGTH-3] + "..."

def validate_git_repository():
    """Validate git repository and provide helpful error messages"""
    if not check_git_repo():
        panel = Panel(
            "[red]Not in a git repository[/red]\n\n"
            "[yellow]This command requires a git repository.[/yellow]\n"
            "To initialize a new git repository:\n"
            "[blue]  git init[/blue]",
            title="Git Repository Required",
            border_style="red"
        )
        console.print(panel)
        sys.exit(1)

def show_welcome_message():
    """Show a welcome message with the current configuration"""
    provider = settings.get_provider()
    model = settings.get_model()
    
    table = Table(show_header=False, box=None, border_style="dim")
    table.add_row("[bold cyan]Sayless[/bold cyan] ", "AI Git Copilot / Autopilot")
    table.add_row("Provider", f"[green]{provider}[/green] {'🌐 (cloud)' if provider == 'openai' else '🏠 (local)'}")
    table.add_row("Model", f"[green]{model}[/green]")
    
    panel = Panel(table, border_style="cyan", title="[bold]Ready[/bold]")
    console.print(panel)
    console.print()

def ensure_openai_configured():
    """Ensure OpenAI is properly configured or guide the user"""
    if settings.get_provider() != 'openai':
        return
        
    api_key = settings.get_openai_api_key()
    if not api_key:
        panel = Panel(
            "[bold red]OpenAI API Key Required[/bold red]\n\n"
            "[bold]You have two options:[/bold]\n\n"
            "[bold cyan]1. Configure OpenAI (Recommended):[/bold cyan]\n"
            "   Set your API key using one of these methods:\n"
            "   [blue]a) Environment variable:[/blue]\n"
            "      export OPENAI_API_KEY=your_api_key\n"
            "   [blue]b) Configuration command:[/blue]\n"
            "      sayless config --openai-key YOUR_API_KEY\n"
            "   [blue]c) Quick switch command:[/blue]\n"
            "      sayless switch openai --key YOUR_API_KEY\n\n"
            "[bold yellow]2. Switch to Ollama (Local AI):[/bold yellow]\n"
            "   [blue]a) Install Ollama first:[/blue]\n"
            "      Visit https://ollama.ai\n"
            "   [blue]b) Then switch to Ollama:[/blue]\n"
            "      sayless switch ollama",
            title="Configuration Required",
            border_style="red"
        )
        console.print(panel)
        sys.exit(1)

def show_config_status():
    """Show current configuration status in a nice table"""
    table = Table(title="Current Configuration", show_header=False, title_style="bold cyan", border_style="cyan")
    provider = settings.get_provider()
    has_openai_key = bool(settings.get_openai_api_key())
    has_github_token = bool(settings.get_github_token())
    
    table.add_row("Provider", f"[green]{provider}[/green] {'(default)' if provider == 'openai' else '(local AI)'}")
    table.add_row("Model", f"[green]{settings.get_model()}[/green]")
    if provider == 'openai':
        table.add_row("OpenAI API Key", f"[{'green' if has_openai_key else 'red'}]{'configured' if has_openai_key else 'not configured'}[/{'green' if has_openai_key else 'red'}]")
    table.add_row("GitHub Token", f"[{'green' if has_github_token else 'red'}]{'configured' if has_github_token else 'not configured'}[/{'green' if has_github_token else 'red'}]")
    
    console.print(table)
    console.print()

def check_git_repo():
    """Check if current directory is a git repository"""
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def has_staged_changes():
    """Check if there are staged changes to commit"""
    try:
        diff = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], 
                                     stderr=subprocess.DEVNULL).decode('utf-8')
        return bool(diff.strip())
    except subprocess.CalledProcessError:
        return False

def get_staged_diff():
    """Get the diff of staged changes"""
    if not check_git_repo():
        panel = Panel(
            "[red]Not in a git repository[/red]\n\n"
            "[yellow]Please run this command inside a git repository.[/yellow]\n"
            "To initialize a new git repository:\n"
            "[blue]  git init[/blue]",
            title="Error",
            border_style="red"
        )
        console.print(panel)
        sys.exit(1)

    if not has_staged_changes():
        panel = Panel(
            "[red]No staged changes found[/red]\n\n"
            "[yellow]Please stage your changes first:[/yellow]\n"
            "[blue]git add <files>[/blue]\n"
            "# or to stage all changes:\n"
            "[blue]git add .[/blue]",
            title="Error",
            border_style="red"
        )
        console.print(panel)
        sys.exit(1)

    try:
        # Get the diff of staged changes
        diff = subprocess.check_output(
            ['git', 'diff', '--cached', '--no-color'],
            stderr=subprocess.PIPE
        ).decode('utf-8')
        
        if not diff.strip():
            console.print(Panel("[red]Staged changes are empty[/red]", title="Error", border_style="red"))
            sys.exit(1)
            
        return diff
    except subprocess.CalledProcessError as e:
        console.print(Panel(f"[red]Failed to get staged changes\nDetails: {e.stderr.decode('utf-8')}[/red]", title="Error", border_style="red"))
        sys.exit(1)

def create_commit(message: str):
    """Create a commit with the given message"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating commit...", total=None)
        try:
            subprocess.run(['git', 'commit', '-m', message], check=True)
            progress.update(task, completed=True)
            console.print("\n[bold green]✓[/bold green] Commit created successfully!")
        except subprocess.CalledProcessError as e:
            progress.update(task, completed=True)
            console.print(Panel(f"[red]Failed to create commit\nDetails: {str(e)}[/red]", title="Error", border_style="red"))
            sys.exit(1)

def get_ai_provider():
    """Get the configured AI provider"""
    provider = settings.get_provider()
    
    if provider == 'openai':
        api_key = settings.get_openai_api_key()
        if not api_key:
            ensure_openai_configured()
        return OpenAIProvider(api_key)
    else:
        return OllamaProvider()

def generate_commit_message(diff: str, provider, model: str) -> str:
    """Generate commit message with fallback handling"""
    try:
        return provider.generate_commit_message(diff, model)
    except Exception as e:
        if isinstance(provider, OpenAIProvider):
            console.print("\n[yellow]OpenAI failed. Trying Ollama as fallback...[/yellow]")
            try:
                with console.status("[bold yellow]Initializing Ollama fallback...[/bold yellow]"):
                    fallback_provider = OllamaProvider()
                return fallback_provider.generate_commit_message(diff, "llama2")
            except Exception:
                console.print(Panel(f"[red]Both OpenAI and Ollama failed.\nOriginal error: {str(e)}[/red]", title="Error", border_style="red"))
                sys.exit(1)
        raise

def get_commit_range(since=None, until=None):
    """Get commit range based on dates"""
    cmd = ['git', 'log', '--no-merges', '--format=%H']
    
    if since:
        cmd.append(f'--since={since}')
    if until:
        cmd.append(f'--until={until}')
    
    try:
        commits = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode('utf-8').strip().split('\n')
        return commits if commits and commits[0] else []
    except subprocess.CalledProcessError:
        return []

def get_commit_details(commit_hash):
    """Get details for a specific commit"""
    try:
        details = subprocess.check_output(
            ['git', 'show', '--no-color', '--format=%s%n%b', commit_hash],
            stderr=subprocess.PIPE
        ).decode('utf-8').strip()
        return details
    except subprocess.CalledProcessError:
        return None

def generate_summary(commits, provider, model):
    """Generate a summary of changes from commits"""
    if not commits:
        return "No changes found in the specified period."
    
    # Get all commit messages and diffs
    commit_details = []
    for commit in commits[:10]:  # Limit to last 10 commits for reasonable context
        details = get_commit_details(commit)
        if details:
            commit_details.append(details)
    
    commit_text = "\n\n".join(commit_details)
    
    prompt = f"""Based on these recent commits, generate a concise summary of changes.
Group related changes together and highlight major updates.
Focus on the business impact and key features/fixes.

Recent commits:

{commit_text}

Generate a summary in this format:
## Major Changes
- (grouped major changes)

## Features
- (new features)

## Fixes & Improvements
- (bug fixes and improvements)

Keep each bullet point concise and clear."""

    try:
        return provider.generate_commit_message(prompt, model)
    except Exception as e:
        if isinstance(provider, OpenAIProvider):
            console.print("[yellow]OpenAI failed, trying Ollama fallback...[/yellow]")
            try:
                fallback_provider = OllamaProvider()
                return fallback_provider.generate_commit_message(prompt, "llama2")
            except Exception:
                raise Exception(f"Both providers failed. Original error: {str(e)}")
        raise

def parse_time_interval(interval):
    """Parse time interval string into timedelta"""
    if not interval:
        return relativedelta(days=1)  # Default to last 24 hours
        
    units = {
        'd': 'days',
        'w': 'weeks',
        'm': 'months',
        'y': 'years',
        'h': 'hours'
    }
    
    unit = interval[-1].lower()
    try:
        value = int(interval[:-1])
        if unit in units:
            return relativedelta(**{units[unit]: value})
    except ValueError:
        pass
    
    # If parsing fails, try to parse as a date
    try:
        since_date = parse_date(interval)
        return since_date
    except ValueError:
        raise ValueError(
            "Invalid time interval. Use format: <number><unit> or a date.\n"
            "Units: h (hours), d (days), w (weeks), m (months), y (years)\n"
            "Example: 2w (2 weeks), 3m (3 months), 2023-01-01"
        )

def format_date(date_str):
    """Convert date string to human readable format"""
    try:
        if isinstance(date_str, str):
            date = parse_date(date_str)
        else:
            date = date_str
        return date.strftime("%A, %B %d, %Y")  # e.g., "Tuesday, May 15, 2023"
    except:
        return date_str

def get_commit_details_for_hash(commit_hash: str) -> Tuple[str, str, str]:
    """Get commit message, diff and date for a commit hash"""
    try:
        # Get list of recent commits for context
        recent_commits = run_git_command(
            ['log', '--oneline', '-n', '5'],
            check=False
        ).stdout.strip().split('\n')
        
        # Try to find the commit in recent history
        matching_commits = [c for c in recent_commits if commit_hash in c.split()[0]]
        if matching_commits:
            # Found a matching commit
            full_hash = matching_commits[0].split()[0]
        else:
            # Try to resolve the commit hash
            try:
                full_hash = run_git_command(['rev-parse', '--verify', commit_hash]).stdout.strip()
            except:
                commits_context = "\nRecent commits:"
                for commit in recent_commits:
                    if commit:  # Check if line is not empty
                        commits_context += f"\n{commit}"
                raise ValueError(
                    f"Invalid commit hash: '{commit_hash}'\n"
                    "Please provide a valid commit hash or prefix."
                    f"{commits_context}"
                )
        
        # Get commit message, date, and stats
        commit_info = run_git_command(
            ['show', '-s', '--format=%B%n%aI', full_hash]
        ).stdout.strip().split('\n')
        
        commit_message = '\n'.join(commit_info[:-1])
        commit_date = commit_info[-1]
        
        # Get commit diff with stats
        diff_stats = run_git_command(
            ['show', '--stat', full_hash]
        ).stdout.strip()
        
        # Get detailed diff
        diff_detail = run_git_command(
            ['show', '--no-color', '--format=', full_hash]
        ).stdout.strip()
        
        # Combine stats and details for better context
        diff = f"Stats:\n{diff_stats}\n\nDetails:\n{diff_detail}"
        
        return commit_message, diff, commit_date
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        raise Exception(f"Failed to get commit details: {error_msg}")

async def index_commit(commit_hash: str, progress=None):
    """Index a single commit"""
    try:
        message, diff, date = get_commit_details_for_hash(commit_hash)
        embeddings = CommitEmbeddings()
        tags = embeddings.get_commit_tags(message, diff)
        await embeddings.add_commit(commit_hash, message, diff, date, tags)
        return True
    except Exception as e:
        if progress:
            console.print(f"[yellow]Failed to index commit {commit_hash}: {str(e)}[/yellow]")
        return False

@app.command()
def switch(
    provider: str = typer.Argument(..., help="AI provider to use (openai or ollama)"),
    key: Optional[str] = typer.Option(None, "--key", help="OpenAI API key (required for OpenAI)"),
    model: Optional[str] = typer.Option(None, "--model", help="Model to use (optional)"),
):
    """Quickly switch between AI providers (OpenAI/Ollama)"""
    
    # Validate provider
    provider = provider.lower()
    if provider not in VALID_PROVIDERS:
        console.print(Panel(
            f"[red]Invalid provider '{provider}'[/red]\n"
            f"[yellow]Valid providers are: {', '.join(VALID_PROVIDERS)}[/yellow]",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Switching to {provider}...", total=None)
        
        if provider == "openai":
            # Check for API key
            api_key = key or os.getenv("OPENAI_API_KEY") or settings.get_openai_api_key()
            if not api_key:
                progress.update(task, completed=True)
                console.print(Panel(
                    "[red]OpenAI API key required[/red]\n"
                    "[yellow]Provide it using one of these methods:[/yellow]\n"
                    "[blue]1. With the switch command:[/blue]\n"
                    "   sayless switch openai --key YOUR_API_KEY\n"
                    "[blue]2. Environment variable:[/blue]\n"
                    "   export OPENAI_API_KEY=your_api_key",
                    title="Error",
                    border_style="red"
                ))
                sys.exit(1)
            
            # Configure OpenAI
            settings.set_openai_api_key(api_key)
            settings.set_provider('openai')
            if not model:
                settings.set_model('gpt-4')  # Set default OpenAI model
            
        elif provider == "ollama":
            settings.set_provider('ollama')
            if not model:
                settings.set_model('llama2')  # Set default Ollama model
        
        if model:
            settings.set_model(model)
        
        time.sleep(0.5)  # Add a small delay for better UX
        progress.update(task, completed=True)
    
    # Show success message and new configuration
    console.print(f"\n[bold green]✓[/bold green] Successfully switched to {provider}!")
    show_config_status()

@app.command()
def config(
    openai_key: Optional[str] = typer.Option(None, "--openai-key", help="Set OpenAI API key"),
    github_token: Optional[str] = typer.Option(None, "--github-token", help="Set GitHub token"),
    use_openai: bool = typer.Option(False, "--use-openai", help="Use OpenAI (default)"),
    use_ollama: bool = typer.Option(False, "--use-ollama", help="Use Ollama (local AI)"),
    model: Optional[str] = typer.Option(None, "--model", help="Set the model to use"),
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
):
    """Configure the AI provider and settings"""
    if show:
        show_config_status()
        provider = settings.get_provider()
        has_openai_key = bool(settings.get_openai_api_key())
        has_github_token = bool(settings.get_github_token())
        
        if not has_openai_key and provider == 'openai':
            console.print(Panel(
                "[yellow]OpenAI API key not configured[/yellow]\n"
                "[blue]Quick setup:[/blue]\n"
                "  sayless switch openai --key YOUR_API_KEY",
                title="Warning",
                border_style="yellow"
            ))
        
        if not has_github_token:
            console.print(Panel(
                "[yellow]GitHub token not configured[/yellow]\n"
                "[blue]Quick setup:[/blue]\n"
                "  sayless config --github-token YOUR_TOKEN",
                title="Warning",
                border_style="yellow"
            ))
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Updating configuration...", total=None)

        if openai_key:
            settings.set_openai_api_key(openai_key)
            settings.set_provider('openai')  # Automatically switch to OpenAI when key is provided
        
        if github_token:
            settings.set_github_token(github_token)

        if use_openai and use_ollama:
            progress.update(task, completed=True)
            console.print(Panel("[red]Cannot use both OpenAI and Ollama at the same time[/red]", title="Error", border_style="red"))
            sys.exit(1)

        if use_openai:
            settings.set_provider('openai')
            if not settings.get_openai_api_key():
                progress.update(task, completed=True)
                console.print(Panel(
                    "[yellow]OpenAI API key not configured[/yellow]\n"
                    "[blue]Quick setup:[/blue]\n"
                    "  sayless switch openai --key YOUR_API_KEY",
                    title="Warning",
                    border_style="yellow"
                ))
        elif use_ollama:
            settings.set_provider('ollama')

        if model:
            settings.set_model(model)

        time.sleep(0.5)  # Add a small delay for better UX
        progress.update(task, completed=True)

    console.print("\n[bold green]✓[/bold green] Configuration updated successfully!")
    show_config_status()

@app.command()
def generate(
    preview: bool = typer.Option(False, help="Preview the commit message without creating the commit"),
    auto_add: bool = typer.Option(False, "-a", help="Automatically run 'git add .' before generating commit"),
):
    """Generate a commit message for staged changes and create the commit"""
    _generate_command(preview, auto_add)

@app.command("g")
def generate_alias(
    preview: bool = typer.Option(False, help="Preview the commit message without creating the commit"),
    auto_add: bool = typer.Option(False, "-a", help="Automatically run 'git add .' before generating commit"),
):
    """Generate a commit message for staged changes and create the commit (alias for generate)"""
    _generate_command(preview, auto_add)

def _generate_command(preview: bool, auto_add: bool):
    """Internal function that implements the generate command logic"""
    show_welcome_message()
    ensure_openai_configured()
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
    
    message = None
    commit_hash = None
    
    with progress:
        try:
            # Check git repository
            task_check = progress.add_task("Checking git repository...", total=None)
            if not check_git_repo():
                progress.update(task_check, visible=False)
                panel = Panel(
                    "[red]Not in a git repository[/red]\n\n"
                    "[yellow]Please run this command inside a git repository.[/yellow]\n"
                    "To initialize a new git repository:\n"
                    "[blue]  git init[/blue]",
                    title="Error",
                    border_style="red"
                )
                console.print(panel)
                sys.exit(1)
            progress.update(task_check, completed=True)
            
            # Auto-add changes if requested
            task_add = None
            if auto_add:
                task_add = progress.add_task("Adding all changes...", total=None)
                try:
                    subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
                    progress.update(task_add, completed=True)
                    console.print("[green]✓[/green] Added all changes")
                except subprocess.CalledProcessError as e:
                    progress.update(task_add, visible=False)
                    console.print(Panel(f"[red]Failed to add changes: {e.stderr.decode('utf-8')}[/red]", title="Error", border_style="red"))
                    sys.exit(1)
            
            # Get staged changes
            task_diff = progress.add_task("Getting staged changes...", total=None)
            diff = get_staged_diff()
            progress.update(task_diff, completed=True)
            
            # Generate message
            task_gen = progress.add_task("Generating commit message...", total=None)
            provider = get_ai_provider()
            model = settings.get_model()
            
            try:
                message = provider.generate_commit_message(diff, model)
                progress.update(task_gen, completed=True)
            except Exception as e:
                if isinstance(provider, OpenAIProvider):
                    progress.update(task_gen, description="OpenAI failed, trying Ollama fallback...")
                    try:
                        fallback_provider = OllamaProvider()
                        message = fallback_provider.generate_commit_message(diff, "llama2")
                        progress.update(task_gen, completed=True)
                    except Exception:
                        progress.update(task_gen, visible=False)
                        console.print(Panel(
                            f"[red]Both OpenAI and Ollama failed.\nOriginal error: {str(e)}[/red]",
                            title="Error",
                            border_style="red"
                        ))
                        sys.exit(1)
                else:
                    progress.update(task_gen, visible=False)
                    raise
        except Exception as e:
            # Ensure all tasks are properly marked as completed or hidden
            for task_id in progress.task_ids:
                progress.update(task_id, visible=False)
            raise e

    if message:
        # Show the generated message in a panel
        console.print(Panel(
            f"[yellow]{message}[/yellow]",
            title="Generated Commit Message",
            border_style="green"
        ))

        if preview:
            console.print("\n[blue]Preview mode: Commit not created[/blue]")
            return

        # Ask for confirmation with styled prompt
        if typer.confirm("\n💭 Do you want to create the commit with this message?"):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as commit_progress:
                try:
                    # Create commit
                    task_commit = commit_progress.add_task("Creating commit...", total=None)
                    result = subprocess.run(
                        ['git', 'commit', '-m', message],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    commit_hash = subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'],
                        stderr=subprocess.PIPE
                    ).decode('utf-8').strip()
                    
                    commit_progress.update(task_commit, completed=True)
                    console.print("\n[bold green]✓[/bold green] Commit created successfully!")
                    
                    # Index the commit
                    task_index = commit_progress.add_task("Indexing commit for search...", total=None)
                    try:
                        success = asyncio.run(index_commit(commit_hash))
                        if success:
                            commit_progress.update(task_index, completed=True)
                            console.print("[green]✓[/green] Commit indexed for search")
                        else:
                            commit_progress.update(task_index, visible=False)
                            console.print("\n[yellow]Note: Failed to index commit[/yellow]")
                    except Exception as e:
                        commit_progress.update(task_index, visible=False)
                        console.print(f"\n[yellow]Note: Failed to index commit: {str(e)}[/yellow]")
                    
                except subprocess.CalledProcessError as e:
                    commit_progress.update(task_commit, visible=False)
                    console.print(Panel(
                        f"[red]Failed to create commit\nDetails: {e.stderr}[/red]",
                        title="Error",
                        border_style="red"
                    ))
                    sys.exit(1)
                except Exception as e:
                    # Ensure all tasks are properly marked as completed or hidden
                    for task_id in commit_progress.task_ids:
                        commit_progress.update(task_id, visible=False)
                    raise e
        else:
            console.print("\n[yellow]Commit cancelled[/yellow]")

@app.command()
def summary(
    commit_hash: str = typer.Argument(..., help="The commit hash or prefix to summarize"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show a more detailed analysis"),
):
    """Generate an AI-powered summary of a specific commit"""
    show_welcome_message()
    ensure_openai_configured()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing commit...", total=None)
        
        try:
            # Get commit details
            message, diff, date = get_commit_details_for_hash(commit_hash)
            
            # Extract stats for better context
            stats_section = diff.split('Stats:')[1].split('Details:')[0].strip()
            details_section = diff.split('Details:')[1].strip()
            
            # Prepare the prompt based on detail level
            if detailed:
                prompt = f"""As an AI Git Copilot, analyze this commit and provide a detailed, practical summary.

Context:
- Commit Message: {message}
- Changed Files: {stats_section}

Changes:
{details_section}

Provide a clear, developer-focused analysis in this format:

<summary>
Write a clear, practical explanation of what changed and why. Focus on real-world impact.
Example: "Updated the welcome message to better reflect the tool's expanded capabilities as a Git Copilot/Autopilot."
</summary>

<impact>
- List the main effects of these changes
- Focus on practical implications
- Note any workflow changes
</impact>

<details>
- List key technical changes
- Note important file modifications
- Highlight any considerations
</details>"""
            else:
                prompt = f"""As an AI Git Copilot, provide a clear, practical summary of this commit.

Context:
- Commit Message: {message}
- Changed Files: {stats_section}

Changes:
{details_section}

Create a natural, developer-focused summary that:
1. Explains what changed in practical terms
2. Why it was changed (real-world context)
3. How it affects usage/workflow

Example good summary:
"Updated the application's welcome message from 'AI Commit Message Generator' to 'AI Git Copilot / Autopilot' to better reflect its expanded capabilities in automating Git workflows."

Keep it clear and practical, focusing on what developers need to know."""

            # Get AI summary with fallback
            try:
                provider = get_ai_provider()
                summary_text = provider.generate_commit_message(prompt, settings.get_model())
            except Exception as e:
                if "Connection" in str(e) and isinstance(provider, OpenAIProvider):
                    progress.update(task, description="OpenAI connection failed, using local Ollama...")
                    fallback_provider = OllamaProvider()
                    summary_text = fallback_provider.generate_commit_message(prompt, "llama2")
                else:
                    raise
            
            progress.update(task, completed=True)
            
            # Format date
            formatted_date = format_date(parse_date(date))
            
            if detailed:
                # Parse structured response
                try:
                    summary_part = summary_text.split("<summary>")[1].split("</summary>")[0].strip()
                    impact_part = summary_text.split("<impact>")[1].split("</impact>")[0].strip()
                    details_part = summary_text.split("<details>")[1].split("</details>")[0].strip()
                    
                    # Display detailed analysis with stats
                    console.print(Panel(
                        "\n".join([
                            f"[bold white]{message}[/bold white]",
                            "",
                            "[bold cyan]Summary[/bold cyan]",
                            f"[white]{summary_part}[/white]",
                            "",
                            "[bold cyan]Impact[/bold cyan]",
                            f"[white]{impact_part}[/white]",
                            "",
                            "[bold cyan]Technical Details[/bold cyan]",
                            f"[white]{details_part}[/white]",
                            "",
                            "[bold cyan]Files Changed[/bold cyan]",
                            f"[white]{stats_section}[/white]"
                        ]),
                        title=f"[yellow]commit {commit_hash[:8]} • {formatted_date}[/yellow]",
                        border_style="yellow",
                        padding=(1, 2)
                    ))
                except:
                    # Fallback to simple display if parsing fails
                    console.print(Panel(
                        "\n".join([
                            f"[bold white]{message}[/bold white]",
                            "",
                            f"[white]{summary_text}[/white]",
                            "",
                            "[bold cyan]Files Changed[/bold cyan]",
                            f"[white]{stats_section}[/white]"
                        ]),
                        title=f"[yellow]commit {commit_hash[:8]} • {formatted_date}[/yellow]",
                        border_style="yellow",
                        padding=(1, 2)
                    ))
            else:
                # Simple summary display with stats
                console.print(Panel(
                    "\n".join([
                        f"[bold white]{message}[/bold white]",
                        "",
                        f"[white]{summary_text}[/white]",
                        "",
                        "[bold cyan]Files Changed[/bold cyan]",
                        f"[white]{stats_section}[/white]"
                    ]),
                    title=f"[yellow]commit {commit_hash[:8]} • {formatted_date}[/yellow]",
                    border_style="yellow",
                    padding=(1, 2)
                ))
            
        except Exception as e:
            progress.update(task, visible=False)
            if "Connection" in str(e):
                console.print(Panel(
                    "[red]Failed to connect to AI services. Make sure either:\n" +
                    "• Your internet connection is working (for OpenAI)\n" +
                    "• Ollama is running (for local AI)[/red]",
                    title="⚠️ Connection Error",
                    border_style="red"
                ))
            else:
                console.print(Panel(
                    f"[red]{str(e)}[/red]",
                    title="⚠️ Error",
                    border_style="red"
                ))
            sys.exit(1)

@app.command()
def since(
    interval: str = typer.Argument(..., help="Time interval (e.g., 2w, 3m, 2023-01-01)"),
    until: str = typer.Option(None, help="End date (default: now)"),
    preview: bool = typer.Option(False, help="Preview the summary without saving"),
    save: bool = typer.Option(False, help="Save the summary to a file"),
):
    """Generate a summary of changes since a specific time or date"""
    show_welcome_message()
    ensure_openai_configured()
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
    
    try:
        since_time = parse_time_interval(interval)
        if isinstance(since_time, datetime.datetime):
            since_str = since_time.strftime("%Y-%m-%d")
            since_display = format_date(since_time)
        else:
            since_date = datetime.datetime.now() - since_time
            since_str = since_date.strftime("%Y-%m-%d")
            since_display = format_date(since_date)
        
        until_display = format_date(until) if until else "now"
        
    except ValueError as e:
        console.print(Panel(
            f"[red]{str(e)}[/red]",
            title="Error",
            border_style="red"
        ))
        return
    
    with progress:
        # Get commits for the period
        task_commits = progress.add_task("Getting commits...", total=None)
        commits = get_commit_range(since=since_str, until=until)
        progress.update(task_commits, completed=True)
        
        if not commits:
            console.print(Panel(
                f"[yellow]No changes found between {since_display} and {until_display}.[/yellow]",
                title="Summary",
                border_style="yellow"
            ))
            return
        
        # Generate summary
        task_summary = progress.add_task("Generating summary...", total=None)
        provider = get_ai_provider()
        model = settings.get_model()
        
        try:
            summary_text = generate_summary(commits, provider, model)
            progress.update(task_summary, completed=True)
        except Exception as e:
            progress.update(task_summary, visible=False)
            console.print(Panel(
                f"[red]Failed to generate summary\nError: {str(e)}[/red]",
                title="Error",
                border_style="red"
            ))
            return
    
    # Show the summary
    title = f"Changes Summary (From {since_display} to {until_display})"
    console.print(Panel(
        f"[green]{summary_text}[/green]",
        title=title,
        border_style="cyan"
    ))
    
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_since_{interval.replace('/', '_')}_{timestamp}.md"
        try:
            with open(filename, 'w') as f:
                f.write(f"# {title}\n\n")
                f.write(summary_text)
            console.print(f"\n[green]Summary saved to: {filename}[/green]")
        except Exception as e:
            console.print(f"\n[red]Failed to save summary: {str(e)}[/red]")

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query for finding similar commits"),
    limit: int = typer.Option(5, help="Maximum number of results to show"),
    index_all: bool = typer.Option(False, help="Re-index all commits before searching"),
):
    """Search for similar commits using AI-powered semantic search"""
    show_welcome_message()
    ensure_openai_configured()
    
    embeddings = CommitEmbeddings()
    
    # Re-index all commits if requested
    if index_all:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Indexing repository history...", total=None)
            
            try:
                # Get all commit hashes
                commits = subprocess.check_output(
                    ['git', 'log', '--format=%H'],
                    stderr=subprocess.PIPE
                ).decode('utf-8').strip().split('\n')
                
                # Process each commit
                indexed_count = 0
                for commit_hash in commits:
                    success = asyncio.run(index_commit(commit_hash, progress))
                    if success:
                        indexed_count += 1
                
                progress.update(task, completed=True)
                console.print(f"\n[green]✨ Repository indexed! Found and processed {indexed_count} commits[/green]")
                
            except Exception as e:
                progress.update(task, visible=False)
                console.print(Panel(
                    f"[red]Oops! Something went wrong while indexing: {str(e)}[/red]",
                    title="⚠️ Error",
                    border_style="red"
                ))
                return
    
    # Search for similar commits
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("🔍 Finding relevant commits...", total=None)
        
        try:
            results = asyncio.run(embeddings.search_commits(query, limit))
            progress.update(task, completed=True)
            
            if not results:
                console.print(Panel(
                    "[yellow]I couldn't find any commits matching your search. Try:\n" +
                    "• Using different keywords\n" +
                    "• Being more general in your search\n" +
                    "• Making sure you've indexed your commits (--index-all)[/yellow]",
                    title="No Results Found",
                    border_style="yellow"
                ))
                return
            
            # Group results by relevance
            high_relevance = []
            medium_relevance = []
            low_relevance = []
            
            for result in results:
                score = 1 - result['score']  # Convert distance to similarity
                if score > 0.8:
                    high_relevance.append(result)
                elif score > 0.5:
                    medium_relevance.append(result)
                else:
                    low_relevance.append(result)
            
            # Print summary
            console.print(f"\n[bold]🔍 Search Results for:[/bold] [cyan]{query}[/cyan]")
            console.print(f"[dim]Found {len(results)} relevant commits in your repository[/dim]\n")
            
            # Function to get AI-augmented summary with fallback
            async def get_commit_summary(message: str, score: float) -> str:
                try:
                    provider = get_ai_provider()
                    try:
                        prompt = f"""Given this commit message, provide a one-line natural explanation of its relevance:
                        Message: {message}
                        
                        Respond in this format:
                        <relevance>Brief explanation of how this commit relates to the search</relevance>"""
                        
                        summary = await provider.generate_commit_message(prompt, settings.get_model())
                        # Extract content between relevance tags
                        if '<relevance>' in summary and '</relevance>' in summary:
                            summary = summary.split('<relevance>')[1].split('</relevance>')[0].strip()
                        return summary
                    except Exception as e:
                        if "Connection" in str(e) and isinstance(provider, OpenAIProvider):
                            console.print("[yellow]Connection error with OpenAI, falling back to local Ollama...[/yellow]")
                            fallback_provider = OllamaProvider()
                            summary = await asyncio.get_event_loop().run_in_executor(
                                None, 
                                lambda: fallback_provider.generate_commit_message(prompt, "llama2")
                            )
                            if '<relevance>' in summary and '</relevance>' in summary:
                                summary = summary.split('<relevance>')[1].split('</relevance>')[0].strip()
                            return summary
                        raise
                except:
                    return None

            # Function to format commit info
            def format_commit_info(result, summary: str = None):
                message = result['message'].split('\n')[0]  # First line only
                
                # Format the message nicely
                info = []
                
                # Add the main message
                info.append(f"[bold white]{message}[/bold white]")
                
                # Add AI summary if available
                if summary:
                    info.append(f"[dim italic]{summary}[/dim italic]")
                
                return "\n".join(info)
            
            # Show best matches
            if high_relevance:
                console.print("\n[bold green]🎯 Best Matches[/bold green]")
                for result in high_relevance:
                    date = format_date(parse_date(result['date']))
                    hash_short = result['commit_hash'][:8]
                    # Get AI summary for high relevance commits
                    summary = asyncio.run(get_commit_summary(result['message'], 1 - result['score']))
                    console.print(Panel(
                        format_commit_info(result, summary=summary),
                        border_style="green",
                        expand=False,
                        padding=(1, 2),
                        title=f"[green]commit {hash_short} • {date}[/green]"
                    ))
            
            # Show good matches
            if medium_relevance:
                console.print("\n[bold yellow]✨ Good Matches[/bold yellow]")
                for result in medium_relevance:
                    date = format_date(parse_date(result['date']))
                    hash_short = result['commit_hash'][:8]
                    # Get AI summary for medium relevance commits
                    summary = asyncio.run(get_commit_summary(result['message'], 1 - result['score']))
                    console.print(Panel(
                        format_commit_info(result, summary=summary),
                        border_style="yellow",
                        expand=False,
                        padding=(1, 2),
                        title=f"[yellow]commit {hash_short} • {date}[/yellow]"
                    ))
            
            # Show other matches
            if low_relevance:
                console.print("\n[bold]📍 Other Related Commits[/bold]")
                for result in low_relevance:
                    date = format_date(parse_date(result['date']))
                    hash_short = result['commit_hash'][:8]
                    # Get AI summary for low relevance commits
                    summary = asyncio.run(get_commit_summary(result['message'], 1 - result['score']))
                    console.print(Panel(
                        format_commit_info(result, summary=summary),
                        border_style="blue",
                        expand=False,
                        padding=(1, 2),
                        title=f"[blue]commit {hash_short} • {date}[/blue]"
                    ))
            
            # Show tip about summary command
            if results:
                console.print("\n[dim]💡 Tip: Get a detailed analysis of any commit with:[/dim]")
                console.print("[dim]  sayless summary <commit-hash> --detailed[/dim]")
            
        except Exception as e:
            progress.update(task, visible=False)
            console.print(Panel(
                f"[red]Oops! Something went wrong while searching: {str(e)}[/red]",
                title="⚠️ Error",
                border_style="red"
            ))

@app.command("branch")
def branch_command(
    description: str = typer.Argument(None, help="Description of the branch/feature"),
    no_checkout: bool = typer.Option(False, "--no-checkout", help="Create branch without switching to it"),
    generate: bool = typer.Option(False, "--generate", "-g", help="Generate branch name from staged changes"),
    auto_add: bool = typer.Option(False, "-a", help="Automatically run 'git add .' before operation"),
):
    """Create a new branch with an AI-generated name"""
    show_welcome_message()
    ensure_openai_configured()
    create_branch(description=description, checkout=not no_checkout, generate=generate, auto_add=auto_add)

@app.command("branches")
def branches_command(
    details: bool = typer.Option(False, "--details", "-d", help="Show AI-generated summary of changes"),
):
    """List branches with optional AI-generated summaries"""
    show_welcome_message()
    ensure_openai_configured()
    list_branches(show_details=details)

@app.command("pr")
def pr_command(
    action: str = typer.Argument(..., help="Action to perform: create, list"),
    base: str = typer.Option(None, "--base", "-b", help="Base branch for PR (default: main)"),
    details: bool = typer.Option(False, "--details", "-d", help="Show AI-generated insights"),
    no_push: bool = typer.Option(False, "--no-push", help="Don't automatically push branch to GitHub"),
):
    """Manage pull requests with AI assistance"""
    show_welcome_message()
    ensure_openai_configured()
    
    if action == "create":
        create_pr(base=base, show_details=details, auto_push=not no_push)
    elif action == "list":
        list_prs(show_details=details)
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: create, list")
        sys.exit(1)

if __name__ == "__main__":
    app()

# Modified on 2025-05-18 14:30:00
