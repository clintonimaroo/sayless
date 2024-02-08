import subprocess
import sys
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import re
from .ai_providers import OpenAIProvider, OllamaProvider
from .config import Config

console = Console()
settings = Config()

def run_git_command(command: List[str], check=True, capture_output=True) -> subprocess.CompletedProcess:
    """Run a git command and handle errors"""
    try:
        result = subprocess.run(['git'] + command, check=check, capture_output=capture_output, text=True)
        if check and result.returncode != 0:
            error_msg = result.stderr.strip()
            
            # Handle specific git errors with better messages
            if "Needed a single revision" in error_msg or "unknown revision" in error_msg:
                # Get recent commits for context
                try:
                    recent = subprocess.run(
                        ['git', 'log', '--oneline', '-n', '5'],
                        capture_output=True,
                        text=True
                    )
                    if recent.returncode == 0:
                        commits = recent.stdout.strip()
                        raise subprocess.CalledProcessError(
                            result.returncode,
                            result.args,
                            output=result.stdout,
                            stderr=f"Invalid commit reference. Please provide a valid commit hash.\n\nRecent commits:\n{commits}"
                        )
                except:
                    pass
            
            # If no specific handling, raise original error
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=error_msg
            )
        return result
    except subprocess.CalledProcessError as e:
        if not check:
            return e
        error_msg = e.stderr.strip() if e.stderr else str(e)
        console.print(Panel(
            f"[red]Git command failed: {' '.join(['git'] + command)}\nError: {error_msg}[/red]",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        if not check:
            raise
        console.print(Panel(
            f"[red]Failed to run git command: {' '.join(['git'] + command)}\nError: {str(e)}[/red]",
            title="Error",
            border_style="red"
        ))
        sys.exit(1)

def get_current_branch() -> str:
    """Get the name of the current branch"""
    result = run_git_command(['branch', '--show-current'])
    return result.stdout.strip()

def sanitize_branch_name(name: str) -> str:
    """Convert a string into a valid git branch name"""
    # Convert to lowercase
    name = name.lower()
    # Replace spaces and special characters with hyphens
    name = re.sub(r'[^a-z0-9]+', '-', name)
    # Remove leading and trailing hyphens
    name = name.strip('-')
    # Limit length while preserving words
    if len(name) > 50:
        words = name.split('-')
        shortened = []
        length = 0
        for word in words:
            if length + len(word) + 1 <= 50:  # +1 for the hyphen
                shortened.append(word)
                length += len(word) + 1
            else:
                break
        name = '-'.join(shortened)
    return name

def get_branch_type(description: str) -> str:
    """Use AI to determine the branch type based on description"""
    provider = settings.get_provider()
    model = settings.get_model()
    
    prompt = f"""Based on this feature description, determine the most appropriate branch type prefix.
Choose one of: feat, fix, docs, style, refactor, perf, test, chore

Description: {description}

Respond with ONLY the type (e.g., 'feat' or 'fix'). No explanation needed."""

    try:
        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        branch_type = ai.generate_commit_message(prompt, model).strip().lower()
        
        # Validate the response
        valid_types = {'feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore'}
        if branch_type not in valid_types:
            # Try to infer from description
            desc_lower = description.lower()
            if any(word in desc_lower for word in ['fix', 'bug', 'issue']):
                return 'fix'
            elif any(word in desc_lower for word in ['doc', 'readme']):
                return 'docs'
            elif any(word in desc_lower for word in ['refactor', 'clean']):
                return 'refactor'
            elif any(word in desc_lower for word in ['test']):
                return 'test'
            elif any(word in desc_lower for word in ['style', 'format']):
                return 'style'
            elif any(word in desc_lower for word in ['perf', 'optimize']):
                return 'perf'
            return 'feat'  # Default to feat if no match
        
        return branch_type
    except:
        # If AI fails, try to infer from description
        desc_lower = description.lower()
        if any(word in desc_lower for word in ['fix', 'bug', 'issue']):
            return 'fix'
        elif any(word in desc_lower for word in ['doc', 'readme']):
            return 'docs'
        elif any(word in desc_lower for word in ['refactor', 'clean']):
            return 'refactor'
        elif any(word in desc_lower for word in ['test']):
            return 'test'
        elif any(word in desc_lower for word in ['style', 'format']):
            return 'style'
        elif any(word in desc_lower for word in ['perf', 'optimize']):
            return 'perf'
        return 'feat'  # Default to feat if all else fails

def get_staged_changes_description() -> str:
    """Get a description of staged changes using AI"""
    try:
        # First check for staged changes
        diff = run_git_command(['diff', '--cached', '--no-color']).stdout
        
        # If no staged changes, try to use the latest commit
        if not diff.strip():
            try:
                # Get the latest commit message and diff
                latest_commit = run_git_command(['log', '-1', '--format=%B']).stdout.strip()
                commit_diff = run_git_command(['log', '-1', '-p', '--no-color']).stdout.strip()
                
                if latest_commit:
                    # Extract the commit type if it follows conventional commit format
                    commit_type = 'feat'  # default
                    if '(' in latest_commit and '):' in latest_commit:
                        type_part = latest_commit.split('(')[0].strip().lower()
                        if type_part in ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore']:
                            commit_type = type_part
                    
                    # Use the commit message as the branch name
                    description = latest_commit.split('\n')[0]  # First line only
                    # Remove conventional commit format if present
                    if '(' in description and '):' in description:
                        description = description.split('):')[1].strip()
                    # Limit length
                    words = description.split()
                    description = ' '.join(words[:4])  # Limit to 4 words
                    return description
                
            except Exception:
                pass
            
            # Get list of modified files for context
            modified_files = run_git_command(['diff', '--name-only']).stdout.strip().split('\n')
            if modified_files and modified_files[0]:
                file_context = "Modified files: " + ", ".join(modified_files[:3])
            else:
                file_context = "No files modified"
            
            raise ValueError(f"No staged changes. Stage changes with 'git add' first.\n{file_context}")

        provider = settings.get_provider()
        model = settings.get_model()
        
        # Get list of modified files
        modified_files = run_git_command(['diff', '--cached', '--name-only']).stdout.strip().split('\n')
        files_context = ", ".join(modified_files[:3])
        
        prompt = f"""You are a branch name generator. Create a short, descriptive branch name based on these changes.
Rules:
1. Use 2-4 words maximum
2. Focus on the main feature or purpose
3. Be specific but concise
4. Use only lowercase letters, numbers, and hyphens
5. Keep total length under 40 characters
6. NEVER include words like 'sorry', 'apologize', or 'need'
7. NEVER explain or ask for more information

Files changed: {files_context}

Example good branch names:
- add-user-auth
- fix-login-bug
- update-api-endpoints
- improve-error-handling
- refactor-db-queries

Changes:
{diff}

Respond with ONLY the branch name, no other text."""
        
        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        description = ai.generate_commit_message(prompt, model).strip()
        
        # Validate and clean up the response
        description = description.lower()
        
        # Remove any conventional commit format if AI included it
        if '(' in description and '):' in description:
            description = description.split('):')[1].strip()
        
        # Remove common error phrases
        error_phrases = ['sorry', 'apologize', 'need', 'please', 'could you', 'i am', "i'm", 'cannot', 'can not']
        if any(phrase in description.lower() for phrase in error_phrases):
            # Fall back to using modified files for branch name
            if modified_files and modified_files[0]:
                main_file = modified_files[0].split('/')[-1].split('.')[0]
                action = 'update'
                if 'test' in main_file.lower():
                    action = 'add-tests'
                elif 'doc' in main_file.lower() or 'readme' in main_file.lower():
                    action = 'update-docs'
                return f"{action}-{main_file}"
            return "update-codebase"
        
        # Limit words and length
        words = description.split()
        description = ' '.join(words[:4])  # Limit to 4 words
        
        # Final validation
        if len(description) > 40:
            words = description.split('-')
            description = '-'.join(words[:3])  # Take first 3 parts if still too long
        
        return description
    except Exception as e:
        raise ValueError(f"Failed to generate branch name: {str(e)}")

def auto_add_changes(progress: Progress = None) -> None:
    """Stage all changes with git add ."""
    task = None
    if progress:
        task = progress.add_task("Staging all changes...", total=None)
    try:
        run_git_command(['add', '.'])
        if task:
            progress.update(task, completed=True)
            console.print("[green]✓[/green] Staged all changes")
    except Exception as e:
        if task:
            progress.update(task, visible=False)
        raise ValueError(f"Failed to stage changes: {str(e)}")

def create_branch(description: str = None, checkout: bool = True, generate: bool = False, auto_add: bool = False) -> str:
    """Create a new branch with an AI-generated name based on description or staged changes"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        try:
            # Auto-add changes if requested
            if auto_add:
                auto_add_changes(progress)
            
            if generate:
                # Generate description from staged changes or latest commit
                task_desc = progress.add_task("Analyzing changes...", total=None)
                description = get_staged_changes_description()
                progress.update(task_desc, completed=True)
            elif not description:
                raise ValueError("Either provide a description or use --generate (-g)")
            
            # Determine branch type
            task_type = progress.add_task("Determining branch type...", total=None)
            branch_type = get_branch_type(description)
            progress.update(task_type, completed=True)
            
            # Generate branch name
            task_name = progress.add_task("Generating branch name...", total=None)
            branch_name = f"{branch_type}/{sanitize_branch_name(description)}"
            
            # Validate final branch name
            if len(branch_name) > 50:
                parts = branch_name.split('/')
                type_part = parts[0]
                name_part = parts[1].split('-')
                branch_name = f"{type_part}/{'-'.join(name_part[:3])}"
            
            progress.update(task_name, completed=True)
            
            # Create and checkout branch
            task_create = progress.add_task(f"Creating branch {branch_name}...", total=None)
            if checkout:
                run_git_command(['checkout', '-b', branch_name])
            else:
                run_git_command(['branch', branch_name])
            progress.update(task_create, completed=True)
            
            console.print(f"\n[bold green]✓[/bold green] Created branch: [cyan]{branch_name}[/cyan]")
            if checkout:
                console.print(f"[green]Switched to branch: [cyan]{branch_name}[/cyan][/green]")
            
            return branch_name
        except Exception as e:
            console.print(Panel(
                f"[red]{str(e)}[/red]\n\n"
                "[yellow]To create a branch, either:[/yellow]\n"
                "1. Provide a description:\n"
                "   [blue]sl branch \"add user authentication\"[/blue]\n"
                "2. Stage changes and use auto-generate:\n"
                "   [blue]sl branch -g -a[/blue] (auto-stages and generates name)",
                title="Error",
                border_style="red"
            ))
            sys.exit(1)

def list_branches(show_details: bool = False) -> List[Dict[str, str]]:
    """List all branches with optional AI-generated summaries"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Getting branches...", total=None)
        
        # Get all branches
        result = run_git_command(['branch', '--format=%(refname:short)'])
        branches = [b.strip() for b in result.stdout.splitlines()]
        progress.update(task, completed=True)
        
        if not branches:
            console.print("[yellow]No branches found[/yellow]")
            return []
        
        # Create table
        table = Table(title="Git Branches", show_header=True, header_style="bold cyan")
        table.add_column("Branch", style="cyan")
        table.add_column("Last Commit", style="yellow")
        if show_details:
            table.add_column("Summary", style="green")
        
        branch_info = []
        for branch in branches:
            # Get last commit info
            commit_info = run_git_command(['log', '-1', '--format=%h %s', branch])
            last_commit = commit_info.stdout.strip()
            
            info = {
                'name': branch,
                'last_commit': last_commit,
            }
            
            if show_details:
                # Get branch summary using AI
                task_summary = progress.add_task(f"Analyzing branch {branch}...", total=None)
                try:
                    summary = get_branch_summary(branch)
                    progress.update(task_summary, completed=True)
                    info['summary'] = summary
                except Exception as e:
                    progress.update(task_summary, visible=False)
                    info['summary'] = "[red]Failed to generate summary[/red]"
            
            branch_info.append(info)
            
            # Add to table
            if show_details:
                table.add_row(branch, last_commit, info.get('summary', ''))
            else:
                table.add_row(branch, last_commit)
        
        console.print(table)
        return branch_info

def get_branch_summary(branch: str) -> str:
    """Generate an AI summary of the branch's changes"""
    # Get branch changes
    try:
        # Find the merge base with main/master
        base_branch = 'main' if run_git_command(['rev-parse', '--verify', 'main'], check=False).returncode == 0 else 'master'
        merge_base = run_git_command(['merge-base', base_branch, branch]).stdout.strip()
        
        # Get diff summary
        diff = run_git_command(['diff', '--stat', merge_base, branch]).stdout
        commits = run_git_command(['log', '--oneline', f'{merge_base}..{branch}']).stdout
        
        provider = settings.get_provider()
        model = settings.get_model()
        
        prompt = f"""Provide a one-line summary of these branch changes:

Commits:
{commits}

Changes:
{diff}

Keep the summary concise and focused on the main purpose of the changes."""

        if provider == 'openai':
            api_key = settings.get_openai_api_key()
            ai = OpenAIProvider(api_key)
        else:
            ai = OllamaProvider()
        
        return ai.generate_commit_message(prompt, model).strip()
    except:
        return "No changes or unable to generate summary" 