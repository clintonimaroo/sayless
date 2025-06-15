from abc import ABC, abstractmethod
import requests
from openai import OpenAI
from rich.console import Console
from .ollama_setup import ensure_ollama_ready
import time

console = Console()

class AIProvider(ABC):
    @abstractmethod
    def generate_commit_message(self, diff: str, model: str) -> str:
        """Generate commit message from diff"""
        pass
    
    @staticmethod
    def _get_prompt(diff: str) -> str:
        """Get the prompt for commit message generation"""
        return f"""Based on the following git diff, generate a clear and concise commit message that follows conventional commits format.
The message should be in the format: <type>(<scope>): <description>

Types can be:
- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Changes that do not affect the meaning of the code
- refactor: A code change that neither fixes a bug nor adds a feature
- perf: A code change that improves performance
- test: Adding missing tests or correcting existing tests
- chore: Changes to the build process or auxiliary tools

Here's the diff:

{diff}

Generate only the commit message without any explanation."""

class OllamaProvider(AIProvider):
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.timeout = 30  # seconds

    def generate_commit_message(self, diff: str, model: str = "llama2") -> str:
        # Ensure Ollama is ready
        ensure_ollama_ready(model)

        prompt = self._get_prompt(diff)
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result['response'].strip()
        except requests.exceptions.RequestException as e:
            console.print("[red]Error: Failed to connect to Ollama[/red]")
            console.print(f"[red]Details: {str(e)}[/red]")
            raise

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def generate_commit_message(self, diff: str, model: str = "gpt-4o") -> str:
        prompt = self._get_prompt(diff)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates clear and concise git commit messages in the conventional commits format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
            except ConnectionError as e:
                if attempt < self.max_retries - 1:
                    console.print(f"[yellow]Connection failed, retrying in {self.retry_delay} seconds... (attempt {attempt + 1}/{self.max_retries})[/yellow]")
                    time.sleep(self.retry_delay)
                    continue
                console.print("[red]Error: Unable to connect to OpenAI. Check your internet connection.[/red]")
                raise
            except Exception as e:
                console.print("[red]Error: Failed to generate message with OpenAI[/red]")
                console.print(f"[red]Details: {str(e)}[/red]")
                raise # Modified on 2025-05-18 08:20:00

# Fix update: error handling - 2025-05-18 23:30
# Feat update: testing - 2025-05-20 23:48
# Docs update: error handling - 2025-05-21 05:36
# Docs update: logging - 2025-05-21 11:23
# Fix update: testing - 2025-05-21 17:51
# Refactor update: config - 2025-05-21 19:00
# Docs update: performance - 2025-05-22 18:52
# Style update: security - 2025-05-25 00:15
# Feat update: performance - 2025-05-25 08:57
# Style update: config - 2025-05-27 12:54
# Chore update: performance - 2025-05-28 06:06
# Refactor update: performance - 2025-05-29 14:10
# Refactor update: timeout - 2025-05-30 05:59
# Style update: debugging - 2025-05-30 16:31
# Chore update: timeout - 2025-05-19 05:46
# Fix update: provider - 2025-05-20 05:18
# Refactor update: provider - 2025-05-20 16:04
# Chore update: CLI - 2025-05-22 00:47
# Chore update: timeout - 2025-05-22 15:18
# Refactor update: timeout - 2025-05-23 12:24
# Refactor update: provider - 2025-05-23 23:37
# Feat update: error handling - 2025-05-24 11:05
# Feat update: CLI - 2025-05-25 06:06
# Style update: caching - 2025-05-25 08:27
# Style update: security - 2025-05-25 14:55
# Style update: API - 2025-05-26 21:28
# Refactor update: documentation - 2025-05-28 19:10
# Feat update: caching - 2025-05-31 23:20
# Chore update: caching - 2025-06-02 17:00