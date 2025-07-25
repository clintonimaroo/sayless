# Sayless

[![Version](https://img.shields.io/badge/version-0.2.1-blue.svg)](https://github.com/teddyoweh/sayless)
[![Python](https://img.shields.io/badge/python-3.7+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Your intelligent Git companion that streamlines repository workflows through AI. Sayless handles everything from commit messages to PR management, making version control more intuitive and efficient.



Sayless is a powerful AI-powered Git automation tool that enhances your development workflow by:
- Generating meaningful commit messages that follow best practices
- Creating contextual branch names based on your changes
- Managing pull requests with AI-generated descriptions and insights
- Providing semantic search across your commit history
- Generating comprehensive summaries of repository changes

## Core Capabilities

### Intelligent Git Operations
- Generate precise commit messages that capture the essence of your changes
- Create meaningful branch names based on your work context
- Get instant PR descriptions that highlight key changes and impact
- Search your entire commit history using natural language

### Repository Insights
- Get clear summaries of codebase changes over time
- Understand the impact of specific commits or PRs
- Track development progress with automated reports
- Export insights for team reviews and documentation

### Flexible AI Integration
- Use OpenAI for cloud-based processing
- Run everything locally with Ollama for complete privacy
- Switch between providers seamlessly based on your needs

## Installation

### Prerequisites
- Python 3.7+
- Git
- OpenAI API key (for cloud features) or Ollama (for local processing)

### Install via pip
```bash
pip install sayless
```

### Initial Configuration

#### Using OpenAI (Recommended)
```bash
# Set your OpenAI API key
sayless config --openai-key YOUR_API_KEY

# Or use environment variable
export OPENAI_API_KEY=your_api_key
```

#### Using Ollama (Local AI)
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Switch to Ollama:
```bash
sayless switch ollama
```

## Usage Guide

### Quick Start
```bash
# Generate commits
sl g                    # Generate commit from staged changes
sl g -a                 # Stage all changes and commit

# Branch management
sl branch "Add auth"    # Create and switch to a contextual branch
sl branches            # List branches with summaries

# Pull requests
sl pr create           # Create a PR with smart title and description
sl pr list --details   # List PRs with AI insights

# Search and analyze
sl search "auth fixes"  # Find relevant commits
sl since 1w            # Summarize last week's changes
```

### Commit Message Generation

Sayless helps you write better commit messages by analyzing your changes:

```bash
# Stage specific files
git add file1.js file2.js
sl g                    # Generate commit message

# Stage and commit all changes
sl g -a                 # Equivalent to: git add . && sayless generate

# Preview without committing
sl g --preview         # Show message without creating commit
```

#### Best Practices
- Stage related changes together for more focused commit messages
- Use `--preview` to review and refine messages before committing
- Let Sayless handle conventional commit formatting automatically

### Branch Management

Create meaningful branch names and track changes effectively:

```bash
# Create branch from description
sl branch "Implement user authentication system"
# Creates: feat/implement-user-authentication-system

# Generate branch name from staged changes
git add .
sl branch -g           # AI analyzes changes and creates branch

# List branches with AI insights
sl branches --details  # Shows branch summaries and analysis
```

#### Advanced Branch Options
```bash
# Create without switching
sl branch "Add logging" --no-checkout

# Auto-stage and generate name
sl branch -g -a        # Stages changes and generates name

# Custom branch type
sl branch "Fix login bug" # Creates: fix/login-bug
sl branch "Update docs"   # Creates: docs/update-docs
```

### Pull Request Management

Create and manage PRs with AI-generated content:

```bash
# Create PR with AI-generated title and description
sl pr create

# Target specific base branch
sl pr create --base develop

# Create with detailed AI insights
sl pr create --details

# List PRs with analysis
sl pr list --details
```

#### PR Best Practices
- Stage all relevant changes before creating PR
- Use `--details` for additional context and impact analysis
- Review AI-generated content before confirming PR creation

### Repository Analysis

Get insights into your codebase and changes:

```bash
# Analyze specific commit
sl summary <commit-hash> --detailed

# Summarize recent changes
sl since 1d            # Last 24 hours
sl since 1w            # Last week
sl since 1m            # Last month
sl since 2023-01-01   # Since specific date

# Save analysis to file
sl since 1w --save    # Exports to markdown file
```

### Semantic Search

Search through your commit history using natural language:

```bash
# Search commits
sl search "authentication improvements"
sl search "bug fixes in login system"
sl search "performance optimizations"

# Index all commits before searching
sl search "database changes" --index-all

# Limit results
sl search "API updates" --limit 10
```

## Advanced Configuration

### AI Provider Settings

#### OpenAI Configuration
```bash
# Switch to OpenAI
sl switch openai --key YOUR_API_KEY

# Change model
sl switch openai --model gpt-4
```

#### Ollama Configuration
```bash
# Switch to Ollama
sl switch ollama

# Change model
sl switch ollama --model llama2
```

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `GITHUB_TOKEN`: GitHub token for PR operations

## Integration with CI/CD

Sayless can be integrated into your CI/CD pipelines:

```bash
# Non-interactive commit generation
sl g -a --yes

# Automated PR creation
sl pr create --no-confirm --base main
```

## Troubleshooting

### Common Issues

1. OpenAI API Issues
```bash
# Verify API key
sl config --show

# Switch to Ollama temporarily
sl switch ollama
```

2. Rate Limiting
If you encounter rate limiting errors, wait a few minutes or switch providers:
```bash
sl switch ollama  # Use local AI instead
```

3. Git Repository Issues
```bash
# Check git status
git status

# Ensure changes are staged
git add .
```

4. PR Creation Issues
```bash
# Verify GitHub token
export GITHUB_TOKEN=your_token

# Check repository access
sl pr list
```

## Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`sl branch "Add new feature"`)
3. Commit your changes (`sl g -a`)
4. Push to your fork
5. Create a Pull Request (`sl pr create`)

## License

MIT License - See [LICENSE](LICENSE) for details.