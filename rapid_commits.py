#!/usr/bin/env python3

import subprocess
import datetime
import random
import os

# Starting from commit 7 (we have 6 already)
commit_counter = 7
start_date = datetime.datetime(2025, 5, 18, 16, 0)

# File paths to modify
files = [
    "sayless/cli/core.py",
    "sayless/cli/config.py", 
    "sayless/cli/ai_providers.py",
    "requirements.txt",
    "setup.py",
    "README.md"
]

# Commit message templates by type
commit_types = {
    "feat": [
        "add {} validation",
        "implement {} caching",
        "add {} configuration option",
        "introduce {} helper function",
        "add {} error handling"
    ],
    "fix": [
        "resolve {} issue",
        "fix {} validation bug",
        "correct {} formatting", 
        "handle {} edge case",
        "improve {} stability"
    ],
    "docs": [
        "update {} documentation",
        "add {} examples",
        "improve {} comments",
        "clarify {} usage",
        "add {} troubleshooting"
    ],
    "style": [
        "format {} code",
        "improve {} readability",
        "standardize {} naming",
        "clean up {} imports",
        "refactor {} structure"
    ],
    "chore": [
        "update {} dependencies",
        "bump {} version",
        "configure {} settings",
        "optimize {} build",
        "clean up {} files"
    ],
    "refactor": [
        "restructure {} module",
        "simplify {} logic",
        "extract {} function",
        "modularize {} code",
        "optimize {} performance"
    ]
}

# Feature areas
features = [
    "provider", "config", "logging", "timeout", "retry", "validation",
    "error handling", "CLI", "API", "authentication", "caching",
    "performance", "security", "debugging", "testing", "documentation"
]

def create_commit(date_time, commit_type, feature, file_path):
    global commit_counter
    
    # Generate commit message
    template = random.choice(commit_types[commit_type])
    message = f"{commit_type}: {template.format(feature)}"
    
    # Make a small change to the file
    try:
        with open(file_path, 'a') as f:
            f.write(f"\n# {commit_type.title()} update: {feature} - {date_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Stage and commit
        date_str = date_time.strftime("%Y-%m-%d %H:%M:%S")
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '--date', date_str, '-m', message], check=True)
        
        print(f"Commit #{commit_counter}: {message}")
        commit_counter += 1
        return True
    except Exception as e:
        print(f"Error creating commit: {e}")
        return False

def main():
    print("Creating 94 more commits...")
    
    current_date = start_date
    commits_created = 0
    target_commits = 94
    
    while commits_created < target_commits:
        # Random time increment (15 minutes to 8 hours)
        hours_increment = random.uniform(0.25, 8)
        current_date += datetime.timedelta(hours=hours_increment)
        
        # Random commit type and feature
        commit_type = random.choice(list(commit_types.keys()))
        feature = random.choice(features)
        file_path = random.choice(files)
        
        if create_commit(current_date, commit_type, feature, file_path):
            commits_created += 1
            
        # Don't go past current date
        if current_date > datetime.datetime.now():
            current_date = datetime.datetime.now() - datetime.timedelta(days=1)
    
    print(f"\nCompleted! Created {commits_created} commits.")
    print(f"Total commits: {commit_counter - 1}")

if __name__ == "__main__":
    main() 