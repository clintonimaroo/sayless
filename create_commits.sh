#!/bin/bash

# Script to create 100 backdated commits
set -e

# Commit counter
commit_num=2

# Start date (May 17, 2025)
start_date="2025-05-17"

# Function to create a commit with backdated timestamp
create_commit() {
    local date_str="$1"
    local time_str="$2"
    local message="$3"
    local file_to_modify="$4"
    local modification="$5"
    
    echo "Creating commit #$commit_num: $message"
    
    # Make a small change
    case "$modification" in
        "add_comment")
            echo "# Modified on $date_str $time_str" >> "$file_to_modify"
            ;;
        "add_newline")
            echo "" >> "$file_to_modify"
            ;;
        "touch_file")
            touch "$file_to_modify"
            ;;
    esac
    
    git add .
    git commit --date="$date_str $time_str" -m "$message"
    ((commit_num++))
}

# Create commits with various types of changes
create_commit "2025-05-17" "12:30:00" "docs: add module header to config.py" "sayless/cli/config.py" "add_comment"
create_commit "2025-05-17" "15:45:00" "style: format imports in core.py" "sayless/cli/core.py" "add_newline"
create_commit "2025-05-18" "08:20:00" "fix: improve error handling" "sayless/cli/ai_providers.py" "add_comment"
create_commit "2025-05-18" "11:15:00" "chore: update dependency versions" "requirements.txt" "add_comment"
create_commit "2025-05-18" "14:30:00" "feat: add debug logging" "sayless/cli/core.py" "add_comment"

echo "Created $((commit_num - 2)) commits so far" 