#!/usr/bin/env bash
set -euo pipefail

# Push local repository state to GitHub with safe defaults.
# Default remote URL is the one requested by the user.

DEFAULT_REPO_URL="https://github.com/fesihkeskin/Policy-KG-Guardrails.git"
REMOTE_NAME="origin"
BRANCH="main"
COMMIT_MSG="chore: sync Policy-KG-Guardrails"
OVERWRITE_REMOTE=0
DRY_RUN=0
PUSH_TAGS=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/push_to_github.sh [options]

Options:
  --repo-url URL          GitHub repository URL
                          (default: https://github.com/fesihkeskin/Policy-KG-Guardrails.git)
  --remote NAME           Remote name (default: origin)
  --branch NAME           Branch to push (default: main)
  --message TEXT          Commit message for staged changes
  --overwrite-remote      Replace existing remote URL if different
  --push-tags             Push tags after branch push
  --dry-run               Show what would happen without pushing
  -h, --help              Show this help

Examples:
  scripts/push_to_github.sh
  scripts/push_to_github.sh --message "feat: initial prototype"
  scripts/push_to_github.sh --dry-run
USAGE
}

REPO_URL="$DEFAULT_REPO_URL"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE_NAME="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    --message)
      COMMIT_MSG="${2:-}"
      shift 2
      ;;
    --overwrite-remote)
      OVERWRITE_REMOTE=1
      shift
      ;;
    --push-tags)
      PUSH_TAGS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: current directory is not a git repository." >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Repository: $REPO_ROOT"
echo "Remote: $REMOTE_NAME -> $REPO_URL"
echo "Branch: $BRANCH"

# Ensure target branch exists or set unborn HEAD to it.
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  CURRENT_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
  if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    git switch "$BRANCH"
  fi
else
  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    git switch -c "$BRANCH"
  else
    git symbolic-ref HEAD "refs/heads/$BRANCH"
  fi
fi

# Configure remote.
if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  CURRENT_URL="$(git remote get-url "$REMOTE_NAME")"
  if [[ "$CURRENT_URL" != "$REPO_URL" ]]; then
    if [[ "$OVERWRITE_REMOTE" -eq 1 ]]; then
      git remote set-url "$REMOTE_NAME" "$REPO_URL"
      echo "Updated remote '$REMOTE_NAME' URL."
    else
      echo "Error: remote '$REMOTE_NAME' URL differs." >&2
      echo "Current: $CURRENT_URL" >&2
      echo "Wanted : $REPO_URL" >&2
      echo "Run again with --overwrite-remote to replace it." >&2
      exit 1
    fi
  fi
else
  git remote add "$REMOTE_NAME" "$REPO_URL"
  echo "Added remote '$REMOTE_NAME'."
fi

# Stage and commit if there are any changes.
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -m "$COMMIT_MSG"
  echo "Committed local changes."
else
  echo "No local changes to commit."
fi

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "Error: no commits found; nothing to push." >&2
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: git push --set-upstream $REMOTE_NAME $BRANCH"
  git push --set-upstream "$REMOTE_NAME" "$BRANCH" --dry-run
  if [[ "$PUSH_TAGS" -eq 1 ]]; then
    echo "Dry run: git push $REMOTE_NAME --tags"
    git push "$REMOTE_NAME" --tags --dry-run
  fi
  exit 0
fi

git push --set-upstream "$REMOTE_NAME" "$BRANCH"
if [[ "$PUSH_TAGS" -eq 1 ]]; then
  git push "$REMOTE_NAME" --tags
fi

echo "Push completed successfully."
