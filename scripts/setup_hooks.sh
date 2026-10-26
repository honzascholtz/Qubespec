#!/usr/bin/env bash
# Configure git to use version-controlled hooks in .githooks/
set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

echo "Setting git core.hooksPath to .githooks..."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
cp -f .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

echo "Git pre-commit hook installed successfully!"
echo "Pylint will now run automatically on staged Python files on every commit."
