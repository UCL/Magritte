#! /bin/bash

GIT_DIR=$(git rev-parse --git-dir)

echo "Installing hooks..."
# remove existing pre-commit
rm $GIT_DIR/hooks/pre-commit
# this command creates symlink to our pre-commit script
ln -s ../../scripts/pre-commit.sh $GIT_DIR/hooks/pre-commit
echo "Done!"

chmod +x pre-commit.sh
chmod +x install_git_hooks.sh
