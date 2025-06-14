#!/usr/bin/env bash

# Watch the git repository and pull and run the train.py script when changes are detected

while true; do
    git pull
    nix develop -c python train.py
    sleep 1
done