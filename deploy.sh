#!/bin/bash
set -e

python3 kustomize/components/applications/merge_applications.py
kustomize build kustomize/overlays | kubectl apply -n rationai-jobs-ns -f -
