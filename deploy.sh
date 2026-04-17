#!/bin/bash
set -e

python kustomize/components/models/merge_models.py
kustomize build kustomize/overlays | kubectl apply -n rationai-jobs-ns -f -
