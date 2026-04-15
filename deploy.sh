#!/bin/bash
set -e

python kustomize/components/models/merge_models.py
kubectl apply -k kustomize/overlays -n rationai-jobs-ns
