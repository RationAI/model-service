# deploy.ps1
python kustomize/components/models/merge_models.py
if ($LASTEXITCODE -ne 0) { exit 1 }
kubectl apply -k kustomize/overlays -n rationai-jobs-ns