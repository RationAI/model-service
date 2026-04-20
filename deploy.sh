#!/bin/bash
set -e

NAMESPACE="rationai-jobs-ns"
HELM_RELEASE_NAME="rayservice-model"
HELM_CHART_DIR="helm/rayservice"
BASE_RAYSERVICE_YAML="rayservice/base/ray-service-base.yaml"
WORKER_DIR="rayservice/components/workers"
APPLICATION_DEFINITIONS_DIR="rayservice/components/applications"

shopt -s nullglob
worker_files=("${WORKER_DIR}"/*.yaml)
shopt -u nullglob

mapfile -t application_definition_files < <(
    find "${APPLICATION_DEFINITIONS_DIR}" -maxdepth 1 -name "*.yaml"
)

set_file_args=()
set_file_args+=(--set-file "baseRayServiceYaml=${BASE_RAYSERVICE_YAML}")

for worker_file in "${worker_files[@]}"; do
    worker_name="$(basename "${worker_file}" .yaml)"
    set_file_args+=(--set-file "workerYamls.${worker_name}=${worker_file}")
done

for app_file in "${application_definition_files[@]}"; do
    app_name="$(basename "${app_file}" .yaml)"
    set_file_args+=(--set-file "applicationDefinitionYamls.${app_name}=${app_file}")
done

helm template "${HELM_RELEASE_NAME}" "${HELM_CHART_DIR}" "${set_file_args[@]}" | kubectl apply -n "${NAMESPACE}" -f -