import os
from typing import Any

import yaml


script_dir = os.path.dirname(os.path.abspath(__file__))
applications_definitions_dir = os.path.join(script_dir, "applications-definitions")
output_file = os.path.join(script_dir, "serve-config-patch.yaml")
base_file = os.path.join(script_dir, "..", "..", "base", "ray-service-base.yaml")

application_files = [
    f for f in os.listdir(applications_definitions_dir) if f.endswith(".yaml")
]

if not application_files:
    raise RuntimeError(
        f"No application definition files found in {applications_definitions_dir}"
    )

merged_applications = []

with open(base_file) as base_f:
    base_data = yaml.safe_load(base_f)

rayservice_name = base_data["metadata"]["name"]

for file_name in sorted(application_files):
    file_path = os.path.join(applications_definitions_dir, file_name)
    with open(file_path) as current_f:
        data = yaml.safe_load(current_f)
        if not data or "applications" not in data:
            raise RuntimeError(f"File {file_name} is missing 'applications' key")
        merged_applications.extend(data["applications"])

serve_config_str = yaml.dump({"applications": merged_applications}, sort_keys=False)


class LiteralString(str):
    pass


def literal_presenter(dumper: Any, data: Any) -> Any:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralString, literal_presenter)

patch = {
    "apiVersion": "ray.io/v1",
    "kind": "RayService",
    "metadata": {"name": rayservice_name},
    "spec": {"serveConfigV2": LiteralString(serve_config_str)},
}

with open(output_file, "w") as out_f:
    yaml.dump(patch, out_f, sort_keys=False)

print(f"Generated {output_file} from {len(application_files)} application files:")
for application_file in sorted(application_files):
    print(f"  - {application_file}")
