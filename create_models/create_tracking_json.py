import argparse
import os
import json


# Load data
parser = argparse.ArgumentParser()
parser.add_argument("-i", '--json_specs', type=str, help='Path to json file with parameters')
args = parser.parse_args()

with open(args.json_specs) as f:
    specs = json.load(f)


general_working_dir = specs.get('general_working_dir', None)
assert general_working_dir is not None, "general_working_dir is None"

progress_file = os.path.join(general_working_dir, "progress.json")
progress = {
    "chunk_reference": False,
    "marker_computation": False,
    "model_creation": False,
    "create_sampled_ref": False,
}

# write progress file
with open(progress_file, 'w') as f:
    json.dump(progress, f)

print("Created progress file")
