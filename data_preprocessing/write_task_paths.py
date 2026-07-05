import h5py
import json
import os
from collections import defaultdict
import argparse
from tqdm import tqdm


class TaskPathsGenerator:
    """Write per-episode task_paths.json files from an HDF5 task dictionary.

    For each episode path listed in the HDF5, creates a task_paths.json in that
    episode directory containing all sibling paths that share the same task key.

    Example output:
    {
        "same": ["path1", "path2", "path3"]
    }
    """

    def __init__(self, hdf5_path: str, dry_run: bool = False):
        self.hdf5_path = hdf5_path
        self.dry_run = dry_run

        self.stats = {
            'total_paths': 0,
            'directories_created': 0,
            'files_created': 0,
            'files_skipped': 0,
            'files_updated': 0,
            'errors': 0
        }

    def extract_and_map_data(self):
        """Build a mapping from episode path to the task keys it belongs to."""
        mapping = defaultdict(lambda: defaultdict(list))

        print("Reading and mapping HDF5 data...")
        with h5py.File(self.hdf5_path, 'r') as f:
            groups = list(f.keys())

            for group_name in groups:
                group = f[group_name]
                if not isinstance(group, h5py.Group):
                    continue

                task_names = list(group.keys())
                for task_name in tqdm(task_names, desc=f"Scanning {group_name}"):
                    task_group = group[task_name]
                    if not isinstance(task_group, h5py.Group) or 'same' not in task_group:
                        continue

                    same_dataset = task_group['same']
                    if not isinstance(same_dataset, h5py.Dataset):
                        continue

                    try:
                        paths = [p.decode('utf-8') if isinstance(p, bytes) else str(p)
                                 for p in same_dataset[:]]

                        for path in paths:
                            mapping[path][group_name].append(task_name)

                    except Exception as e:
                        print(f"Warning: failed to read task {task_name}: {e}")

        return mapping

    def generate_json_content(self, episode_dir: str, groups_data: dict) -> dict:
        """Collect all paths for the task keys associated with this episode."""
        result = {}

        with h5py.File(self.hdf5_path, 'r') as f:
            for group_name, task_names in groups_data.items():
                if group_name not in f:
                    continue

                group = f[group_name]
                for task_name in task_names:
                    if task_name not in group:
                        continue

                    task_group = group[task_name]
                    if not isinstance(task_group, h5py.Group):
                        continue

                    for key_name in task_group.keys():
                        dataset = task_group[key_name]
                        if not isinstance(dataset, h5py.Dataset):
                            continue

                        try:
                            paths = [p.decode('utf-8') if isinstance(p, bytes) else str(p)
                                     for p in dataset[:]]

                            if key_name in result:
                                existing_paths = set(result[key_name])
                                existing_paths.update(paths)
                                result[key_name] = sorted(list(existing_paths))
                            else:
                                result[key_name] = sorted(list(set(paths)))

                        except Exception as e:
                            print(f"Warning: failed to process key {key_name}: {e}")

        return result

    def write_json_file(self, target_dir: str, json_data: dict, overwrite: bool = False):
        """Write task_paths.json into the episode directory."""
        target_dir = os.path.normpath(target_dir)
        json_file_path = os.path.join(target_dir, 'task_paths.json')

        if os.path.exists(json_file_path) and not overwrite:
            self.stats['files_skipped'] += 1
            return False

        if self.dry_run:
            if os.path.exists(json_file_path):
                self.stats['files_updated'] += 1
            else:
                self.stats['files_created'] += 1
            return True

        try:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                self.stats['directories_created'] += 1

            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            self.stats['files_created'] += 1
            return True

        except Exception as e:
            print(f"Failed to write {json_file_path}: {e}")
            self.stats['errors'] += 1
            return False

    def run(self, overwrite: bool = False, max_paths: int = None):
        path_mapping = self.extract_and_map_data()

        self.stats['total_paths'] = len(path_mapping)
        print(f"\nMapping complete. Found {self.stats['total_paths']} episode directories.")

        items_to_process = list(path_mapping.items())
        if max_paths:
            print(f"Test mode: processing first {max_paths} paths only.")
            items_to_process = items_to_process[:max_paths]

        mode_label = "Preview" if self.dry_run else "Write"
        print(f"\n[{mode_label}] Generating task_paths.json files...")

        with tqdm(total=len(items_to_process), desc="Writing task_paths.json") as pbar:
            for episode_dir, groups_data in items_to_process:
                json_content = self.generate_json_content(episode_dir, groups_data)
                self.write_json_file(episode_dir, json_content, overwrite)
                pbar.update(1)

    def print_report(self):
        print("\n" + "=" * 50)
        print("Execution Report")
        print("=" * 50)
        print(f"Total episode paths:  {self.stats['total_paths']}")
        print(f"Directories created:  {self.stats['directories_created']}")
        print(f"Files written:        {self.stats['files_created']}")
        print(f"Files skipped:        {self.stats['files_skipped']}")
        print(f"Errors:               {self.stats['errors']}")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Write task_paths.json into each episode directory listed in an HDF5 task dictionary.'
    )
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='HDF5 task dictionary produced by build_task_dictionary.py')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing task_paths.json files')
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview mode: do not write any files')
    parser.add_argument('--max_paths', type=int, default=None,
                        help='Process only the first N paths (for testing)')

    args = parser.parse_args()

    generator = TaskPathsGenerator(
        hdf5_path=args.hdf5_path,
        dry_run=args.dry_run,
    )

    generator.run(overwrite=args.overwrite, max_paths=args.max_paths)
    generator.print_report()
