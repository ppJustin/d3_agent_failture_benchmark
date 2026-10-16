"""Generate local experiment taxonomy files from a separately downloaded dataset."""
import argparse
import json
from pathlib import Path
import yaml

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'whowhen_pro/taxonomy.yaml')
    args = parser.parse_args()
    taxonomy = yaml.safe_load(args.source.read_text())
    if not isinstance(taxonomy, dict) or not all(isinstance(v, dict) and 'name' in v and 'description' in v for v in taxonomy.values()):
        raise ValueError('Expected a mapping of error codes to name/description')
    for name in ['attribution', 'v0_test', 'v0_image', 'v0_image_gui', 'v0_video']:
        folder = root/name
        if folder.is_dir():
            path = folder/'taxonomy.json'
            if path.exists():
                print('Keep existing:', path)
            else:
                path.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2)+'\n')
                print('Generated:', path)

if __name__ == '__main__':
    main()
