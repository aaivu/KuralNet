import os
import runpy


def execute_extractors(extractors_dir):
    for filename in os.listdir(extractors_dir):
        if filename.endswith(".py") and filename != "master_extractor.py":
            file_path = os.path.join(extractors_dir, filename)
            runpy.run_path(file_path, run_name="__main__")


if __name__ == "__main__":
    execute_extractors("data/meta_extractors")
