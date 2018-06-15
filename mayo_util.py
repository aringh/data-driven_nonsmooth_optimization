"""Helper function for reading and randomizing order of training data."""

import os
import random

# Path to data folder
DATA_FOLDER = "Give path to data folder"


class FileLoader(object):
    """Helper class for reading the data."""

    def __init__(self, folder, exclude):
        self.folder = folder
        self.exclude = exclude
        self.load_files()

    def load_files(self):
        self.files = []
        for (dirpath, dirnames, filenames) in os.walk(self.folder):
            self.files.extend([os.path.join(self.folder, fi) for fi in filenames
                               if not fi.startswith(self.exclude)])
        random.shuffle(self.files)

    def next_file(self):
        if not self.files:
            self.load_files()
        return self.files.pop()
