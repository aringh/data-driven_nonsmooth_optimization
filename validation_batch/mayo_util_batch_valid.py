"""Utils for reading and randomizing data for validation."""

import os
import random

VALID_DATA_FOLDER = "Give folder to validation data"
# As can be seen in the training code, we have trained on all phantoms except
# phantom L286. Thus the path to the validation data folder should point to
# where phantom L286 is.


class FileLoader(object):
    def __init__(self, folder, exclude='files to exclude'):
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
