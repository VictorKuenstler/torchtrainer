import os


def check_file_exists(file):
    return os.path.isfile(file)


def remove_file(file):
    os.remove(file)


def get_num_lines(file):
    return sum(1 for _ in open(file))
