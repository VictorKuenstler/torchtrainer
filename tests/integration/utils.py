import os
import shutil


def check_file_exists(file):
    return os.path.isfile(file)


def remove_file(file):
    os.remove(file)


def get_num_lines(file):
    return sum(1 for _ in open(file))


def create_test_directory(directory):
    os.mkdir(directory)


def delete_folder(directory):
    shutil.rmtree(directory)


def get_num_files_in_directory(directory):
    return len([name for name in os.listdir(directory) if name.endswith('.pt')])
