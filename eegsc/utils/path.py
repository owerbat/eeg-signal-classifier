import os


def get_data_path():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    if not os.path.exists(path):
        os.makedirs(path)
    return path
