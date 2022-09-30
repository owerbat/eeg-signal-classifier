import os



def make_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_data_path():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    return make_dir(path)
