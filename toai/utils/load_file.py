import pickle  # nosec


def load_file(filename, mode="rb"):
    with open(filename, mode=mode) as f:
        return pickle.load(f)  # nosec
