import pickle  # nosec


def save_file(obj, filename, mode="wb"):
    with open(filename, mode=mode) as f:
        pickle.dump(obj, f)
