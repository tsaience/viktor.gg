import pickle

### converts champ names to lowercase, space, punction free rep
def process_champ_name(champ_name):
    champ_name_lower = champ_name.lower()
    champ_name_processed = "".join(champ_name_lower.split("'"))
    champ_name_processed_space = "".join(champ_name_processed.split(" "))
    champ_name_processed_and = champ_name_processed_space.split("&")[0]
    champ_name_processed_period = "".join(champ_name_processed_and.split("."))
    return champ_name_processed_period


# pickele utility functions
def export_pickle(dictionary, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        return data