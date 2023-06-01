import json


def load_json(file_path):
    """
    Load a JSON file and return its contents as a Python object.
    :param file_path: Path of the json file
    :return: The loaded JSON data as a Python object.
    """
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
    return json_data
