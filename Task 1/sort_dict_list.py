def sort_dict_list(dict_list, key, reverse=False):
    """
    Sort a list of dictionaries by a specified key.
    
    Args:
        dict_list (list): List of dictionaries to sort
        key (str): Dictionary key to sort by
        reverse (bool): Sort in descending order if True
    
    Returns:
        list: Sorted list of dictionaries
    """
    try:
        return sorted(dict_list, key=lambda x: x[key], reverse=reverse)
    except KeyError:
        raise KeyError(f"Key '{key}' not found in one or more dictionaries")

# Example usage
if __name__ == "__main__":
    data = [
        {"name": "John", "age": 30},
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 35}
    ]
    sorted_data = sort_dict_list(data, "age")
    print(sorted_data)