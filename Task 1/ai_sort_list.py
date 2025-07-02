def ai_sort_dict_list(dict_list, key):
    result = dict_list.copy()
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if result[i][key] > result[j][key]:
                result[i], result[j] = result[j], result[i]
    return result
# Example usage
data = [
    {"name": "John", "age": 30},
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 35}
]
sorted_data = ai_sort_dict_list(data, "age")
print(sorted_data)