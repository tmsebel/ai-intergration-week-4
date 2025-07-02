def sort_by_name(dict_list):
    return sorted(dict_list, key=lambda x: x['name'])

people = [
    {'name': 'Tawana', 'age': 30},
    {'name': 'Banele', 'age': 25},
    {'name': 'Siyabonga', 'age': 10}
]

sorted_people = sort_by_name(people)
sorted_by_age = sorted(people, key=lambda x: x['age'])
print(sorted_people)
print(sorted_by_age)
