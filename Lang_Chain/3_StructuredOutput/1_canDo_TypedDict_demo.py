from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person1 : Person = {'name': 'Nihal', 'age': 21}

new_person2 : Person = {'name': 'Niair', 'age': '21'} # both works fine

print(new_person1)
print(new_person2)