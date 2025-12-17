from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

      name: str
      sample_name2: str = "Nihal"  # default value --> new_student1 = {}
      age: Optional[int] = None
      email: EmailStr # Pydantic has built-in types like EmailStr to validate email format, soif the email is not correct then it will raise an error
      cgpa: float = Field(gt=0, lt=10, default=6, description = "Cumulative Grade Point Average (Decimal)")  # Field constraint will put the range (greater than 0 and less than 10), if not range then it will raise an error

# new_student1 = {"name": "Nihal", "age": '21', "email": "nihal@example.com"} # Pydantic is smart will convert age to int (type casting)
# or
# new_student1 = Student(name="Nihal") # in this case we do not need to create an object.
# new_student1 = {"name": 21} # will raise an error, as name should be a string. In case of Pydantic it check if the answer that we are getting is correct or not. 

new_student1 = {"name": "Nihal", "email": "nihal@gmail.com", "cgpa": 8}


#
obj = Student(**new_student1)

print(obj)
# print(type(Student))
# print(new_student1.keys())

# ---- Dictionary Conversion ----
# student_dict = dict(obj)
# print(student_dict)
# print(type(student_dict))
# print(student_dict['email'])

# ---- JSON Conversion ----
# student_json = obj.model_dump_json(indent=2)
# print(student_json)
# print(type(student_json))