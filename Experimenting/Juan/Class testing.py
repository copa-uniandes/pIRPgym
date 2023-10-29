from typing import Union


class testing():

    def __init__(self,number):
        self.number = number


class testing1():
    @staticmethod
    def test_function(testing_object:testing):
        print(testing_object.number)