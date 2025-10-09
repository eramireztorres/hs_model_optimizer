import argparse
import functools
from inspect import signature, Parameter
from typing import List, Literal, get_type_hints


import re

def extract_arg_descriptions(docstring):
    """Extract argument descriptions from a function's docstring."""
    # Extract content under 'Args:'
    arg_section = re.search(r"Args:(.*?)(Returns:|$)", docstring, re.DOTALL)
    if not arg_section:
        return {}

    # Split individual arguments
    args = re.findall(r"- (\w+) \((.*?)\): (.*?)(- |\n|$)", arg_section.group(1), re.DOTALL)

    # Return a dictionary with argument names as keys and their descriptions as values
    return {arg[0]: arg[2].strip() for arg in args}


def cli_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        type_hints = get_type_hints(func)
        # parser = argparse.ArgumentParser(description=func.__doc__)
        parser = argparse.ArgumentParser()
        
        # Extract argument descriptions from the docstring
        arg_descriptions = extract_arg_descriptions(func.__doc__ or "")

        reserved_shorthands = {'-h'}  # initialize with known shorthands

        for name, param in sig.parameters.items():
            # Convert pythonic snake_case to hyphenated-case for CLI
            cli_name = name.replace('_', '-')

            # Create short versions based on initials of words separated by underscores
            initials = '-' + ''.join([word[0] for word in name.split('_')])
            if initials in reserved_shorthands:
                initials = None  # skip this shorthand

            is_list = (
                name.endswith('_list') or 
                (name in type_hints and (
                    type_hints[name] == list or 
                    (hasattr(type_hints[name], '__origin__') and type_hints[name].__origin__ is list)
                ))
            )

            choices = None
            if name in type_hints and hasattr(type_hints[name], '__args__'):
                # Check if the type hint is a Literal
                if getattr(type_hints[name], '__origin__', None) == Literal:
                    choices = type_hints[name].__args__
            
            arg_kwargs = {
                'type': str,
                'choices': choices
            }
            
            if name in type_hints:
                if type_hints[name] == int:
                    arg_kwargs['type'] = int
                elif type_hints[name] == float:
                    arg_kwargs['type'] = float
                        
            if param.default == Parameter.empty:  # If no default is provided, set as required
                arg_kwargs['required'] = True


            if is_list:
                arg_kwargs['nargs'] = '*'

            arg_flags = [f'--{cli_name}']  # Note the use of cli_name here
            if initials:
                arg_flags.append(initials)
            arg_kwargs['default'] = param.default
            # arg_kwargs['help'] = f"(default: {param.default})"
            arg_kwargs['dest'] = name
            if name in arg_descriptions:
                arg_help = arg_descriptions[name]
                if param.default != Parameter.empty:
                    arg_help += f" (default: {param.default})"
                arg_kwargs['help'] = arg_help
            
            action = parser.add_argument(*arg_flags, **arg_kwargs)
            
            # add the generated short version to reserved_shorthands to avoid future conflicts
            if initials:
                reserved_shorthands.add(initials)

        parsed_args = vars(parser.parse_args())
        # Convert the CLI args back to function args by replacing hyphens with underscores
        func_args = {k.replace('-', '_'): v for k, v in parsed_args.items()}
        
        return func(**func_args)

    return wrapper

import unittest
from unittest.mock import patch
from io import StringIO
import sys

class TestCliDecorator(unittest.TestCase):

    def run_cli(self, func, cli_args):
        """Utility function to emulate CLI argument passing and capture the output."""
        sys.argv = ["test_program_name"] + cli_args.split()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            func()
            return mock_stdout.getvalue().strip()

    def test_positional_args(self):
        @cli_decorator
        def mock_function(name, age):
            print(f"{name} is {age} years old")

        output = self.run_cli(mock_function, "--name Alice --age 30")
        self.assertEqual(output, "Alice is 30 years old")

    def test_default_args(self):
        @cli_decorator
        def mock_function(color="red"):
            print(f"Color: {color}")

        output = self.run_cli(mock_function, "")
        self.assertEqual(output, "Color: red")

        output = self.run_cli(mock_function, "--color blue")
        self.assertEqual(output, "Color: blue")

    def test_list_args(self):
        @cli_decorator
        def mock_function(fruits_list=None):
            for fruit in fruits_list:
                print(f"Fruit: {fruit}")

        output = self.run_cli(mock_function, "--fruits-list apple orange")
        self.assertEqual(output, "Fruit: apple\nFruit: orange")

    def test_literal_args(self):
        @cli_decorator
        def mock_function(choice: Literal["A", "B", "C"]):
            print(f"Choice: {choice}")

        output = self.run_cli(mock_function, "--choice A")
        self.assertEqual(output, "Choice: A")

        # Test that an invalid choice raises an error
        with self.assertRaises(SystemExit):
            self.run_cli(mock_function, "--choice D")
            
    def test_numeric_type_hints(self):
        @cli_decorator
        def mock_numeric_function(a: int, b: float):
            self.assertIsInstance(a, int)  # assert that 'a' is of type int
            self.assertIsInstance(b, float)  # assert that 'b' is of type float
            print(f"{a} and {b}")

        output = self.run_cli(mock_numeric_function, "--a 10 --b 3.14")
        self.assertEqual(output, "10 and 3.14")

        # Test that invalid numeric values raise an error
        with self.assertRaises(SystemExit):
            self.run_cli(mock_numeric_function, "--a ten --b 3.14")

        with self.assertRaises(SystemExit):
            self.run_cli(mock_numeric_function, "--a 10 --b three-point-one-four")


if __name__ == '__main__':
    unittest.main()

