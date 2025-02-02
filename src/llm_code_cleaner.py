import re
from typing import Optional


class LLMCodeCleaner:
    """
    A class to clean and extract valid Python code from LLM responses.

    This class is designed to remove Markdown formatting (e.g. triple backticks)
    and extract a specific function (named 'load_model') from the LLM response,
    by extracting the entire indented block of code following the function
    definition.

    Assumptions:
      - The LLM returns code wrapped in Markdown code blocks.
      - The code contains a function named 'load_model'.
      - The code uses consistent indentation (spaces or tabs).
    """

    @staticmethod
    def clean_code(llm_code: str) -> str:
        """
        Cleans the LLM's response by removing Markdown backticks and extracting
        the 'load_model' function block by capturing the entire indented block.

        Args:
            llm_code (str): The raw code returned by the LLM.

        Returns:
            str: The cleaned code ready for execution. If the 'load_model'
                 function is not found, an empty string is returned.

        Raises:
            TypeError: If llm_code is not a string.
        """
        # Validate input type.
        if not isinstance(llm_code, str):
            raise TypeError("llm_code must be a string")

        # Remove leading triple backticks and an optional language specifier.
        llm_code = re.sub(r'^```(?:\w+)?\n', '', llm_code).strip()
        # Remove trailing triple backticks (possibly preceded by whitespace).
        llm_code = re.sub(r'\n?```$', '', llm_code).strip()

        # Split the code into individual lines and expand tabs for consistency.
        lines = [line.expandtabs(4) for line in llm_code.splitlines()]

        # Find the function definition line for 'load_model'.
        start_index: Optional[int] = None
        base_indent: Optional[int] = None
        for i, line in enumerate(lines):
            if re.match(r'^\s*def\s+load_model\s*\(', line):
                start_index = i
                base_indent = len(line) - len(line.lstrip())
                break

        if start_index is None or base_indent is None:
            return ""

        # Collect the function definition and all lines that belong to its block.
        cleaned_lines = [lines[start_index]]
        for line in lines[start_index + 1:]:
            # Always include empty lines.
            if line.strip() == "":
                cleaned_lines.append(line)
                continue

            current_indent = len(line) - len(line.lstrip())
            # Include lines that are more indented than the function definition.
            if current_indent > base_indent:
                cleaned_lines.append(line)
            else:
                break

        return "\n".join(cleaned_lines)


# class LLMCodeCleaner:
#     """
#     A class to clean and extract valid Python code from LLM responses.

#     This class is designed to remove Markdown formatting (e.g. triple backticks)
#     and extract a specific function (named 'load_model') from the LLM response,
#     up to its first return statement with the expected indentation.

#     Assumptions:
#       - The LLM returns code wrapped in Markdown code blocks.
#       - The code contains a function named 'load_model'.
#       - Indentation is assumed to be 4 spaces.
#     """

#     @staticmethod
#     def clean_code(llm_code: str) -> str:
#         """
#         Cleans the LLM's response by removing Markdown backticks and extracting
#         the 'load_model' function block up to its return statement.

#         Args:
#             llm_code (str): The raw code returned by the LLM.

#         Returns:
#             str: The cleaned code ready for execution. If the 'load_model'
#                  function or its expected return statement is not found, an empty
#                  string is returned.

#         Raises:
#             TypeError: If llm_code is not a string.
#         """
#         # Validate input type.
#         if not isinstance(llm_code, str):
#             raise TypeError("llm_code must be a string")

#         # Remove leading triple backticks and an optional language specifier.
#         llm_code = re.sub(r'^```(?:\w+)?\n', '', llm_code).strip()
#         # Remove trailing triple backticks (possibly preceded by whitespace).
#         llm_code = re.sub(r'\n?```$', '', llm_code).strip()

#         # Split the code into individual lines.
#         lines = llm_code.splitlines()
#         cleaned_lines = []
#         in_main_function = False
#         function_indent: Optional[int] = None

#         for line in lines:
#             # Look for the start of the 'load_model' function definition.
#             if not in_main_function:
#                 if re.match(r'^\s*def\s+load_model\s*\(', line):
#                     in_main_function = True
#                     function_indent = len(line) - len(line.lstrip())
#                     cleaned_lines.append(line)
#                     continue
#             else:
#                 # Already inside the target function; add the line.
#                 cleaned_lines.append(line)
#                 # Check for a 'return' statement with the expected indentation level.
#                 current_indent = len(line) - len(line.lstrip())
#                 if (line.lstrip().startswith("return") and 
#                     function_indent is not None and 
#                     current_indent == function_indent + 4):
#                     break

#         return "\n".join(cleaned_lines)


# Example usage:
if __name__ == "__main__":
    raw_llm_code = """
    ```python
    def load_model():
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Define base models
        base_models = [
            ('xgb', XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8)),
            ('lgbm', LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8))
        ]

        # Define the meta-learner
        meta_learner = Pipeline([
            ('scaler', StandardScaler()),  # Scale features for Logistic Regression
            ('lr', LogisticRegression(C=1.0, max_iter=200))
        ])

        # Build the stacking classifier
        model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5  # Use 5-fold cross-validation for stacking
        )
        return model

    def extra_function():
        print("This should not be part of the output.")

    ```
    """
    cleaner = LLMCodeCleaner()
    cleaned_code = cleaner.clean_code(raw_llm_code)
    print(cleaned_code)
