from string import Template
from base_model_api import BaseModelAPI

class ErrorCorrector:
    def __init__(self, llm_model: "BaseModelAPI", prompt_path: str):
        self.llm_model = llm_model
        self.prompt_template = self._load_prompt(prompt_path)
        
    def get_error_fix(self, faulty_code: str, error_msg: str) -> str:
        template = Template(self.prompt_template)
        prompt = template.safe_substitute(
            faulty_code=faulty_code,
            error_msg=error_msg
        )
        return self.llm_model.get_response(prompt)
        
    def _load_prompt(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()



# from base_model_api import BaseModelAPI
# import logging

# class SafePromptFormatter:
#     @staticmethod
#     def escape_braces(content: str) -> str:
#         """Escape curly braces to prevent formatting issues"""
#         return content.replace('{', '{{').replace('}', '}}')

# class ErrorCorrector:
#     def __init__(self, llm_model: BaseModelAPI, prompt_path: str):
#         self.llm_model = llm_model
#         self.prompt_template = self._load_prompt(prompt_path)
#         self.formatter = SafePromptFormatter()

#     def get_error_fix(self, faulty_code: str, error_msg: str) -> str:
#         try:
#             # Sanitize inputs before formatting
#             safe_code = self.formatter.escape_braces(faulty_code)
#             safe_error = self.formatter.escape_braces(error_msg)
            
#             prompt = self.prompt_template.format(
#                 faulty_code=safe_code,
#                 error_msg=safe_error
#             )
#             response = self.llm_model.get_response(prompt)
#             return self._sanitize_response(response)
#         except Exception as e:
#             logging.error(f"Error correction failed: {str(e)}")
#             return ""

#     def _load_prompt(self, path: str) -> str:
#         try:
#             with open(path, 'r') as f:
#                 return f.read()
#         except FileNotFoundError:
#             logging.error(f"Prompt file not found: {path}")
#             return ""

#     def _sanitize_response(self, response: str) -> str:
#         """Remove any remaining escaped braces from the response"""
#         return response.replace('{{', '{').replace('}}', '}')


# from base_model_api import BaseModelAPI  


# class ErrorCorrector:
#     def __init__(self, llm_model: BaseModelAPI, prompt_path: str):
#         self.llm_model = llm_model
#         self.prompt_template = self._load_prompt(prompt_path)
        
#     def get_error_fix(self, faulty_code: str, error_msg: str) -> str:
#         prompt = self.prompt_template.format(
#             faulty_code=faulty_code,
#             error_msg=error_msg
#         )
#         return self.llm_model.get_response(prompt)
        
#     def _load_prompt(self, path: str) -> str:
#         # Similar to LLMImprover's prompt loading
#         with open(path, 'r') as f:
#             return f.read()