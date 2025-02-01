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

