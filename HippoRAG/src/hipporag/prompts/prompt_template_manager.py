import os
import asyncio
import importlib.util
from string import Template
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, field, asdict

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplateManager:
    # templates_dir: Optional[str] = field(
    #     default=None, 
    #     metadata={"help": "Directory containing template scripts. Default to the `templates` dir under dir whether this class is defined."}
    # )

    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {"system": "system", "user": "user", "assistant": "assistant"},
        metadata={
            "help": "Mapping from default roles in prompte template files to specific LLM providers' defined roles."}
    )
    language: str = "EN"
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(
        init=False,
        default_factory=dict,
        metadata={
            "help": "A dict from prompt template names to templates. A prompt template can be a Template instance or a chat history which is a list of dict with content as Template instance."}
    )

    def __post_init__(self) -> None:
        """
        Initialize the templates directory and load templates.
        """
        # if self.templates_dir is None:
        #     current_file_path = os.path.abspath(__file__)
        #     package_dir = os.path.dirname(current_file_path)
        #     self.templates_dir = os.path.join(package_dir, "templates")
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)

        # abs path to dir where each *.py file (exclude __init__.py) contains a variable prompt_template (a str or a chat history with content as raw str for being converted to a Template)
        self.templates_dir = os.path.join(package_dir, "templates")

        self._load_templates()

    def _load_templates(self) -> None:
        """
        Load all templates from Python scripts in the templates directory.
        Uses language-specific files if available based on self.language.
        """
        if not os.path.exists(self.templates_dir):
            logger.error(f"Templates directory '{self.templates_dir}' does not exist.")
            raise FileNotFoundError(f"Templates directory '{self.templates_dir}' does not exist.")

        logger.info(f"Loading templates from directory: {self.templates_dir}")

        # 构建语言后缀映射 (例如: en -> _en, cn -> _cn)
        lang_suffix = f"_{self.language}" if self.language else ""

        # 第一遍扫描: 优先加载语言特定版本
        for filename in os.listdir(self.templates_dir):
            base_name, ext = os.path.splitext(filename)
            if ext == ".py" and filename != "__init__.py":
                # 检查是否是语言特定版本 (例如: ner_cn.py)
                if base_name.endswith(lang_suffix):
                    script_name = base_name[:-len(lang_suffix)]  # 获取原始名称 (例如: ner)
                    self._load_template_file(filename, script_name)

        # 第二遍扫描: 加载通用版本 (如果没有语言特定版本)
        for filename in os.listdir(self.templates_dir):
            base_name, ext = os.path.splitext(filename)
            if ext == ".py" and filename != "__init__.py":
                # 排除已加载的语言特定版本
                if not base_name.endswith(lang_suffix):
                    script_name = base_name
                    # 检查是否已有语言特定版本
                    if script_name not in self.templates:
                        self._load_template_file(filename, script_name)

    def _load_template_file(self, filename: str, script_name: str) -> None:
        """辅助方法: 加载单个模板文件"""
        try:
            try:
                module_name = f"src.hipporag.prompts.templates.{os.path.splitext(filename)[0]}"
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                module_name = f".prompts.templates.{os.path.splitext(filename)[0]}"
                module = importlib.import_module(module_name, 'hipporag')

            if not hasattr(module, "prompt_template"):
                logger.error(f"Module '{module_name}' does not define a 'prompt_template'.")
                raise AttributeError(f"Module '{module_name}' does not define a 'prompt_template'.")

            prompt_template = module.prompt_template
            logger.debug(f"Loaded template from {module_name}")

            if isinstance(prompt_template, Template):
                self.templates[script_name] = prompt_template
            elif isinstance(prompt_template, str):
                self.templates[script_name] = Template(prompt_template)
            elif isinstance(prompt_template, list) and all(
                    isinstance(item, dict) and "role" in item and "content" in item for item in prompt_template
            ):
                # Adjust roles based on the provided role mapping
                for item in prompt_template:
                    item["role"] = self.role_mapping.get(item["role"], item["role"])
                    item["content"] = item["content"] if isinstance(item["content"], Template) else Template(
                        item["content"])
                self.templates[script_name] = prompt_template
            else:
                raise TypeError(
                    f"Invalid prompt_template format in '{module_name}.py'. Must be a Template or List[Dict]."
                )

            logger.debug(f"Successfully loaded template '{script_name}' from '{module_name}.py'.")

        except Exception as e:
            logger.error(f"Failed to load template from '{module_name}.py': {e}")
            raise

    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        Render a template with the provided variables.

        Args:
            name (str): The name of the template.
            kwargs: Placeholder values for the template.

        Returns:
            Union[str, List[Dict[str, Any]]]: The rendered template or chat history.

        Raises:
            ValueError: If a required variable is missing.
        """
        template = self.get_template(name)
        if isinstance(template, Template):
            # Render a single string template
            try:
                result = template.substitute(**kwargs)
                logger.debug(f"Successfully rendered template '{name}' with variables: {kwargs}.")
                return result
            except KeyError as e:
                logger.error(f"Missing variable for template '{name}': {e}")
                raise ValueError(f"Missing variable for template '{name}': {e}")
        elif isinstance(template, list):
            # Render a chat history
            try:
                rendered_list = [
                    {"role": item["role"], "content": item["content"].substitute(**kwargs)}
                    for item in template
                ]
                logger.debug(f"Successfully rendered chat history template '{name}' with variables: {kwargs}.")
                return rendered_list
            except KeyError as e:
                logger.error(f"Missing variable in chat history template '{name}': {e}")
                raise ValueError(f"Missing variable in chat history template '{name}': {e}")

    def sync_render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        return asyncio.run(self.render(name, **kwargs))

    def list_template_names(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List[str]: A list of template names.
        """
        logger.info("Listing all available template names.")

        return list(self.templates.keys())

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        """
        Retrieve a template by name.

        Args:
            name (str): The name of the template.

        Returns:
            Union[Template, List[Dict[str, Any]]]: The requested template.

        Raises:
            KeyError: If the template is not found.
        """
        if name not in self.templates:
            logger.error(f"Template '{name}' not found.")
            raise KeyError(f"Template '{name}' not found.")
        logger.debug(f"Retrieved template '{name}'.")

        return self.templates[name]

    def print_template(self, name: str) -> None:
        """
        Print the prompt template string or chat history structure for the given template name.

        Args:
            name (str): The name of the template.

        Raises:
            KeyError: If the template is not found.
        """
        try:
            template = self.get_template(name)
            print(f"Template name: {name}")
            if isinstance(template, Template):
                print(template.template)
            elif isinstance(template, list):
                for item in template:
                    print(f"Role: {item['role']}, Content: {item['content']}")
            logger.info(f"Printed template '{name}'.")
        except KeyError as e:
            logger.error(f"Failed to print template '{name}': {e}")
            raise

    def is_template_name_valid(self, name: str) -> bool:
        return name in self.templates


