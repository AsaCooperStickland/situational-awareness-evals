from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from wandb.apis.public import Run


class Model(ABC):
    name: str

    @staticmethod
    def from_id(model_id: str, **kwargs) -> "Model":
        from sitaevals.models.openai_complete import OpenAIAPI

        return OpenAIAPI(model_name=model_id, **kwargs)

    @abstractmethod
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        pass

    @abstractmethod
    def generate(
        self, inputs: Union[str, List[str]], max_tokens: int, **kwargs
    ) -> List[str]:
        pass

    @abstractmethod
    def cond_log_prob(
        self, inputs: Union[str, List[str]], targets, **kwargs
    ) -> List[List[float]]:
        pass

    @abstractmethod
    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List["Run"]:
        pass
