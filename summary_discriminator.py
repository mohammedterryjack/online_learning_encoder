from typing import List, Optional, Any, Dict
from json import dumps, load
from os.path import exists

from numpy import ndarray, array
from sklearn.neural_network import MLPRegressor

from website_summariser.encoder import Encoder
from website_summariser.utils import DiscriminatorSettings, LiveLearningSettings


class SummaryDiscriminator:
    """
    learn to evaluate summaries
    by aligning to human feedback
    """

    def __init__(self, filepath: str = DiscriminatorSettings.PATH.value) -> None:
        self.text: str = ""
        self.summaries: List[str] = list()
        self.scores: List[float] = list()
        self.modelpath = filepath
        self.model = MLPRegressor(
            solver=DiscriminatorSettings.SOLVER.value,
            hidden_layer_sizes=DiscriminatorSettings.LAYER_SIZE.value,
        )
        if exists(filepath):
            self.model.__dict__ = self._load_model_parameters(filepath)

    def evaluate(self, text: str, summaries: List[str]) -> None:
        self.text = text
        self.summaries = summaries
        self.scores = self.model.predict(
            Encoder.batch_encode(text=text, summaries=summaries)
        )

    def get_best(self) -> Optional[str]:
        if any(self.scores):
            best_index = self.scores.argmax()
            return self.summaries[best_index]

    def _align(self, text: str, summaries: List[str], human_choice: List[bool]) -> None:
        inputs = Encoder.batch_encode(text=text, summaries=summaries)
        targets = list(map(float, human_choice))
        self.model.partial_fit(inputs, targets)
        self._save_model_parameters()

    def _save_model_parameters(self) -> None:
        model_json = dumps(
            self.model.__dict__,
            indent=2,
            default=lambda value: value.tolist()
            if isinstance(value, ndarray)
            else None,
        )
        with open(self.modelpath, "w") as model_file:
            model_file.write(model_json)

    @staticmethod
    def _load_model_parameters(modelpath: str) -> Dict[str, Any]:
        with open(modelpath) as model_file:
            model_parameters = load(model_file)
        for key, value in model_parameters.items():
            if isinstance(value, list) and isinstance(value[0], list):
                model_parameters[key] = array(
                    list(map(lambda sublist: array(sublist), value)), dtype=object
                )
        model_parameters.pop(DiscriminatorSettings.OPTIMISER.value, None)
        return model_parameters

    @staticmethod
    def _get_human_choices(text: str, summaries: List[str]) -> List[bool]:
        number_of_options = len(summaries)
        user_choices = [False] * number_of_options
        head = LiveLearningSettings.MESSAGE_EXTRACT.value.format(extract=text)
        body = LiveLearningSettings.OPTION_DELIMITER.value.join(
            map(
                lambda index, summary: LiveLearningSettings.MESSAGE_SUMMARY.value.format(
                    index=index, summary=summary
                ),
                range(number_of_options),
                summaries,
            )
        )
        tail = LiveLearningSettings.MESSAGE_SELECT.value
        message = LiveLearningSettings.OPTION_DELIMITER.value.join((head, body, tail))
        valid_choice = False
        while not valid_choice:
            choices = input(message)

            if not any(choices):
                valid_choice = True
                continue

            try:
                indexes = list(
                    map(int, choices.split(LiveLearningSettings.CHOICE_DELIMITER.value))
                )
            except ValueError:
                print(LiveLearningSettings.MESSAGE_INVALID_TYPE.value)
                continue

            try:
                for index in indexes:
                    user_choices[index] = True
                valid_choice = True
            except IndexError:
                print(LiveLearningSettings.MESSAGE_INVALID_VALUE.value)
                continue

        return user_choices
