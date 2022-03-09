from typing import List
from scipy.stats import logistic

from numpy import max, concatenate, ndarray, array, zeros

from conversation_metrics.structures.utterance import Utterance
from conversation_metrics.metrics.tit_for_tat import TitForTat
from conversation_metrics.models.custom_models import customise_models

customise_models(
    measure_formality=None,
    measure_sentiment=None,
    extract_entities=None,
    vectorise=None,
)


class Encoder:
    """
    a feature encoder based on conversation quality metrics
    """

    @staticmethod
    def batch_encode(text: str, summaries: List[str]) -> List[ndarray]:
        text_features = Utterance(text=text, utterance_index=0)
        text_embedding = Encoder.create_feature_vector(text_features)
        return list(
            map(
                lambda summary: Encoder.encode(
                    summary_features=Utterance(text=summary, utterance_index=0),
                    text_features=text_features,
                    text_embedding=text_embedding,
                ),
                summaries,
            )
        )

    @staticmethod
    def encode(
        text_embedding: ndarray, text_features: Utterance, summary_features: Utterance
    ) -> ndarray:
        summary_embedding = Encoder.create_feature_vector(summary_features)
        return concatenate(
            [
                text_embedding,
                summary_embedding,
                Encoder.get_comparative_features_via_quality_metric(
                    text_features, summary_features
                ),
                Encoder.get_comparative_features_via_absolute_difference(
                    text_embedding, summary_embedding
                ),
                Encoder.get_comparative_features_via_product(
                    text_embedding, summary_embedding
                ),
            ]
        )

    @staticmethod
    def create_feature_vector(features: Utterance) -> ndarray:
        return concatenate(
            [
                Encoder.get_semantic_features(features),
                Encoder.get_length_related_features(features),
                Encoder.get_other_features(features),
            ]
        )

    @staticmethod
    def get_semantic_features(features: Utterance) -> ndarray:
        if not any(features.entities):
            return zeros(300)
        return max(
            array(list(map(lambda entity: entity.semantics, features.entities))), axis=0
        )

    @staticmethod
    def get_length_related_features(features: Utterance) -> List[float]:
        return [
            logistic.cdf(len(features.text)),
            logistic.cdf(len(features.text.split())),
            logistic.cdf(len(features.entities)),
        ]

    @staticmethod
    def get_other_features(features: Utterance) -> List[float]:
        return [features.formality, features.sentiment]

    @staticmethod
    def get_comparative_features_via_quality_metric(
        text_features: Utterance, summary_features: Utterance
    ) -> List[float]:
        overlap_score = text_features.mean_semantic_overlap(summary_features)
        return [
            overlap_score,
            TitForTat().measure(
                formality_prior=text_features.formality,
                formality_after=summary_features.formality,
                sentiment_prior=text_features.sentiment,
                sentiment_after=summary_features.sentiment,
                mean_semantic_overlap=overlap_score,
            ),
        ]

    @staticmethod
    def get_comparative_features_via_absolute_difference(
        text_embedding: ndarray, summary_embedding: ndarray
    ) -> ndarray:
        return abs(text_embedding - summary_embedding)

    @staticmethod
    def get_comparative_features_via_product(
        text_embedding: ndarray, summary_embedding: ndarray
    ) -> ndarray:
        return text_embedding * summary_embedding
