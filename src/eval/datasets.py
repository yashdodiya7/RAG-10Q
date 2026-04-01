"""
Test datasets for evaluation.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalDataset:
    """Evaluation dataset container."""
    questions: List[str]
    answers: List[str]
    contexts: List[List[str]]
    ground_truths: List[str]


TESLA_10Q_DATASET = EvalDataset(
    questions=[
        'Which company filed the Form 10-K annual report for the fiscal year ended December 31, 2025?',
        'What is the Commission File Number listed in the document?',
        'Where is the principal executive office of the registrant located?',
        'What is the telephone number of the registrant?',
        'What is the trading symbol for the common stock?',
        'On which exchange is the common stock registered?',
        'Is the registrant indicated as a well-known seasoned issuer?',
        'What state is listed as the jurisdiction of incorporation or organization?',
        'Are there any securities registered pursuant to Section 12(g) of the Act?',
        'Is the registrant required to file reports pursuant to Section 13 or 15(d)?'
    ],
    answers=[
        'Tesla, Inc. filed the Form 10-K annual report for the fiscal year ended December 31, 2025.',
        'The Commission File Number listed is 001-34756.',
        'The principal executive office is located at 1 Tesla Road, Austin, Texas 78725.',
        "The registrant's telephone number is (512) 516-8177.",
        'The trading symbol for the common stock is TSLA.',
        'The common stock is registered on The Nasdaq Global Select Market.',
        'Yes, the registrant is indicated as a well-known seasoned issuer.',
        'The jurisdiction of incorporation or organization listed is Texas.',
        'No, there are no securities registered pursuant to Section 12(g) of the Act.',
        'Yes, the registrant is required to file reports pursuant to Section 13 or 15(d).'
    ],
    contexts=[
        ['FORM 10-K ... For the fiscal year ended December 31, 2025 ... Tesla, Inc. (Exact name of registrant as specified in its charter)'],
        ['Commission File Number: 001-34756'],
        ['1 Tesla Road Austin, Texas 78725 (Address of principal executive offices) (Zip Code)'],
        ['(512) 516-8177 (Registrant\'s telephone number, including area code)'],
        ['Common stock TSLA The Nasdaq Global Select Market'],
        ['Title of each class Trading Symbol(s) Name of each exchange on which registered Common stock TSLA The Nasdaq Global Select Market'],
        ['Indicate by check mark whether the registrant is a well-known seasoned issuer ... Yes x No o'],
        ['Texas 91-2197729 (State or other jurisdiction of incorporation or organization)'],
        ['Securities registered pursuant to Section 12(g) of the Act: None'],
        ['Indicate by check mark if the registrant is not required to file reports pursuant to Section 13 or 15(d) of the Act. Yes o No x']
    ],
    ground_truths=[
        'Tesla, Inc.',
        '001-34756',
        '1 Tesla Road, Austin, Texas 78725',
        '(512) 516-8177',
        'TSLA',
        'The Nasdaq Global Select Market',
        'Yes',
        'Texas',
        'None',
        'Yes'
    ]
)
class AutomotiveDataset:
    """Container for automotive evaluation datasets."""
    
    @staticmethod
    def create_empty() -> EvalDataset:
        """Create empty automotive evaluation dataset."""
        return EvalDataset(
            questions=[],
            answers=[],
            contexts=[],
            ground_truths=[]
        )
    
    @staticmethod
    def from_sources(
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> EvalDataset:
        """Create dataset from provided sources."""
        return EvalDataset(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths
        )
