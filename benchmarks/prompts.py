"""Benchmark prompt suite for speculative decoding evaluation.

Diverse prompts across categories with varying output lengths to test
acceptance rate and speedup under different generation conditions.
"""

from dataclasses import dataclass


@dataclass
class BenchmarkPrompt:
    name: str
    category: str
    prompt_text: str
    max_new_tokens: int


PROMPTS = [
    # --- Code Generation ---
    BenchmarkPrompt(
        name="merge_sort",
        category="code_generation",
        prompt_text="Write a Python function that implements merge sort with type hints and docstrings:\n\n```python\ndef merge_sort(",
        max_new_tokens=256,
    ),
    BenchmarkPrompt(
        name="binary_tree",
        category="code_generation",
        prompt_text="Implement a binary search tree in Python with insert, search, and delete methods:\n\n```python\nclass BSTNode:",
        max_new_tokens=512,
    ),
    BenchmarkPrompt(
        name="http_server",
        category="code_generation",
        prompt_text="Write a simple HTTP server in Python using only the socket library that handles GET requests:\n\n```python\nimport socket\n\ndef start_server(",
        max_new_tokens=256,
    ),
    BenchmarkPrompt(
        name="json_parser",
        category="code_generation",
        prompt_text="Write a recursive descent JSON parser in Python:\n\n```python\nclass JSONParser:\n    def __init__(self, text: str):",
        max_new_tokens=512,
    ),

    # --- Summarization ---
    BenchmarkPrompt(
        name="summarize_ml",
        category="summarization",
        prompt_text=(
            "Neural networks are computing systems inspired by biological neural networks "
            "that constitute animal brains. An artificial neural network is based on a "
            "collection of connected units or nodes called artificial neurons, which loosely "
            "model the neurons in a biological brain. Each connection, like the synapses in "
            "a biological brain, can transmit a signal to other neurons. An artificial neuron "
            "receives signals then processes them and can signal neurons connected to it. "
            "The signal at a connection is a real number, and the output of each neuron is "
            "computed by some non-linear function of the sum of its inputs. Neurons and "
            "connections typically have a weight that adjusts as learning proceeds. The weight "
            "increases or decreases the strength of the signal at a connection.\n\n"
            "Summarize the above passage in 3 concise sentences:"
        ),
        max_new_tokens=128,
    ),
    BenchmarkPrompt(
        name="summarize_history",
        category="summarization",
        prompt_text=(
            "The Industrial Revolution, which took place from the 18th to 19th centuries, "
            "was a period during which predominantly agrarian, rural societies in Europe and "
            "America became industrial and urban. Prior to the Industrial Revolution, which "
            "began in Britain in the late 1700s, manufacturing was often done in people's "
            "homes, using hand tools or basic machines. Industrialization marked a shift to "
            "powered, special-purpose machinery, factories and mass production. The iron and "
            "textile industries, along with the development of the steam engine, played central "
            "roles in the Industrial Revolution, which also saw improved systems of "
            "transportation, communication and banking.\n\n"
            "Provide a brief summary in 2-3 sentences:"
        ),
        max_new_tokens=128,
    ),
    BenchmarkPrompt(
        name="summarize_science",
        category="summarization",
        prompt_text=(
            "Photosynthesis is a process used by plants and other organisms to convert light "
            "energy into chemical energy that, through cellular respiration, can later be "
            "released to fuel the organism's activities. Some of this chemical energy is stored "
            "in carbohydrate molecules, such as sugars and starches, which are synthesized from "
            "carbon dioxide and water. In most cases, oxygen is released as a waste product. "
            "Most plants, algae, and cyanobacteria perform photosynthesis; such organisms are "
            "called photoautotrophs. Photosynthesis is largely responsible for producing and "
            "maintaining the oxygen content of the Earth's atmosphere.\n\n"
            "Summarize in 2 sentences:"
        ),
        max_new_tokens=64,
    ),

    # --- Question Answering ---
    BenchmarkPrompt(
        name="qa_physics",
        category="qa",
        prompt_text="Question: What is the speed of light in a vacuum, and why is it considered a fundamental constant in physics?\n\nAnswer:",
        max_new_tokens=128,
    ),
    BenchmarkPrompt(
        name="qa_math",
        category="qa",
        prompt_text="Question: What is the Pythagorean theorem and how is it used to find the length of a side of a right triangle?\n\nAnswer:",
        max_new_tokens=128,
    ),
    BenchmarkPrompt(
        name="qa_cs",
        category="qa",
        prompt_text="Question: What is the difference between a stack and a queue data structure? Give an example use case for each.\n\nAnswer:",
        max_new_tokens=128,
    ),
    BenchmarkPrompt(
        name="qa_biology",
        category="qa",
        prompt_text="Question: How does DNA replication work in eukaryotic cells?\n\nAnswer:",
        max_new_tokens=256,
    ),

    # --- Creative Writing ---
    BenchmarkPrompt(
        name="mystery_opening",
        category="creative_writing",
        prompt_text="Write the opening paragraph of a mystery novel set in Tokyo during cherry blossom season:\n\n",
        max_new_tokens=256,
    ),
    BenchmarkPrompt(
        name="scifi_scene",
        category="creative_writing",
        prompt_text="Write a scene where an astronaut discovers something unexpected on Mars:\n\n",
        max_new_tokens=256,
    ),
    BenchmarkPrompt(
        name="poem",
        category="creative_writing",
        prompt_text="Write a short poem about the beauty of mathematics:\n\n",
        max_new_tokens=128,
    ),
    BenchmarkPrompt(
        name="dialogue",
        category="creative_writing",
        prompt_text="Write a dialogue between a time traveler from the year 2200 and a medieval blacksmith:\n\n",
        max_new_tokens=256,
    ),

    # --- Reasoning ---
    BenchmarkPrompt(
        name="math_word_problem",
        category="reasoning",
        prompt_text=(
            "A train leaves Station A at 9:00 AM traveling at 60 mph. Another train leaves "
            "Station B, 300 miles away, at 10:00 AM traveling toward Station A at 90 mph. "
            "At what time do the trains meet? Let's think step by step:\n\n"
        ),
        max_new_tokens=256,
    ),
    BenchmarkPrompt(
        name="logic_puzzle",
        category="reasoning",
        prompt_text=(
            "There are three boxes. One contains only apples, one contains only oranges, "
            "and one contains both apples and oranges. The boxes are labeled, but all labels "
            "are wrong. You can pick one fruit from one box. How can you determine the "
            "contents of all boxes? Let's reason through this:\n\n"
        ),
        max_new_tokens=256,
    ),
    BenchmarkPrompt(
        name="probability",
        category="reasoning",
        prompt_text=(
            "If you roll two fair six-sided dice, what is the probability that the sum is "
            "greater than 8? Show your work step by step:\n\n"
        ),
        max_new_tokens=256,
    ),
    BenchmarkPrompt(
        name="algorithm_analysis",
        category="reasoning",
        prompt_text=(
            "Explain why the time complexity of merge sort is O(n log n) in all cases. "
            "Walk through the analysis step by step:\n\n"
        ),
        max_new_tokens=256,
    ),

    # --- Translation ---
    BenchmarkPrompt(
        name="translate_en_fr",
        category="translation",
        prompt_text=(
            "Translate the following English text to French:\n\n"
            "The rapid advancement of artificial intelligence has transformed many industries, "
            "from healthcare to finance. Machine learning algorithms can now analyze vast "
            "amounts of data and make predictions with remarkable accuracy.\n\n"
            "French translation:\n"
        ),
        max_new_tokens=128,
    ),
    BenchmarkPrompt(
        name="translate_en_es",
        category="translation",
        prompt_text=(
            "Translate the following English text to Spanish:\n\n"
            "Climate change is one of the most pressing challenges facing humanity today. "
            "Rising global temperatures are causing more frequent extreme weather events "
            "and threatening ecosystems worldwide.\n\n"
            "Spanish translation:\n"
        ),
        max_new_tokens=128,
    ),
    BenchmarkPrompt(
        name="translate_en_de",
        category="translation",
        prompt_text=(
            "Translate the following English text to German:\n\n"
            "The history of computing stretches back thousands of years to ancient "
            "calculating devices like the abacus. Modern electronic computers emerged "
            "in the mid-20th century and have since revolutionized every aspect of human life.\n\n"
            "German translation:\n"
        ),
        max_new_tokens=128,
    ),
]


def get_prompts_by_category(category: str) -> list[BenchmarkPrompt]:
    return [p for p in PROMPTS if p.category == category]


def get_sweep_subset() -> list[BenchmarkPrompt]:
    """Return a smaller subset of prompts for hyperparameter sweeps."""
    categories = ["code_generation", "summarization", "qa", "creative_writing", "reasoning", "translation"]
    subset = []
    for cat in categories:
        prompts = get_prompts_by_category(cat)
        if prompts:
            subset.append(prompts[0])
    return subset


CATEGORIES = sorted(set(p.category for p in PROMPTS))
