# Task 1.4: Sequence Model Failure Analysis

As part of the evaluation of our scratch character-level GPT model trained on the TinyStories dataset, we analyzed three specific failure cases observed in the generated text samples.

## Failure Case 1: Random Character Gibberish / Hallucination
- **Generated Snippet:** `i3?RjBF&n:swoukrqs.`
- **Failure Type:** Loss of coherence / Lexical hallucination
- **Observation:** In the early epochs and low-parameter regime, the model generates strings of punctuation and disconnected alphanumeric tokens. This occurs because the character-level model hasn't fully captured English word structures or morphemes, resulting in high perplexity outputs.

## Failure Case 2: Severe Character Repetition
- **Generated Snippet:** `mmmmmmhvl` / `sssscmhvl`
- **Failure Type:** Repetition / Local loop trapping
- **Observation:** The self-attention mechanism occasionally gets trapped in local probability peaks where a single character (or small character cluster) dominates the next-token prediction distribution due to insufficient training epochs or lack of strong regularization/temperature tuning.

## Failure Case 3: Broken Syntax and Fragmented Words
- **Generated Snippet:** `fpAZhLNAwjqbSTjqVs&rYgvhSlJUrkJRWG`
- **Failure Type:** Broken grammar / Structural breakdown
- **Observation:** While the model learns character transition constraints to some extent, long-range dependencies break down past the context block size limit, leading to uppercase/lowercase mingling and total syntax fragmentation.
