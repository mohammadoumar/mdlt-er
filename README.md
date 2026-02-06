# ComicEmotion — Multidisciplinary Multimodal Emotion Recognition for Comics

## Summary
ComicEmotion is a research-oriented framework for detecting and analyzing emotions in comics using multimodal generative AI. The project integrates multidisciplinary knowledge (art, linguistics, psychology), multi-task pipelines (emotion classification, intensity, character attribution, scene context), and multilingual support to handle speech balloons, captions, and visual cues.

## Goals
- Robust emotion recognition across panels, pages, and entire narratives.
- Jointly model visual, textual, and layout signals.
- Support multiple languages and cross-lingual transfer for low-resource comics.
- Provide modular experiments for research and prototyping.

## Key Features
- Multimodal input: images, OCR text, speech-bubble layouts, metadata.
- Multi-task outputs: emotion labels, intensity scores, character-wise emotion attribution, sentiment, sarcasm/irony flags, action-context tags.
- Multilingual NLP: tokenization, translation, and language-specific emotion taxonomies.
- Multidisciplinary priors: visual affect cues, comic-specific conventions, cultural annotations.
- Extensible model zoo: vision encoders, multimodal transformers, graph-based scene models, multimodal LLMs for generative explanations.

## Architecture (high-level)
- Preprocessing: panel extraction, speech-bubble detection, OCR, layout parsing.
- Representation: image features, text embeddings, spatial/layout encodings, character graphs.
- Core models:
    - Vision + Language encoders for classification and retrieval.
    - Graph Neural Networks for character interaction modeling.
    - Multimodal generative models for explanation and counterfactuals.
- Heads:
    - Emotion classification (categorical)
    - Emotion intensity/regression
    - Character attribution / coreference resolution
    - Multilingual normalization & translation
    - Explanation generation (textual)

## Data & Annotation
- Recommended sources: public comic datasets, crowdsourced annotations, simulated augmentations.
- Annotation schema: canonical emotion set (configurable), intensity (0–1), actor ID, confidence, cultural/context tags, language tag.
- Quality: inter-annotator agreement, adjudication workflow, balanced sampling across languages and art styles.

## Training Strategy
- Pretrain vision and language encoders on broad multimodal corpora.
- Multi-task finetuning with task-specific heads and weighted loss scheduling.
- Curriculum learning: visual cues → text cues → joint reasoning.
- Data augmentation: style transfer, synthetic speech-bubbles, translation augmentation.

## Evaluation
- Metrics: accuracy/F1 for classification, MAE/RMSE for intensity, precision/recall for attribution, BLEU/ROUGE for generated explanations.
- Robustness tests: cross-style, cross-language, occlusion, noisy-OCR resilience.
- Human evaluation: qualitative assessment for generated explanations and cultural appropriateness.

## Ethics & Responsible Use
- Respect copyright and creator rights for comic content; use licensed or public-domain corpora.
- Document dataset provenance and annotation guidelines.
- Monitor for cultural bias in emotion labels and translations; include mitigation and audit steps.
- Include opt-out and privacy guidance for sensitive content if using real-world material.

## Project Structure (suggested)
- data/ — dataset manifests, annotation formats
- src/preproc/ — OCR, layout, panel segmentation
- src/models/ — model definitions and training scripts
- src/eval/ — metrics and evaluation suites
- experiments/ — config files, checkpoints, logs
- docs/ — annotation guidelines, taxonomy, API docs

## Quick Examples (conceptual)
- Panel-level emotion classification: image + OCR → emotion + intensity
- Character-centric timeline: sequence of panels → per-character emotion trajectory
- Multilingual normalization: raw text (any language) → canonical emotion labels

## Contributing
- Open contribution model: issue templates for data, models, ethics concerns.
- Require reproducible experiments, minimal working example, and dataset citations.
- Follow the annotation and coding style guides in docs/.

## Roadmap
- Expand multilingual taxonomies and low-resource transfer.
- Integrate multimodal LLMs for richer explanations and interactive analysis.
- Benchmark suite across styles, eras, and languages.

## License
- Add project license and dataset-specific licenses; default to an open research-friendly license (choose and document explicitly).
