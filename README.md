# 📣 Advancing Multimodal Emotion Recognition in Comics via Multi-disciplinary, Multi-Task, Multi-Lingual Transfer Learning 📣

## ℹ️ Summary
This work proposes a sequential transfer learning framework for multimodal emotion recognition in comics, integrating multidisciplinary, multi-task, and multi-lingual adaptation within a unified pipeline. Leveraging large-scale vision–language models, we progressively refine representations through cross-domain supervision from fine art (EmoArt), auxiliary visual question answering (EMID), and multilingual adaptation using Japanese manga (MangaVQA). The staged transfer process promotes domain-invariant affective features, strengthens visual–linguistic grounding, and enhances cross-cultural robustness. In addition, we investigate merged models obtained from sequentially fine-tuned adapters to consolidate knowledge across tasks while preserving stability. Extensive experiments demonstrate consistent improvements across architectures, highlighting the effectiveness of structured sequential transfer learning and model merging over single-stage adaptation. Our results advance robust, context-aware emotion understanding in comics and broader multimodal narrative media.

## 🪜 Goals
- Develop a sequential transfer learning framework for multimodal emotion recognition in comics that integrates multidisciplinary, multi-task, and multi-lingual adaptation within a unified training pipeline.
- Enhance cross-modal representation quality by leveraging auxiliary supervision (fine art emotion datasets and visual question answering) to promote domain-invariant and semantically grounded affective features.
- Evaluate cross-lingual and cross-cultural generalization through adaptation to Japanese manga, assessing robustness to linguistic and stylistic variation..
- Analyze stage-wise and merged model configurations to quantify the cumulative impact of sequential transfer and determine its effectiveness across different vision–language architectures..

<!-- ## Key Features
- Multimodal input: images, OCR text, speech-bubble layouts, metadata.
- Multi-task outputs: emotion labels, intensity scores, character-wise emotion attribution, sentiment, sarcasm/irony flags, action-context tags.
- Multilingual NLP: tokenization, translation, and language-specific emotion taxonomies.
- Multidisciplinary priors: visual affect cues, comic-specific conventions, cultural annotations.
- Extensible model zoo: vision encoders, multimodal transformers, graph-based scene models, multimodal LLMs for generative explanations. -->

## Datasets
- Recommended sources: public comic datasets, crowdsourced annotations, simulated augmentations.
- Annotation schema: canonical emotion set (configurable), intensity (0–1), actor ID, confidence, cultural/context tags, language tag.
- Quality: inter-annotator agreement, adjudication workflow, balanced sampling across languages and art styles.

## Models and Architecture
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



## Training Strategy
- Pretrain vision and language encoders on broad multimodal corpora.
- Multi-task finetuning with task-specific heads and weighted loss scheduling.
- Curriculum learning: visual cues → text cues → joint reasoning.
- Data augmentation: style transfer, synthetic speech-bubbles, translation augmentation.

## Evaluation
- Metrics: accuracy/F1 for classification, MAE/RMSE for intensity, precision/recall for attribution, BLEU/ROUGE for generated explanations.
- Robustness tests: cross-style, cross-language, occlusion, noisy-OCR resilience.
- Human evaluation: qualitative assessment for generated explanations and cultural appropriateness.

## Project Structure (suggested)
- data/ — dataset manifests, annotation formats
- src/preproc/ — OCR, layout, panel segmentation
- src/models/ — model definitions and training scripts
- src/eval/ — metrics and evaluation suites
- experiments/ — config files, checkpoints, logs
- docs/ — annotation guidelines, taxonomy, API docs

## Ethics & Responsible Use
- Respect copyright and creator rights for comic content; use licensed or public-domain corpora.
- Document dataset provenance and annotation guidelines.
- Monitor for cultural bias in emotion labels and translations; include mitigation and audit steps.
- Include opt-out and privacy guidance for sensitive content if using real-world material.

# 📦 Requirements

We use the following versions of the packages:

```
torch==2.4.0
gradio==4.43.0
pydantic==2.9.0
LLaMA-Factory==0.9.0
transformers==4.44.2
bitsandbytes==0.43.1
```

For fine-tuning, you need to install LLaMA-Factory. Run the following command to install LLaMA-Factory and all the necessary dependencies and updates:

```
bash setup.sh
```

<br>

# 💻 Platform and Compute

- For fine-tuning LLMs, we use [**LLaMA-Factory.**](https://github.com/hiyouga/LLaMA-Factory)
- For model checkpoints, we use [**Unsloth.**](https://unsloth.ai/)
- We also use [**Hugging Face.**](https://huggingface.co/)

All experiments have been performed on the High Performance Cluster at [**La Rochelle Université.**](https://www.univ-larochelle.fr/)
