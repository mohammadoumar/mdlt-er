# 📣 Advancing Multimodal Emotion Recognition in Comics via Multi-disciplinary, Multi-Task, Multi-Lingual Transfer Learning 📣

# ℹ️ Summary
This work proposes a sequential transfer learning framework for multimodal emotion recognition in comics, integrating multidisciplinary, multi-task, and multi-lingual adaptation within a unified pipeline. Leveraging large-scale vision–language models, we progressively refine representations through cross-domain supervision from fine art (EmoArt), auxiliary visual question answering (EMID), and multilingual adaptation using Japanese manga (MangaVQA). The staged transfer process promotes domain-invariant affective features, strengthens visual–linguistic grounding, and enhances cross-cultural robustness. In addition, we investigate merged models obtained from sequentially fine-tuned adapters to consolidate knowledge across tasks while preserving stability. Extensive experiments demonstrate consistent improvements across architectures, highlighting the effectiveness of structured sequential transfer learning and model merging over single-stage adaptation. Our results advance robust, context-aware emotion understanding in comics and broader multimodal narrative media.

# 🪜 Goals
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

# 🧮 Datasets

We experiment with four datasets:

- EmoComics35: A comics dataset with emotion annotations at utterance level, for multimodal emotion classification.
- EmoArt: A fine art emotion dataset annotated with a canonical emotion taxonomy, arousal and valence annotations, attributes analysis, and art description, enabling cross-domain affect modeling. -- [**EmoArt**](https://zhiliangzhang.github.io/EmoArt-130k/#)
- Emotionally paired Music and Image Dataset (EMID): A multimodal dataset pairing images with emotionally aligned music clips, annotated with discrete emotion labels. -- [**EMID**](https://huggingface.co/datasets/orrzohar/EMID-Emotion-Matching)
- MangaVQA: A Japanese manga-based visual question answering dataset containing panel-level image–question–answer triples, designed to evaluate multimodal reasoning and cross-lingual visual–textual grounding in narrative contexts. --  [**MangaVQA**](https://huggingface.co/datasets/hal-utokyo/MangaVQA)

# ⛓️ Models

We experiment with the following models:

- **LLaMA-Vision** -- LLaMA-3.2-Vision-11B-Instruct, LLaMA-3.2-Vision-90B-Instruct -- [**Meta AI**](https://huggingface.co/meta-llama)

- **Qwen-VL** -- Qwen-2.5-VL-7B-Instruct, Qwen-2.5-VL-72B-Instruct, Qwen-3-VL-8B-Instruct -- [**Qwen**](https://huggingface.co/Qwen)

-  **LLaVA** -- LLaVA-1.5, LLaVA-NeXT -- [**Llava Hugging Face**](https://huggingface.co/llava-hf)

- **Pixtral** -- [**Mistral AI**](https://huggingface.co/mistralai)

<br>


# 🎛️ Training Strategy
- Initialize large-scale vision–language models (e.g., Qwen-VL, LLaMA-Vision, Pixtral, LLaVA) and establish a multimodal baseline on EmoComics35 for panel-level emotion recognition.
- Apply multi-disciplinary sequential transfer learning by first adapting the model on EmoArt to learn domain-invariant affective primitives (color, composition, stylization), followed by refinement on comics data.
- Introduce multi-task supervision through sequential fine-tuning on EMID (VQA), strengthening visual–linguistic grounding and contextual reasoning before re-specializing for emotion classification.
- Extend to a multilingual transfer stage via adaptation on MangaVQA (Japanese), promoting cross-lingual semantic abstraction and cross-cultural robustness.
- Perform model merging across sequential stages to integrate accumulated knowledge from multi-disciplinary, multi-task, and multilingual adaptations, and evaluate both stage-wise and merged configurations to quantify cumulative transfer gains.

# 🔧 Evaluation
- Stage-wise Sequential Transfer Evaluation: Systematically evaluate model performance after each sequential transfer learning stage (multimodal baseline, multi-disciplinary, multi-task, multilingual) to quantify incremental gains, stability, and generalization using accuracy and macro-F1. Controlled ablations assess the individual contribution of each transfer dimension.
- Merged Model and Cross-Architecture Assessment: Evaluate merged models that integrate knowledge from multiple transfer stages, comparing them against stage-wise counterparts across different vision–language architectures to measure cumulative knowledge integration, robustness, and cross-domain effectiveness.


# 📦 Requirements

We use the following versions of the packages:

```
accelerate>=1.12.0
bitsandbytes>=0.49.1
peft>=0.18.1"
pillow>=12.1.1"
torch>=2.7.0
transformers>=4.51.1
unsloth==2026.2.1
LLaMA-Factory==0.9.5
```

<br>

# 💻 Platform and Compute

- For fine-tuning LLMs, we use [**LLaMA-Factory.**](https://github.com/hiyouga/LLaMA-Factory)
- For model checkpoints, we use [**Unsloth.**](https://unsloth.ai/)
- We also use [**Hugging Face.**](https://huggingface.co/)

All experiments have been performed on the High Performance Cluster at [**La Rochelle Université.**](https://www.univ-larochelle.fr/)


# Ethics & Responsible Use
Datasets and models are the property of their respective owners. Please consult their official documentation and websites for licensing terms, usage restrictions, and responsible AI policies prior to use.
