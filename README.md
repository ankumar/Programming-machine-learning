## Leaderboards, Benchmarks & Evaluations

- [Artificial Analysis](https://artificialanalysis.ai/)

--

**We don't know how to measure LLM abilities well. Most tests are groups of multiple choice questions, tasks, or trivia - they don't represent real world uses well, they are subject to gaming & results are impacted by prompt design in unknown ways. Or they use human preference.**   

Non-trivial Taxonomy in real-world, Starting with clear domains;Common LLM workloads:  
1) Languages  
2) Coding - "Code a login component in React"
3) Freshness - "What was the Warriors game score last night?"  
4) Multimodal (images and video)  
5) Agent, Reasoning, Chat, RAG, few-shot benchmark,  
6) In [claude 3 model card](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) Coding, Creative Writing, Instruction-following, Long Document Q&A
7) Rankings by domain -> https://huggingface.co/models -> **Tasks** & **Languages**

- [LMSYS Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
  - [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
  - [Predictive Human Preference(PHP)](https://huyenchip.com/2024/02/28/predictive-human-preference.html) / **Conclusion:** "LMSYS folks told me that due to the noisiness of crowd-sourced annotations and the costs of expert annotations, they’ve found that using GPT-4 to compare two responses works better. Depending on the complexity of the queries, generating 10,000 comparisons using GPT-4 would cost only $200 - 500, making this very affordable for companies that want to test it out."  

- [Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models](https://arxiv.org/abs/2402.13887)  

- [SWE-bench coding benchmark](https://www.swebench.com/)
- [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [A holistic framework for evaluating foundation models](https://crfm.stanford.edu/helm/lite/latest/)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)
- [Collections](https://huggingface.co/collections/clefourrier/leaderboards-and-benchmarks-64f99d2e11e92ca5568a7cce)
- [Functional Benchmarks and the Reasoning Gap](https://github.com/ConsequentAI/fneval)
- [CodeMind is a generic framework for evaluating inductive code reasoning of LLMs. It is equipped with a static analysis component that enables in-depth analysis of the results.](https://github.com/Intelligent-CAT-Lab/CodeMind)
- [OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement](https://github.com/OpenCodeInterpreter)
- [Yet Another Applied LLM Benchmark](https://github.com/carlini/yet-another-applied-llm-benchmark)
- [We’re doing cutting edge research for transparent, auditable AI alignment](https://www.synthlabs.ai/)
"Current methods of “alignment” are insufficient;evaluations are even worse. Human intent reflects a rich tapestry of preferences, collapsed by uniform models. AI`s potential hinges on trust, from interpretable data to every layer built upon it. Informed decisions around risk are not binary. Training on raw human data doesn’t scale. Your models should adapt and scale, automatically." -> [Suppressing Pink Elephants with Direct Principle Feedback](https://arxiv.org/abs/2402.07896)

- []()

--

- [OpenAI Evals](https://github.com/openai/evals)
- [Martian Model Router](https://docs.withmartian.com/martian-model-router) / [OpenAI Evals](https://github.com/withmartian/martian-evals)
- [Intelligent Language Model Router](https://docs.pulze.ai/overview/introductions)
- [Evaluate-Iterate-Improve](https://github.com/uptrain-ai/uptrain)
- [Open-Source Evaluation for GenAI Application Pipelines](https://github.com/relari-ai/continuous-eval)

--

- [A large-scale, fine-grained, diverse preference dataset](https://github.com/OpenBMB/UltraFeedback)
- [Generate Synthetic Data Using OpenAI, MistralAI or AnthropicAI](https://github.com/migtissera/Sensei)

## "Open" vs Closed Models  

**Weights, Training & Inference Code, Data & Evaluation**  
- [OLMo](https://allenai.org/olmo) / [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)

---
- [Llama](https://llama.meta.com/)
  - [Discover the possibilities of building on Llama](https://llama.meta.com/community-stories/)
- [Gemma](https://opensource.googleblog.com/2024/02/building-open-models-responsibly-gemini-era.html)
- [Grok open release](https://github.com/xai-org/grok-1)
- [Robust recipes to align language models with human and AI preferences](https://github.com/huggingface/alignment-handbook)
- [A natural language interface for computers](https://github.com/KillianLucas/open-interpreter)
- [OpenDevin](https://github.com/OpenDevin/OpenDevin)
- []()
- []()
- [PeoplePlusAI - Where AI meets people and purpose](https://github.com/PeoplePlusAI)
- [What I learned from looking at 900 most popular open source AI tools](https://huyenchip.com/2024/03/14/ai-oss.html)
- [Little guide to building Large Language Models in 2024](https://docs.google.com/presentation/d/1IkzESdOwdmwvPxIELYJi8--K3EZ98_cL6c5ZcLKSyVg/edit?usp=sharing)

## Inferencing

- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index)
- [vLLM](https://github.com/vllm-project/vllm)
- [LLM inference in C/C++](https://github.com/ggerganov/llama.cpp)
- [Baseten](https://github.com/basetenlabs) / https://github.com/basetenlabs/truss
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) / (Use Case: https://www.perplexity.ai/hub/blog/introducing-pplx-api)
- []()
- []()
- []()

## Training

**abstract away cloud infra burdens, Launch jobs & clusters on any cloud, Maximize GPU usage**  
- [SkyPilot: A framework for running ML and batch jobs on any cloud cost-effectively](https://github.com/skypilot-org)
- []()
- []()

## Compound AI Systems

**state-of-the-art AI results are increasingly obtained by compound systems with multiple components, not just monolithic models**  

- [The Shift from Models to Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
- [ALTO: An Efficient Network Orchestrator for Compound AI Systems](https://arxiv.org/pdf/2403.04311.pdf)
- []()
- []()

### Prompt
- [Anthropic Prompt library](https://docs.anthropic.com/claude/prompt-library)
- [Prompt Engineering with Llama 2](https://github.com/facebookresearch/llama-recipes/blob/main/examples/Prompt_Engineering_with_Llama_2.ipynb)
- 

### Hugging Face Chat UI
- [Chat UI](https://github.com/huggingface/chat-ui)  
  - [Hugging Face's Chat Assistants](https://huggingface.co/chat/assistants)
  - [Open source implementation of the OpenAI 'data analysis mode' (aka ChatGPT + Python execution) based on Mistral-7B](https://github.com/xingyaoww/code-act) / [Hugging Face for Chat UI, ProjectJupyter for code executor](https://chat.xwang.dev)
- [Tokenizer Playground - How different models tokenize text](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)

### Vercel Generative UI
- [AI RSC Demo](https://sdk.vercel.ai/demo)
  - [Prompt Playground](https://sdk.vercel.ai/prompt) Like Chatbot Arena https://chat.lmsys.org/
  - [AI SDK](https://sdk.vercel.ai/)
  - [v0](https://v0.dev/)
  
### Programming—not prompting—Language Models
- [Stanford DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy)
- [Stanford NLP Python Library for Understanding and Improving PyTorch Models via Interventions](https://github.com/stanfordnlp/pyvene)

### Frameworks & Scripting
- [LangChain OpenGPTs](https://github.com/langchain-ai/opengpts) / [Elixir implementation of a LangChain style framework](https://github.com/brainlid/langchain) / [LangChain for Go](https://github.com/tmc/langchaingo)
- [GPTScript](https://github.com/gptscript-ai/gptscript)
-

## Standards
- [Establishing industry wide AI best practices and standards for AI Engineers](https://github.com/AI-Engineer-Foundation)
- [The Data Provenance Initiative](https://github.com/Data-Provenance-Initiative/Data-Provenance-Collection)

## Security
- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://llm-attacks.org/)
- [How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs](https://chats-lab.github.io/persuasive_jailbreaker/)
- [Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations](https://www.nist.gov/news-events/news/2024/01/nist-identifies-types-cyberattacks-manipulate-behavior-ai-systems)
- []()
- []()

## Articles & Talks
- [LLM App Stack aka Emerging Architectures for LLM Applications](https://github.com/a16z-infra/llm-app-stack)
- [A Guide to Large Language Model Abstractions](https://www.twosigma.com/articles/a-guide-to-large-language-model-abstractions/)
- [AI Fundamentals: Benchmarks 101](https://www.latent.space/p/benchmarks-101)
- [The Modern AI Stack: Design Principles for the Future of Enterprise AI Architectures](https://menlovc.com/perspective/the-modern-ai-stack-design-principles-for-the-future-of-enterprise-ai-architectures/)
- [AI Copilot Interfaces](https://byrnemluke.com/ideas/llm-interfaces)
- [Evaluating LLMs is a minefield](https://www.cs.princeton.edu/~arvindn/talks/evaluating_llms_minefield/)
- [Large Language Models and Theories of Meaning](https://drive.google.com/file/d/15oXHUBaUWwhFziB1G2Tg5kQ1PPJ3XqP7/view)

## AI Twitter & Discord
- [@karpathy](https://twitter.com/karpathy)
- [@simonw](https://twitter.com/simonw)
- []()
- [@antirez](https://twitter.com/antirez/status/1746857737584099781)
- [@ivanfioravanti](https://twitter.com/ivanfioravanti)

## Papers

- [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)
- [On the Planning Abilities of Large Language Models : A Critical Investigation](https://arxiv.org/abs/2305.15771)
- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/pdf/2304.12244.pdf)
- [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://arxiv.org/abs/2401.08406)
- [Multi-line AI-assisted Code Authoring](https://huggingface.co/papers/2402.04141)
- [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/pdf/2402.14207.pdf)
- []()
- []()
- []()

## Learning

**A best place to learn all in one place** https://huggingface.co/docs & [Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/index)

- [Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization](https://github.com/karpathy/minbpe)
  - [Byte Pair Encoding: building the GPT tokenizer with Karpathy](https://francescopochetti.com/byte-pair-encoding-building-the-gpt-tokenizer-with-karpathy/)

- Why not implement this in PyTorch?
  - https://github.com/ml-explore/mlx/issues/12

- [AI GUIDE](https://ai-guide.future.mozilla.org/)
- [A little guide to building Large Language Models in 2024](https://docs.google.com/presentation/d/1IkzESdOwdmwvPxIELYJi8--K3EZ98_cL6c5ZcLKSyVg/edit?usp=sharing)
- [Large Language Model Course](https://github.com/mlabonne/llm-course)
- [RAGStack](https://docs.datastax.com/en/ragstack/docs/index.html)
- [Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines](https://github.com/explodinggradients/ragas)
- [Easily use and train state of the art retrieval methods in any RAG pipeline. Designed for modularity and ease-of-use, backed by research](https://github.com/bclavie/RAGatouille)
- [Building the open-source feedback layer for LLMs](https://github.com/argilla-io)
- [Data Provenance Explorer](https://dataprovenance.org/)

