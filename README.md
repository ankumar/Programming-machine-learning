![Attention Timeline](https://github.com/ankumar/awesome-llm-architectures/assets/658791/6c70defb-8a21-4397-ba13-2b49783f0bb5)

Application Programming Interfaces (APIs) are at the heart of all internet software, evolving with Foundational Model [**APIs**](https://artificialanalysis.ai/). An Exploration of **Artificial** programming **[intelligence](http://www.incompleteideas.net/IncIdeas/DefinitionOfIntelligence.html)**  ...

## Compound AI Systems
**state-of-the-art AI results are increasingly obtained by compound systems with multiple components, not just monolithic models.**  
[The Shift from Models to Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)  

**Generative language models (LMs) are often chained together and combined with other components into compound AI systems. Compound AI system applications include retrieval-augmented generation (RAG). structured prompting, chatbot verification, multi-hop question answering, agents, and SQL query generation.**
[ALTO: An Efficient Network Orchestrator for Compound AI Systems](https://arxiv.org/pdf/2403.04311.pdf)  

<img width="607" alt="Screenshot 2024-03-25 at 1 44 32 PM" src="https://github.com/ankumar/Artificial-programming-intelligence/assets/658791/e21c5ef7-0515-4e06-a222-f2bed645cb9a">.  
A thread has a rich discussion of the AI vs Cloud Providers https://x.com/rakyll/status/1771641289840242754?s=20. The emerging AI Cloud is simpler to use, as we evolve Compound AI Systems **I hope we preserve the simplicity**.  
- **Chat + Assistants:**
  - [OpenAI GPT Actions (**ChatPlus subscription is needed to access**) ](https://chat.openai.com/g/g-7kq4uSfJ4-javelin) 
  - [HuggingChat](https://hf.co/chat/assistant/65bdbbf7f10680b82361aa45)

- **Tools:** 

- **Evaluation:** [https://openfeature.dev/](https://openfeature.dev/docs/reference/concepts/evaluation-context)

## Hardware & Accelerators

- Apple M1/M2/M3: Get a MacBook Pro M3 Max (CPU, GPU cores + up to 128GB Unified Memory) https://www.apple.com/shop/buy-mac/macbook-pro/16-inch 
  - pip install -U mlx
  - https://github.com/ml-explore
    
- NVIDIA: Hopper -> [Blackwell](https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief)

- [Trying to Build commodity ~petaflop compute node](https://tinygrad.org/) / https://x.com/karpathy/status/1770164518758633590?s=20

## Leaderboards, Benchmarks & Evaluations

Intelligence is the computational part of the **ability to achieve goals**. A goal achieving system is one that is more usefully understood in terms of outcomes than in terms of mechanisms.  
[The Definition of Intelligence](http://www.incompleteideas.net/IncIdeas/DefinitionOfIntelligence.html)

**We don't know how to measure LLM abilities well. Most tests are groups of multiple choice questions, tasks, or trivia - they don't represent real world uses well, they are subject to gaming & results are impacted by prompt design in unknown ways. Or they use human preference.** Non-trivial Taxonomy in real-world, starting with clear domains / Common LLM workloads:  

1) Languages - Rankings by domain -> https://huggingface.co/models -> **Tasks** & **Languages**
2) Model Card - [claude 3 model card](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf) Coding, Creative Writing, Instruction-following, Long Document Q&A
3) Chat, [RAG](https://trec-rag.github.io/about/), few-shot benchmark, etc.  
4) Coding - "Code a login component in React"
5) Freshness - "What was the Warriors game score last night?"  
6) Agent - Web Agents -> https://turkingbench.github.io/
7) Multimodal (images and video)
8) Reasoning? 

- [LMSYS Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
  - [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
  - [Predictive Human Preference(PHP)](https://huyenchip.com/2024/02/28/predictive-human-preference.html) / **Conclusion:** "LMSYS folks told me that due to the noisiness of crowd-sourced annotations and the costs of expert annotations, they’ve found that using GPT-4 to compare two responses works better. Depending on the complexity of the queries, generating 10,000 comparisons using GPT-4 would cost only $200 - 500, making this very affordable for companies that want to test it out."
 
- [AI2 WildBench Leaderboard](https://huggingface.co/spaces/allenai/WildBench)

- [Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models](https://arxiv.org/abs/2402.13887)  

- [SWE-bench coding benchmark](https://www.swebench.com/)
  - [SWE-bench Lite](https://www.swebench.com/lite)

- [Yale Semantic Parsing and Text-to-SQL Challenge](https://yale-lily.github.io/spider)    

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

### "Open" & Closed   

**Weights, Training & Inference Code, Data & Evaluation**  

[Tour of Modern LLMs (and surrounding topics)](https://phontron.com/class/anlp2024/assets/slides/anlp-15-tourofllms.pdf)

- [OLMo](https://allenai.org/olmo) / [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)

- [NOUS RESEARCH](https://nousresearch.com/forge/)

- [An awesome repository of local AI tools](https://github.com/janhq/awesome-local-ai)
  - [Jan - Rethink the Computer](https://jan.ai/)

- []()
  
--

- [Llama](https://llama.meta.com/)
  - [Discover the possibilities of building on Llama](https://llama.meta.com/community-stories/)
- [Gemma](https://opensource.googleblog.com/2024/02/building-open-models-responsibly-gemini-era.html)
- [Grok open release](https://github.com/xai-org/grok-1)
- [Robust recipes to align language models with human and AI preferences](https://github.com/huggingface/alignment-handbook)
- [A natural language interface for computers](https://github.com/KillianLucas/open-interpreter)
- [OpenDevin](https://github.com/OpenDevin/OpenDevin)

The successful art of model merging is often based purely on experience and intuition of a passionate model hacker, .... In fact, the current Open LLM Leaderboard is dominated by merged models. Surprisingly, merged models work without any additional training, making it very cost-effective (**no GPUs required at all!**), and so many people, researchers, hackers, and hobbyists alike, are trying this to create the best models for their purposes. 
- [Evolving New Foundation Models: Unleashing the Power of Automating Model Development](https://sakana.ai/evolutionary-model-merge/)


- []()

![image](https://github.com/ankumar/Artificial-programming-intelligence/assets/658791/8eaba9f3-ec3d-4803-a7e6-4b6864039c0a)
[What I learned from looking at 900 most popular open source AI tools](https://huyenchip.com/2024/03/14/ai-oss.html)

- [Little guide to building Large Language Models in 2024](https://docs.google.com/presentation/d/1IkzESdOwdmwvPxIELYJi8--K3EZ98_cL6c5ZcLKSyVg/edit?usp=sharing)

- [PeoplePlusAI - Where AI meets people and purpose](https://github.com/PeoplePlusAI)

### Inferencing

**Gateway API Concepts:** https://gateway-api.sigs.k8s.io/
**Use Cases:** https://gateway-api.sigs.k8s.io/#use-cases

Cloud Providers: In the ever-evolving cloud computing landscape, understanding the Gateway API is crucial for those using Kubernetes. This API enhances application traffic management, offering better routing and security. For seamless integration of AI into cloud-native applications, a robust framework 
 ? streamlining the deployment and management of AI-driven solutions. Dive into the Gateway API for insights and explore Use Cases for cutting-edge application management.

![image](https://github.com/ankumar/artificial-programming-intelligence/assets/658791/73be23c5-6812-42c6-822c-2bec13ffce52)

- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index)
- [vLLM](https://github.com/vllm-project/vllm)
- [LLM inference in C/C++](https://github.com/ggerganov/llama.cpp)
- [Baseten](https://github.com/basetenlabs) / https://github.com/basetenlabs/truss
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) / (Use Case: https://www.perplexity.ai/hub/blog/introducing-pplx-api)
- []()
- []()
- []()

### Training

**abstract away cloud infra burdens, Launch jobs & clusters on any cloud, Maximize GPU usage**  
- [SkyPilot: A framework for running ML and batch jobs on any cloud cost-effectively](https://github.com/skypilot-org)
- []()
- []()

### Prompt

- [Anthropic Prompt library](https://docs.anthropic.com/claude/prompt-library) / [Prompt engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Prompt Engineering with Llama 2](https://github.com/facebookresearch/llama-recipes/blob/main/examples/Prompt_Engineering_with_Llama_2.ipynb)
- 
  
### Programming—not prompting—Language Models

- [Stanford DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy)

- [Stanford NLP Python Library for Understanding and Improving PyTorch Models via Interventions](https://github.com/stanfordnlp/pyvene)

### Frameworks & Scripting

- [LangChain OpenGPTs](https://github.com/langchain-ai/opengpts) / [Elixir implementation of a LangChain style framework](https://github.com/brainlid/langchain) / [LangChain for Go](https://github.com/tmc/langchaingo)
- [GPTScript](https://github.com/gptscript-ai/gptscript)
-

### UI/UX

#### Hugging Face

- [Demo your machine learning model](https://www.gradio.app/)
  
- [Chat UI](https://github.com/huggingface/chat-ui)  
  - [Hugging Face's Chat Assistants](https://huggingface.co/chat/assistants)
  - [Open source implementation of the OpenAI 'data analysis mode' (aka ChatGPT + Python execution) based on Mistral-7B](https://github.com/xingyaoww/code-act) / [Hugging Face for Chat UI, ProjectJupyter for code executor](https://chat.xwang.dev)

- [Tokenizer Playground - How different models tokenize text](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)

#### Vercel
- [AI RSC Demo](https://sdk.vercel.ai/demo)
  - [Prompt Playground](https://sdk.vercel.ai/prompt) Like Chatbot Arena https://chat.lmsys.org/
  - [AI SDK](https://sdk.vercel.ai/)
  - [v0](https://v0.dev/)

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
- [Demystifying Embedding Spaces using Large Language Models](https://arxiv.org/abs/2310.04475)
- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/pdf/2304.12244.pdf)
- [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://arxiv.org/abs/2401.08406)
- [Multi-line AI-assisted Code Authoring](https://huggingface.co/papers/2402.04141)
- [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/pdf/2402.14207.pdf)
- []()
- []()
- []()

## Learning
Artificial intelligence has recently experienced remarkable advances, fueled by large models, vast datasets, accelerated hardware, and, last but not least, the transformative power of differentiable programming. This new programming paradigm enables end-to-end differentiation of complex computer programs (including those with control flows and data structures), making gradient-based optimization of program parameters possible.  
As an emerging paradigm, differentiable programming builds upon several areas of computer science and applied mathematics, including automatic differentiation, graphical models, optimization and statistics. This book presents a comprehensive review of the fundamental concepts useful for differentiable programming. We adopt two main perspectives, that of optimization and that of probability, with clear analogies between the two.  
Differentiable programming is not merely the differentiation of programs, but also the thoughtful design of programs intended for differentiation. By making programs differentiable, we inherently introduce probability distributions over their execution, providing a means to quantify the uncertainty associated with program outputs.
[The Elements of Differentiable Programming (Draft, ~380 Pages!)](https://arxiv.org/pdf/2403.14606.pdf)

**A best place to learn all in one place** https://huggingface.co/docs & [Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/index)

- [Machine Learning cheatsheets for Stanford's CS 229](https://github.com/afshinea/stanford-cs-229-machine-learning/blob/master/en/README.md)

- [Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization](https://github.com/karpathy/minbpe)
  - [Byte Pair Encoding: building the GPT tokenizer with Karpathy](https://francescopochetti.com/byte-pair-encoding-building-the-gpt-tokenizer-with-karpathy/)

- Why not implement this in PyTorch?
  - https://github.com/ml-explore/mlx/issues/12

- [AI GUIDE](https://ai-guide.future.mozilla.org/)

- [A little guide to building Large Language Models in 2024](https://docs.google.com/presentation/d/1IkzESdOwdmwvPxIELYJi8--K3EZ98_cL6c5ZcLKSyVg/edit?usp=sharing)

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

- [Large Language Model Course](https://github.com/mlabonne/llm-course)

- [RAGStack](https://docs.datastax.com/en/ragstack/docs/index.html)

- [Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines](https://github.com/explodinggradients/ragas)

- [Easily use and train state of the art retrieval methods in any RAG pipeline. Designed for modularity and ease-of-use, backed by research](https://github.com/bclavie/RAGatouille)

- [Building the open-source feedback layer for LLMs](https://github.com/argilla-io)

- [Data Provenance Explorer](https://dataprovenance.org/)
- 

--

![(12) United States Patent](https://github.com/ankumar/awesome-llm-architectures/assets/658791/d5c14f6e-8242-4a64-b721-82f2944f3241) 

![GD7LKuvCY5w8OtmsOMASxS5mpCBUvkGdXxeYMSSYx3ZrOogCkN0GOdoBvCD5DK-1t64R31RpIJhUMYAhmRQJb7KuK1rFLSRiVPGXfPGrwQxorwybqc7rE-F7nVTH](https://github.com/ankumar/awesome-llm-architectures/assets/658791/bc4e7567-bddf-4a6c-9b96-236067a6bf71)

- [Foundation models](https://docs.google.com/document/d/1POj8OKdKRYYnhPF_OwPVpCnv-xVGkYCS0Hw_OmOUNRo/edit?usp=sharing)

- ["Natural Language APIs"](https://docs.google.com/document/d/1E-sZ60oS5Iw8rZaxImInCdERm4ZNhBlfqJWpGbfT9KQ/edit?usp=sharing)

- [System Design](https://docs.google.com/document/d/1lL2VEVRs574OecdgTsYz-4ygnEhzJc2y8wgvPjhaW2Y/edit?usp=sharing)
