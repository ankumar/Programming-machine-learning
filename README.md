# “Compound AI Systems”

This Article [The Shift from Models to Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) from Berkeley AI Research(BAIR) introduces the concept of Compound AI Systems where the focus is rapidly changing: state-of-the-art AI results are increasingly obtained by compound systems with multiple components, not just monolithic models. 

_Figuring out the best practices for developing compound AI systems is still an open question, but there are already exciting approaches to aid with design, end-to-end optimization, and operation. We believe that compound AI systems will remain the best way to maximize the quality and reliability of AI applications going forward, and may be one of the most important trends in AI in 2024._

![image](https://github.com/user-attachments/assets/d874e82f-9b89-487e-894c-1ed3b247294d)
<p align="center">Increasingly many new AI results are from compound systems.</p>

## Endpoints
 
started with this one endpoint text Input/Output 
- Completions (Legacy)		/completions

Then ChatGPT with messages Input/Output 
- Chat completions        	/chat/completions

- Image generation		/images/generations
- Text to speech			/audio/speech
- Speech to text			/audio/transcriptions
- Embeddings			/embeddings
- Moderation			/moderations
- Fine-tuning                 		/fine_tuning/jobs
- Batch				/files /batches

- Function Calling - Query Database, Send Emails etc.
- Response Formats - Building AGI with OpenAI's Structured Outputs API

Specification for the OpenAI API openai-openapi/openapi.yaml at master
Full coverage of OpenAI endpoints here: astra-assistants-api/coverage.md at main 

Specification for the Azure OpenAI API Azure OpenAI Service REST API reference

**More Specifications:**
- for the Bedrock API - ?
- for the Gemini API - ?
- for the Llama API - ?

## Problem Spaces

**Foundation Models:** Emphasize the creation and application of large-scale models that can be adapted to a wide range of tasks with minimal task-specific tuning.  

- [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html)
  - [Open Source LLM Tools](https://huyenchip.com/llama-police)

- [Llama Stack RFC](https://github.com/meta-llama/llama-stack/blob/main/rfcs/RFC-0001-llama-stack.md)
  - https://github.com/meta-llama/llama-stack-apps
  - https://github.com/meta-llama/llama-stack

**Predictive Human Preference (PHP):** Leveraging human feedback in the loop of model training to refine outputs or predictions based on what is preferred or desired by humans.   

- Predictive Human Preference - [php](php)

**Fine Tuning:** The process of training an existing pre-trained model on a specific task or dataset to improve its performance on that task.  

- https://llama.meta.com/docs/how-to-guides/fine-tuning
- https://github.com/hiyouga/LLaMA-Factory

**Cross-cutting Themes:**

"Our results show conditioning away risk of attack remains an unsolved problem; for example, all tested models showed between 25% and 50% successful prompt injection tests."

https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/

**Personal Identifiable Information (PII) and Security:** These considerations are crucial for ensuring that ML models respect privacy and are secure against potential threats.  

- Personal Identifiable Information - [pii](pii)  

**Code, SQL, Genomics, and More:** These areas highlight the interdisciplinary nature of ML, where knowledge in programming, databases, biology, and other fields converge to advance ML applications.  

**Neural Architecture Search (NAS):** Highlights the automation of the design of neural network architectures to optimize performance for specific tasks.  

- Biology (Collab w/ [Ashish Phal](https://www.linkedin.com/in/ashish-phal-548b37125/)) - [genomics](docs/genomics.md)  

**Few-Shot and Zero-Shot Learning:** Points to learning paradigms that aim to reduce the dependency on large labeled datasets for training models.  

**Federated Learning:** Focuses on privacy-preserving techniques that enable model training across multiple decentralized devices or servers holding local data samples.  
 
**Transformers in Vision and Beyond:** Discusses the application of transformer models, originally designed for NLP tasks, in other domains like vision and audio processing.  

**Reinforcement Learning Enhancements:** Looks at advancements in RL techniques that improve efficiency and applicability in various decision-making contexts.   

**MLOps and AutoML:** Concentrates on the operationalization of ML models and the automation of the ML pipeline to streamline development and deployment processes.  

**Hybrid Models:** Explores the integration of different model types or AI approaches to leverage their respective strengths in solving complex problems.  

**AI Ethics and Bias Mitigation:** Underlines the importance of developing fair and ethical AI systems by addressing and mitigating biases in ML models.  

**Energy-Efficient ML:** Reflects the growing concern and need for environmentally sustainable AI by developing models that require less computational power and energy. 

**Hardware:** Points to the importance of developing and utilizing hardware optimized for ML tasks to improve efficiency and performance.  










