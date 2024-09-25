1. [**The Shift from Models to Compound AI Systems**](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)  
This article from Berkeley AI Research (BAIR) highlights a growing trend where AI advancements increasingly rely on **compound AI systems**—combinations of multiple models and components—rather than traditional monolithic models. Compound systems offer more flexibility and adaptability, as each component can specialize in a different task or phase of the pipeline.

<p align="center">**state-of-the-art AI results are increasingly obtained by compound systems with multiple components, not just monolithic models**.</p>

Key points:
	•	**Design, optimization, and operation:** Approaches are still emerging, but compound AI systems are proving more efficient for complex tasks.
	•	**Maximizing reliability and quality:** These systems promise higher reliability, particularly for large-scale applications, by breaking down tasks into smaller, more manageable units. 
	•	**Trend for 2024:** BAIR sees this as one of the most important trends, where developers will focus on how to assemble these components in effective ways.  
"_Figuring out the best practices for developing compound AI systems is still an open question, but there are already exciting approaches to aid with design, end-to-end optimization, and operation. We believe that compound AI systems will remain the best way to maximize the quality and reliability of AI applications going forward, and may be one of the most important trends in AI in 2024._"

![image](https://github.com/user-attachments/assets/d874e82f-9b89-487e-894c-1ed3b247294d)
<p align="center">Increasingly many new AI results are from compound systems.</p>

2. [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html)
  - [Open Source LLM Tools](https://huyenchip.com/llama-police)

## OpenAPI
started with this one endpoint Completions **/completions** with Text Input/Output (now Legacy)

Then ChatGPT, Chat completions **/chat/completions**  with Messages Input/Output
  
3. Embeddings			**/embeddings**
4. Image generation		**/images/generations**
5. Text to speech			**/audio/speech**
6. Speech to text			**/audio/transcriptions**
7. Moderation			**/moderations**
8. Fine-tuning  **/fine_tuning/jobs**
9. Batch				**/files /batches**

- Function Calling - Query Database, Send Emails etc.
- Response Formats - [Building AGI with OpenAI's Structured Outputs API](https://www.youtube.com/watch?v=NjOfH9D8aJo)

REST API:  
- OpenAPI specification for the [OpenAI API](https://github.com/openai/openai-openapi/blob/master/openapi.yaml)  
- Specification for the [Azure OpenAI API](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs)

Python SDK:  
- OpenAI Python [API](https://github.com/openai/openai-python/blob/main/api.md) library in the The official Python library for the [OpenAI API](https://github.com/openai/openai-python)

- [Drop in replacement](https://github.com/datastax/astra-assistants-api) for the OpenAI Assistants API
  - Full coverage of OpenAI endpoints in the repo [here](https://github.com/datastax/astra-assistants-api/blob/main/coverage.md)
 
- [Llama Stack RFC](https://github.com/meta-llama/llama-stack/blob/main/rfcs/RFC-0001-llama-stack.md)
  - https://github.com/meta-llama/llama-stack-apps
  - https://github.com/meta-llama/llama-stack

**More Specifications:**
- for the Bedrock API - ?
- for the Gemini API - ?
- for the Llama API - ?

**More Languages:**
- The official [Go](https://github.com/openai/openai-go) library for the OpenAI API
    
## Authentication & Authorization

### 1. API Key
An API key is a simple string (often alphanumeric) used to authenticate requests. It can be included as:
- **URL Parameter**:  
  `https://example.com/api/resource?api_key=YOUR_API_KEY`
- **Header**:  
  `Authorization: ApiKey YOUR_API_KEY`

API keys are typically used for simple authentication and are suited for server-to-server communication but are less secure if exposed in URLs.

---

### 2. Bearer Token
A **Bearer Token** is a security token that is issued as part of OAuth 2.0. This token grants the bearer access to resources. It's usually passed in the request header:
- **Header**:  
  `Authorization: Bearer YOUR_TOKEN`

Bearer tokens offer more security compared to API keys, especially when combined with token expiration and refresh mechanisms.

---

### 3. Microsoft Entra ID (formerly Azure AD)
Entra ID provides **OAuth 2.0** or **OpenID Connect (OIDC)** based authentication and authorization, mostly used for securing enterprise apps. The flow typically involves:
- **Access Token**: Obtained after a user or service authenticates with Entra ID.
- **Header**:  
  `Authorization: Bearer YOUR_ACCESS_TOKEN`

Entra ID is often used in conjunction with Microsoft services or enterprise environments for user-based or service-based authentication.

---

### 4. AWS Signature
**AWS Signature Version 4** is used to securely sign API requests to AWS services. This method calculates a signature based on the request parameters, headers, and the user's secret access key. The signature is added to the request as:
- **Authorization Header**:  
  `Authorization: AWS4-HMAC-SHA256 Credential=ACCESS_KEY/..., SignedHeaders=..., Signature=SIGNATURE`

It is typically more secure because the signature is derived dynamically and is time-limited.


## Guardrails
- Sensitive information filters - PII types, Regex patterns etc.
- Content filters - Configure content filters to detect & block harmful user inputs and model responses
- Denied topics
- Word filters
- Contextual grounding check

## Assistants
- Tune Personality & Capabilities
- Call Models
- Access Tools in parallel
    - Built-in code_interpreter, file_search etc.
    - Function Calling
- Persistent Threads
- File Formats

## Agents
- “Agent is a more overloaded term at this point than node, service, and instance.”
  
-> https://x.com/rakyll/status/1837164761362133057

- “I'm wondering what would be the base requirements of "true agent" (i.e. not just over-hyped marketing). 
For me: 
Can use APIs reliably. 
APIs by other companies, not just ones specifically written for the agent.
The API usage should cover a large subset of the services that the agent is aiming to cover. 

I.e. if your agent is supposed to order food, it should be able to find an open restaurant with take away, figure out how to do the delivery and at least support the 3 large delivery companies.”

-> https://x.com/gwenshap/status/1837167653338681819

## Telemetry
- [OpenTelemetry](https://opentelemetry.io/)
  - [Semantic Conventions for Generative AI systems | OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
    - [Introduce semantic conventions for modern AI (LLMs, vector databases, etc.)](https://github.com/open-telemetry/semantic-conventions/issues/327)

# Problem Spaces

![image](https://github.com/user-attachments/assets/6830c307-62ad-4255-ab3a-5d7037176e2b)
<p align="center">Source: https://github.com/rasbt/LLMs-from-scratch</p>


**Foundation Models:** Emphasize the creation and application of large-scale models that can be adapted to a wide range of tasks with minimal task-specific tuning.  

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










