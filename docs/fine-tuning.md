**FastChat:** https://github.com/lm-sys/FastChat/blob/main/docs/training.md  

- **Apple M1**  

```
pip install -U mlx
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
brew install rust 
brew install cmake
pip3 install --upgrade pip
pip3 install -e ".[model_worker,webui]"
pip install "mlx-lm>=0.0.6"
python3 -m fastchat.serve.mlx_worker --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

```
pip3 install -e ".[train]"
```

**Axolotl:** https://github.com/OpenAccess-AI-Collective/axolotl  
- Using Modal: https://github.com/modal-labs/llm-finetuning

**LM Buddy:** https://github.com/mozilla-ai/lm-buddy  

**Levanter:** https://levanter.readthedocs.io/en/latest/Fine-Tuning/

"While Levanter's main focus is pretraining, we can also use it for fine-tuning. As an example, we'll show how to reproduce Stanford Alpaca, using Levanter and either Llama 1 or Llama 2 7B. The script we develop will be designed for Alpaca, defaulting to using its dataset and prompts, but it should work for any single-turn instruction-following task."

- **Apple M1**  

```
conda create -n levanter-metal python=3.10 pip
conda activate levanter-metal

# Installing a compatible version of jax and jaxlib
pip install jax-metal==0.0.5

# Installing Levanter from source to get the latest updates
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .

# Run a test
python -m levanter.main.train_lm --config config/gpt2_nano.yaml
```
