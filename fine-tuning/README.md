- https://github.com/OpenAccess-AI-Collective/axolotl

- https://levanter.readthedocs.io/en/latest/Fine-Tuning/

"While Levanter's main focus is pretraining, we can also use it for fine-tuning. As an example, we'll show how to reproduce Stanford Alpaca, using Levanter and either Llama 1 or Llama 2 7B. The script we develop will be designed for Alpaca, defaulting to using its dataset and prompts, but it should work for any single-turn instruction-following task."

**Apple M1**  

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
