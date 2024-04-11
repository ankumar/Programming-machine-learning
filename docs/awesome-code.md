https://github.com/karpathy

1. **Get Code** - git clone https://github.com/karpathy/nanoGPT.git
2. **Get Data** - python data/shakespeare_char/prepare.py
3. **Get PyTorch Nightly** - pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu   
4. **Training/ I have a Apple M1** - python train.py config/train_shakespeare_char.py --device=mps --compile=False (Ctrl+C to stop!)
5. **Generating Text** - python sample.py --out_dir=out-shakespeare-char --device=mps 
6. **Get Data** - python data/openwebtext/prepare.py
7. **Training/ I have a Apple M1** - python3 train.py --batch_size=32 --device=mps --compile=False 

The Alignment Handbook - https://github.com/huggingface/alignment-handbook

Hackable implementation of state-of-the-art open-source large language models - https://github.com/mozilla-ai/lit-gpt  

Better data, better AI - https://github.com/lilacai/lilac  

Structured data extraction in PHP - https://github.com/adrienbrault/instructor-php  

https://github.com/vbwyrde/DSPY_VBWyrde  

https://github.com/boxabirds/pytorch2mlx

--

- [DeepMind](https://github.com/google-deepmind/long-form-factuality/tree/main/longfact)
- [AI2](https://github.com/allenai/open-instruct?tab=readme-ov-file)
- [LMSYS](https://github.com/lm-sys)
- [ml-explore](https://github.com/ml-explore)
- [NousResearch](https://github.com/orgs/NousResearch/repositories)

--

https://github.com/bigcode-project  
- https://huggingface.co/blog/starcoder


S/W Agent capable of writing Go plugins (specs TBD, this looks interesting https://www.benthos.dev/blog/2021/10/12/new-plugins-stable)

Exploring these repos currently:
- https://github.com/princeton-nlp/SWE-agent
- https://github.com/stitionai/devika  
- https://github.com/OpenDevin/OpenDevin

--

- https://medium.com/snowflake/mistral-snowflake-the-new-frontier-in-sql-copilot-products-f71b8a939899
