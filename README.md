1. **Get Code** - git clone https://github.com/karpathy/nanoGPT.git
2. **Get Data** - python data/shakespeare_char/prepare.py
3. **Get PyTorch Nightly** - pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu   
4. **Training/ I have a Apple M1** - python train.py config/train_shakespeare_char.py --device=mps --compile=False (Ctrl+C to stop!)
5. **Generating Text** - python sample.py --out_dir=out-shakespeare-char --device=mps 
6. **Get Data** - python data/openwebtext/prepare.py
7. **Training/ I have a Apple M1** - python3 train.py --batch_size=32 --device=mps --compile=False 
   
## Problem spaces:

Personal Identifiable Information - [pii](pii)  
Genomics - [genomics](genomics)  
Predictive Human Preference - [php](php)  
Code - [coding](coding)  
SQL - [sql](sql)  
Conversations - [toxicity](toxicity)  
Fine Tuning - [fine-tuning](fine-tuning)  
Audio - [audio](audio)  
Small Language Models - [llm](llm)  



