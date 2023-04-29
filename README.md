# Select, Extend, and Generate: Generative Knowledge Selection for Open-Domain Dialogue Response Generation
The codes for our DASFAA 2023 work [SEG-CKRG](https://link.springer.com/chapter/10.1007/978-3-031-30675-4_48)

This released code has been verified in the recent PyTorch 2.0 version (but not the torch.compile mode).

# Abstract
Incorporating external commonsense knowledge can enhance machines’ cognition and facilitate informative dialogues. However, current commonsense knowledge-grounded dialogue generation works can only select knowledge from a finite set of candidates retrieved by information retrieval (IR) tools. This paradigm suffers from: 1) The knowledge candidate space is limited because IR tools can only retrieve existing knowledge from the given knowledge base, and the model can only use the retrieved knowledge; 2) The knowledge selection procedure lacks enough interpretability to explain the selected result. Moreover, with the increasing popularity of pre-trained language models (PLMs), many knowledge selection methods of non-PLM models have become incapable because of the input/structure restrictions of PLMs. To this end, we propose a simple but elegant SEG-CKRG, and introduce a novel PLM-friendly Generative Knowledge Selection (GenSel) to select knowledge via a generative procedure. Besides selecting the knowledge facts from the retrieved candidate set, GenSel can also generate newly extended knowledge. GenSel also improves interpretability because the output of the knowledge selection is a natural language text. Finally, SEG-CKRG uses GPT-2 as the backbone language model. Extensive experiments and analyses on a Chinese dataset have verified the superior performance of SEG-CKRG.


## Dataset & GPT2 Checkpoint
The raw dataset can be found in this project: [dataset](https://github.com/pku-sixing/ACL2020-ConKADI)

Here, we provide the processed [dataset](https://drive.google.com/file/d/1S4H-PxNEUZZq0o4SYTKNaG0XwQhhXGm_/view?usp=sharing) and the GPT2 [Checkpoint](https://drive.google.com/file/d/1IGW0wx5AVb0BmA9T_MMI6PrxtN4YeCKS/view?usp=sharing)

## Training
Please use the following script to run our model. Use `--pretrained_model`  to locate our GPT2 [Checkpoint](https://drive.google.com/file/d/1IGW0wx5AVb0BmA9T_MMI6PrxtN4YeCKS/view?usp=sharing)

```shell
python train_FineUniGPT.py --train_path dataset/train.pkl  --dev_path dataset/dev.pkl --log_path log/Uni2_KG60GenSP_Dialog.log --save_model_path model/Uni2_KG60GenSP_Dialog  --max_len 512 --device 3 --batch_size 32 --gpu0_bsz 0.4  --log_step 100  --model_config config/FineGPT_SC.json --pretrained_model  gpt2/epoch200 --sp_embed_init_range 0.005  --epochs 15  --gpt2_hybrid

```

## Inference 
Please use the following script to run our model
```shell
python generate_FineUniGPT.py --device  0,2,3,1 --checkpoint model/Uni2_KG60GenSP_Dialog/model/best_model/ --model_config config/FineGPT_SC.json --test_path dataset/test.pkl  --save_path results.txt --thread 4  --topk 5 --beam_width 5 --generation_mode knowledge_dialogue --gpt2_hybrid  --length_penalty 1.5
```

## Citation

Please kindly cite this work if you are interested in this work.

```shell
@InProceedings{SEG-CKRG,
author="Wu, Sixing
and Xue, Ping
and Tao, Ye
and Li, Ying
and Wu, Zhonghai",
editor="Wang, Xin
and Sapino, Maria Luisa
and Han, Wook-Shin
and El Abbadi, Amr
and Dobbie, Gill
and Feng, Zhiyong
and Shao, Yingxiao
and Yin, Hongzhi",
title="Select, Extend, and Generate: Generative Knowledge Selection for Open-Domain Dialogue Response Generation",
booktitle="Database Systems for Advanced Applications",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="649--664",
abstract="Incorporating external commonsense knowledge can enhance machines' cognition and facilitate informative dialogues. However, current commonsense knowledge-grounded dialogue generation works can only select knowledge from a finite set of candidates retrieved by information retrieval (IR) tools. This paradigm suffers from: 1) The knowledge candidate space is limited because IR tools can only retrieve existing knowledge from the given knowledge base, and the model can only use the retrieved knowledge; 2) The knowledge selection procedure lacks enough interpretability to explain the selected result. Moreover, with the increasing popularity of pre-trained language models (PLMs), many knowledge selection methods of non-PLM models have become incapable because of the input/structure restrictions of PLMs. To this end, we propose a simple but elegant SEG-CKRG, and introduce a novel PLM-friendly Generative Knowledge Selection (GenSel) to select knowledge via a generative procedure. Besides selecting the knowledge facts from the retrieved candidate set, GenSel can also generate newly extended knowledge. GenSel also improves interpretability because the output of the knowledge selection is a natural language text. Finally, SEG-CKRG uses GPT-2 as the backbone language model. Extensive experiments and analyses on a Chinese dataset have verified the superior performance of SEG-CKRG.",
isbn="978-3-031-30675-4"
}
```