# Waterfall Transformer
## Description
[Transformer models](https://arxiv.org/abs/1706.03762) face scaling challenges with their attention mechanism because computing the attention matrix naively requires O(n²·d) space, where n is the sequence length and d is the embedding dimension. Numerous approaches have been explored to alleviate this issue such as: [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370), [FlashAttention](https://arxiv.org/abs/2205.14135), [Ring Attention](https://arxiv.org/abs/2310.01889) etc, all with varying degrees of successes. This project explores an alternative approach to avoid the high memory cost of explicitly computing the full attention matrix (QKᵀ) in memory by leveraging the fact that in **Decoder-only** Transformers used for next-token prediction tasks, tokens are restricted to attending only to earlier positions. Moreover, communication between tokens occurs exclusively in the Self-Attention layers, as all other layers, FeedForward layers, operate independently on each token in a point-wise manner. In my opinion there's no reason why all the sequences of tokens need to be processed by one model, and that by leveraging multiple sequential models, it becomes feasible to scale the sequence length without substantial increase in memory consumption.

For example, given T sequence of tokens, one could use 3 independent models for training. This would require the T tokens be split into 3 chunks. The first model will train on the first chunk of tokens but with the addition of storing all the (K,V) vectors from each Self-Attention layer. The second model will train on the second chunk of tokens but for each of it's Self-Attention layer, it will need to append the first model's (K, V) vector onto it's own (K,V) vectors before each attention computation. The new concatenated (K, V) are then stored for the next model. The third model will train on the third chunk of tokens and much like the second model, will append the previous (K,V) vectors to it's own. A unique causal mask will need to be computed for each model to factor for the split tokens inputs.

All models are trained separately where no gradient flows between them. The last model, worst case, should have a space complexity of O(m·n·d) where m is the sequence length of input chunk, n is the sequence length of the K vector and d is the embedding dimension, polynomial growth rate. The flow of data, specifically the intermediary (K, V) tokens can be thought of as a waterfall, previous models sends information to the succedding model in a sequential order.

## Simple visualization of algorithm
<p align="center">
  <img alt="Basic Attention" src="assets/Basic attention.jpg" />
</p>

<p align="center">
  <img alt="Waterfall Attention" src="assets/Waterfall attention.jpg" />
</p>

## Implementation
The project includes the following scripts and code:
- Script to generate Vocabulary dictionary.
- Script to generate subword dataset using Vocabulary.
- Script to generate train/test dataset.
- Code to generate text from saved models.
- Code to train model.

## Requirements
- Anaconda (Optional)
- Python 3

## Installing.
1. (Optional) Install [Anaconda](https://docs.anaconda.com/) on your machine.
    - Create anaconda environment:
    ```
    conda create --name <env_name> python=3.12
    ```
    - To activate anaconda environment, if not already activated:
    ```
    conda activate <env_name>
    ```
2. (If not installed) Install [Pytorch 2.5](https://pytorch.org/get-started/locally/) based on your hardware requirements and preferences.
3. Install Python depencies:
    ```
    pip install -r requirements.txt
    ```

## Dataset Creation (Needs to be run once)
Dataset used was [Plain Text Wikipedia 2020-11](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011). Once downloaded, the json file was converted to raw plain-text files and all the Non-Latin characters like Greek symbols was removed (simplified the dataset). Script used for both operations are not provided!

### Generating vocabulary.
To generate vocabulary `.json` file, run the following (Needs to be generated first before everything else):
```
python scripts/generate_vocabulary.py <args>
```

### Generating subword dataset.
To generate subword dataset you need a `.csv` file listing all the raw `.txt` files from the dataset, run the following to generate a `.json` dataset file.
```
python scripts/generate_subword_dataset.py <args>
```

### Generating train/test dataset.
To generate the train/test dataset to be used for training model, run the following:
```
python scripts/generate_training_dataset.py <args>
```

## Training model.
To train the model, one first needs a config.json file (Example file can be found in **training_results/config_json_example/config.json**), then run the following:
```
python train_waterfall_transformer_all.py <args>
```

## Generating text.
To generate text using saved checkpoint models, run the following:
```
python generate_text.py <args>
```

## Results
Models were trained on a single **RTX 3080** 10GB GPU. (Log file can be found in **training_results/** folder, no models are shared).

No testing loss charts was developed due to time constraints.

Example of text generated by the model(s), with a `temperature` value of 0.75:
```
 s and is the most popular server in overall office.Douglas Wright No. JC  Raven Tribute Show! Trailsite  , mobile detector; satellite stations  , 1998. The station opened on 9 December 2007 prior to the AT&T; broadcasting facilities and a programme by Interstate 82/50 (Business Radio 104.0) on 30 May 2007 at Air New York (U.S.) and at about 750 (FE 105.1). "Raven with the Iowa Display to Sixth Avenue (Hove 95.0) on Fox Radio (Broadcasting 40.7 FSR. 500 (Radio 93.05) (Max 100.2: ABC 106.02, Fox Radio 90.115), CBC Radio 101.1 (FM 92.1) (1991) on New 106.2 (2nd FM 91.07) on 5 July 2009 (Radio 101.08). ===Public radio stations (and ESPN)=== Although a new sign-of-the-air television station, a sky-themed station (Network 102.0) has been transmitted for television service on CBS
```

## Interpretation / Observation
The text generation is mostly gibberish due to the model's small parameter size but despite using different models to generate the text, it's still mostly coherent. It needs more thorough testing to ensure the models are acting as one and are not simply ignoring the preceding models contributions.

It was noted that by reducing the length of sequences each model worked on due to chunking, it was necessary to increase the batch size accordingly to ensure the model processed a sufficient number of tokens per batch to get better performance. Batch size, for a total sequence length of 256 subword tokens, went up from 32 to 80, which was possible due to the reduced memory consumption.

Technically one could skip the waterfall structure and simply use one model trained in a recurrent manner, where accumulated (K, V) are reused for the next input chunks. Tradeoff with this arrangement is that training time may increase but the complexity from dealing with multiple models reduces.

## Additional Relevant Links.
- [Transformers Explained Visually (Part 2): How it works, step-by-step](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34/)
- [LLM Temperature](https://www.hopsworks.ai/dictionary/llm-temperature)

