# Waterfall Transformer
## Description
[Transformer-based models](https://arxiv.org/abs/1706.03762) are known for facing scaling challenges due to their attention mechanism. This is because computing the attention matrix: $\mathrm{QK^T}$, naively necessitates a space complexity of $O(n^2 \cdot d)$, where $\mathrm{n}$ is the token sequence length and $\mathrm{d}$ is the embedding dimension. Numerous approaches have been explored to alleviate this issue such as: [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370), [FlashAttention](https://arxiv.org/abs/2205.14135), [Ring Attention](https://arxiv.org/abs/2310.01889), etc all with varying degrees of successes.

## How it works
This project explores a unique approach to avoid the high memory cost of naively computing the full attention matrix $(\mathrm{QK^T})$ in memory. This is done through leveraging the fact that in causal **Decoder-only** Transformers, those usually used for next-token prediction tasks, tokens are restricted to focussing only on the preceeding tokens. Moreover, communication between tokens occurs exclusively in the **Masked Self-Attention** layers, as all other layers such as **FeedForward** layers, independently process each token in a point-wise manner.

There's no reason why all the sequences of tokens need to be processed by a **singular Transformer-based model** given the leftward access of information during the attention process means the sequential input can technically be viewed in terms of input chunks i.e each token only focuses on those preceeding them and therefore all token before that token including it can be viewed as a chunk. By leveraging **multiple Transformer-based models** with clever engineering, it is possible to scale the sequence length, $\mathrm{n}$, supported without a substantial increase in memory consumption using separate chunks as input. For example, given $\mathrm{T}$ sequence of tokens, one could use 3 independent models (model_<sub>0</sub>, model_<sub>1</sub>, and model_<sub>2</sub>) for training. This would require the $\mathrm{T}$ tokens being split into 3 equal chunks ($\mathrm{T_0}$, $\mathrm{T_1}$, and $\mathrm{T_2}$):
1. model_<sub>0</sub> will train on $\mathrm{T_0}$. It will need to store **ALL** the $(\mathrm{K,V})$ vectors from each **Masked Self-Attention** layer.
2. model_<sub>1</sub> will train on $\mathrm{T_1}$ but for each of its **Masked Self-Attention** layer, it will need to append model_<sub>0</sub>'s $(\mathrm{K,V})$ vector pairs onto the respective $(\mathrm{K,V})$ vectors before each attention computation. The new concatenated $(\mathrm{K,V})$ from model_<sub>1</sub> are then stored for the next model.
3. model_<sub>2</sub> will train on $\mathrm{T_2}$ and much like model_<sub>1</sub>, will append the previous $(\mathrm{K,V})$ vectors to its own.

A unique **causal mask** operation will need to be computed for model_<sub>1</sub> and model_<sub>2</sub> to factor for the shifted token positions of the input token chunks.

All models are trained independently of each other in that there is no gradient flow between the models despite information exchange in the form of $(\mathrm{K,V})$ vectors. The final model, model_<sub>2</sub>, should have a space complexity (**worst case**) of $O(m \cdot n \cdot d)$ where $\mathrm{m}$ is the sequence length of input chunks $\mathrm{T_2}$ from previous models, $\mathrm{n}$ is the sequence length of each $(\mathrm{K,V})$ vectors, and $\mathrm{d}$ is the embedding dimension.

The flow of information, specifically the intermediary $(\mathrm{K, V})$ vectors, between the model(s) can be thought of as a **waterfall**; it can only flow in one direction, from the *top* to *bottom*.

## Visualization of the algorithm
### Visualization of a singular Transformer-based model
<p align="center">
  <img alt="Basic Attention" src="assets/Basic attention.jpg" />
</p>

### Visualization of a waterfall Transformer-based models
<p align="center">
  <img alt="Waterfall Attention" src="assets/Waterfall attention.jpg" />
</p>

## Implementation
The project includes the following code:
- Code to generate Vocabulary dictionary.
- Code to generate subword dataset using Vocabulary.
- Code to generate train/test dataset.
- Code to generate text from saved pre-trained models.
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

## Pre-trained Models
Model weights will be shared soon...

## Results
Models were trained on a single **RTX 3080** 10GB GPU. (Log file can be found [here](training_results/Logs)).

No testing loss charts was developed due to time constraints.

Example of text generated by the model(s), with a `temperature` value of 0.75:

"
 s and is the most popular server in overall office.Douglas Wright No. JC  Raven Tribute Show! Trailsite  , mobile detector; satellite stations  , 1998. The station opened on 9 December 2007 prior to the AT&T; broadcasting facilities and a programme by Interstate 82/50 (Business Radio 104.0) on 30 May 2007 at Air New York (U.S.) and at about 750 (FE 105.1). "Raven with the Iowa Display to Sixth Avenue (Hove 95.0) on Fox Radio (Broadcasting 40.7 FSR. 500 (Radio 93.05) (Max 100.2: ABC 106.02, Fox Radio 90.115), CBC Radio 101.1 (FM 92.1) (1991) on New 106.2 (2nd FM 91.07) on 5 July 2009 (Radio 101.08). ===Public radio stations (and ESPN)=== Although a new sign-of-the-air television station, a sky-themed station (Network 102.0) has been transmitted for television service on CBS
"

## Observations
* The text generation is mostly gibberish due to the model's small parameter count. Howerver, despite the fact that different models were used, they still generated mostly coherent and plausible text.
* It was noted that by reducing the length of sequences each model worked on due to chunking, it was necessary to increase the batch size accordingly to ensure the model processed a sufficient number of tokens per batch to get better performance. During training the batch size went up from 32 to 80, for a total sequence length ($\mathrm{T}$) of 256 subword tokens, which was possible due to the reduced memory consumption from the attention matrix computation.

## Conclusion
* With bigger models and datasets, and more GPUs, the approach used could be properly tested and evaluated for consistency and accuracy as well as verifying if parallelization i.e using multiple GPUs for each model, is feasible.
* Hypothetically one could skip the waterfall structure and simply use a **singular Transformer-based model** trained in a recurrent manner. This entails recursively passing the previous $(\mathrm{K,V})$ vectors back into the model corresponding to the input chunks being processed. The obvious tradeoff with this setup is that training time / complexity may increase and so would potential instability if not done properly. However the complexity from dealing with multiple models reduces. More research could be done here.

## Additional Relevant Links.
- [Transformers Explained Visually (Part 2): How it works, step-by-step](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34/)
- [LLM Temperature](https://www.hopsworks.ai/dictionary/llm-temperature)

