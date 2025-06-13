import pathlib
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Decoder_Transformer import DecoderTransformer

from utils.model_utils import load_model

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.1:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    project_name = "Generate Text"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--temperature",
        help="Temperature parameter for softmax sampling.",
        type=restricted_float,
        default=1.0)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint.",
        required=False,
        default=None,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    temperature = args["temperature"]
    model_checkpoint = args["model_checkpoint"]

    loaded_model_status, loaded_model_dict = load_model(model_checkpoint)
    if not loaded_model_status:
        raise Exception("An error occured while loading model checkpoint!")

    # Model Params (From model checkpoints).
    num_heads = loaded_model_dict["num_heads"]
    num_models = loaded_model_dict["num_models"]
    hidden_dim = loaded_model_dict["hidden_dim"]    
    embedding_dim = loaded_model_dict["embedding_dim"]
    context_window = loaded_model_dict["context_window"]
    activation_type = loaded_model_dict["activation_type"]
    num_decoder_blocks = loaded_model_dict["num_decoder_blocks"]

    vocab = loaded_model_dict["vocab"]
    vocab_size = len(vocab)

    # TODO: Save this in vocab dictionary.
    start_token = vocab_size
    padding_token = vocab_size + 1

    num_seq_chunks = context_window // num_models

    saved_models_list = loaded_model_dict["saved_models"]

    models_list = []
    for model_index in range(num_models):
        temp_model = DecoderTransformer(
            model_index=model_index,
            num_seq_chunks=num_seq_chunks,
            num_embeddings=vocab_size + 2,  # Includes [START] and [PAD] tokens.
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            out_classes=vocab_size + 1,  # Includes only [PAD] tokens.
            num_blocks=num_decoder_blocks,
            activation_type=activation_type)

        curr_model_state_dict = saved_models_list[model_index]["model"]
        temp_model.custom_load_state_dict(curr_model_state_dict)

        temp_model = temp_model.to(device)

        temp_models_dict = {}
        temp_models_dict["model"] = temp_model

        models_list.append(temp_models_dict)

    prev_kv = None
    curr_model_index = 0
    all_generated_token_list = [int(start_token)]

    for model_dict in models_list:
        curr_model = model_dict["model"]
        curr_model.eval()

        temp_generated_tokens = [all_generated_token_list[-1]]
        for _ in range(num_seq_chunks):
            temp_generated_tokens_tensor = torch.tensor(
                [temp_generated_tokens],
                device=device)

            with torch.no_grad():
                curr_kv, out_classifier = curr_model(
                    x=temp_generated_tokens_tensor,
                    prev_kv_list=prev_kv)  # (N,Seq,Class)

                # Multinomial sampling from classes.
                probs = F.softmax(out_classifier[0][-1] / temperature, dim=0)  # (Class,)

                # Pick most likely token for next generation for each Token Sequence (Seq).
                next_token = torch.multinomial(probs, 1)  # (1,)

            temp_generated_tokens.append(next_token.item())

        all_generated_token_list.extend(temp_generated_tokens[1:])

        prev_kv = curr_kv

    # Remove invalid tokens if any like padding token, not in vocab list.
    cleaned_pred_tokens = [clean_token for clean_token in all_generated_token_list if clean_token < vocab_size]
    pred_token_list = [vocab[c] for c in cleaned_pred_tokens]
    pred_txt = "".join(pred_token_list)

    print("*" * 100, "\n\n", pred_txt, "\n\n", "*" * 100)

if __name__ == "__main__":
    main()
