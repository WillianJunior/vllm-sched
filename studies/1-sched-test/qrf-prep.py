import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm
import os
from time import time
import argparse
import joblib


def main():
    args = parse_args()
    
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f]

    # Load LLM model for tokenization and embedding
    llm_model_path = args.llm_model
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    model = AutoModelForCausalLM.from_pretrained(llm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Evaluation mode
    model.eval()
    # Try to run in GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    random_forest_model, X_train, y_train = joblib.load(args.random_forest_model)


    print("REQ_TOKEN PROMPT ESTIMATED")

    init = 0
    prompts = prompts[init:]

    for i,l in enumerate(prompts):
        # Tokenize
        in_tokens = tokenizer(
            l,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        in_tokens_to_decode = {k: v.to(device) for k, v in in_tokens.items()}
        in_tokens = list(in_tokens_to_decode['input_ids'].cpu().numpy()[0])
        if len(in_tokens) < 4:
            print(l)
            print(in_tokens['input_ids'].cpu().numpy()[0])
            0/0
        # Token 60 is ']' and token[3] is '['
        req_ID = "".join(map(str, in_tokens[3:in_tokens.index(60)]))
        #req_token = in_tokens['input_ids'].cpu().numpy()[0,3]
        n_prompt_tokens = len(in_tokens)
        #n_prompt_tokens = (in_tokens["attention_mask"] == 1).sum().item()

        #if n_prompt_tokens > 15000:
            #print(f"{i};{len(l)};{n_prompt_tokens};-1;-1")
            #continue


        with torch.no_grad():
            decode_out = model(**in_tokens_to_decode, output_hidden_states=True)
            prompt_embeddings = decode_out.hidden_states[-1].mean(dim=1).cpu().numpy()
            
            # too expensive to run 1 by 1...
            n_generated = -1
            #n_generated = model.generate(**in_tokens, pad_token_id=tokenizer.eos_token_id, max_new_tokens=4000).shape[1] - n_prompt_tokens

            n_estimated = QRF(random_forest_model, prompt_embeddings, X_train, y_train)[0]

            del in_tokens, decode_out
            torch.cuda.empty_cache()

            #print(f"{i};{len(l)};{n_prompt_tokens};{n_generated};{int(n_estimated)}")
            print(f"{req_ID} {n_prompt_tokens} {int(n_estimated)}")





def load_dataset(prompts_path, targets_path, min_len=-1, max_len=-1, need_targets=True):
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_raw = [line.strip() for line in f]
    
    if not need_targets:
        return prompts_raw, None

    with open(targets_path, "r", encoding="utf-8") as f:
        y_raw = [line.strip() for line in f]

    prompts_raw = prompts_raw[: len(y_raw)]

    prompts = []
    y = []
    for text, val in zip(prompts_raw, y_raw):
        # Filter out empty targets
        if val != "":
            # Filter out min/max len targets
            if max_len > 0 and int(val) >= max_len:
                continue
            if min_len > 0 and int(val) <= min_len:
                continue

            prompts.append(text)
            y.append(int(val))

    y = np.array(y)

    print(f"DSET_SIZE: {len(prompts)}")

    print("Percentiles of decode lengths:")
    print_quantiles(y)

    return prompts, y


def get_embeddings(
    prompts, llm_model_path, embeddings_path, force_embedding=False, pooling="mean"
):
    # Check if embeddings already exist
    if os.path.exists(embeddings_path) and not force_embedding:
        embeddings = np.load(embeddings_path)
        print(f"Embeddings loaded from {embeddings_path}")
        return embeddings, False

    # Load LLM model for tokenization and embedding
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    model = AutoModel.from_pretrained(llm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Evaluation mode
    model.eval()
    # Try to run in GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # All tokens are loaded into memory in parallel (prefill). This can
    # overwhelm the memory. More memory = bigger batch size.
    batch_size = 1

    all_embeddings = []
    all_prompt_tokens = []

    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc="Calculating embeddings",
    ):
        batch_texts = prompts[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        for j, text in enumerate(batch_texts):
            token_count = (inputs["attention_mask"][j] == 1).sum().item()
            all_prompt_tokens.append(token_count)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            if pooling == "mean":
                batch_embeddings = last_hidden_state.mean(dim=1)
            elif pooling == "cls":
                batch_embeddings = last_hidden_state[:, 0, :]
            else:
                raise ValueError("Unsupported pooling type: 'mean' or 'cls'")

            all_embeddings.append(batch_embeddings.cpu().numpy())

            del inputs, outputs
            torch.cuda.empty_cache()

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    print("Percentiles of prompt lengths:")
    print_quantiles(all_prompt_tokens)

    return embeddings, True


def get_random_forest_model(
    X_train,
    y_train,
    X_val,
    y_val,
    random_forest_model_path,
    force_training=False,
):
    # Check if random forest model already exist
    if os.path.exists(random_forest_model_path) and not force_training:
        print(f"Random forest model loaded form {random_forest_model_path}")
        if (random_forest_model_path.split(".")[-1] == "qrf"):
            random_forest_model, X_train, y_train = joblib.load(random_forest_model_path)
            return random_forest_model, X_train, y_train
        else:
            random_forest_model = joblib.load(random_forest_model_path)
            return random_forest_model

    t0 = time()
    n_estimators_total = 200
    steps = 10
    step = n_estimators_total // steps

    # val_mae_progress = []

    random_forest_model = RandomForestRegressor(
        n_estimators=step,
        min_samples_leaf=3,
        max_depth=None,
        random_state=42,
        n_jobs=-1,  # CPU parallel run
        warm_start=True,
        max_features="log2",  # should reduce over-fitting?
    )

    pbar = tqdm(total=n_estimators_total, desc="Training Random Forest")

    while random_forest_model.n_estimators <= n_estimators_total:
        random_forest_model.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = random_forest_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        # val_mae_progress.append(val_mae)

        # Update progress bar with MAE
        pbar.set_postfix({"val_MAE": f"{val_mae:.4f}"})
        pbar.update(step)

        # Increment number of trees for the next iteration
        random_forest_model.n_estimators += step

    pbar.close()

    t1 = time()
    print(f"Trained in {t1 - t0:.2f} secs")
    
    qrf_full_model = (random_forest_model, X_train, y_train)
    joblib.dump(qrf_full_model, random_forest_model_path + ".qrf")
    joblib.dump(random_forest_model, random_forest_model_path)
    print(f"Saved random forest model to {random_forest_model_path}")

    return random_forest_model


def evaluate_forest_model(random_forest_model, X_train, y_train, X_test, y_test):
    # Validate QRF
    t0 = time()
    y_pred_qrf = QRF(
        random_forest_model,
        X_test,
        X_train,
        y_train,
    )
    t1 = time()
    print(f"=== QRF ({t1 - t0:.2f} secs for {len(X_test)} points) ======{'=' * 22}")

    mae = mean_absolute_error(y_test, y_pred_qrf)
    ratio_error = np.abs(np.array(y_pred_qrf) - np.array(y_test)) / np.array(y_test)
    mare = np.mean(ratio_error)
    ratio = np.array(y_pred_qrf) / np.array(y_test)
    mean_ratio = np.mean(ratio)
    print(f"MAE:        {mae:.2f}")
    print(f"MARE:       {mare:.2f}")
    print(f"Mean ratio: {mean_ratio:.2f}")
    print("---------------------------------------------------------------")
    print("Absolute ratio error percentiles:")
    print_quantiles(ratio_error)
    print("Ratio percentiles:")
    print_quantiles(ratio)

    print(f"=== Just RF ({t1 - t0:.2f} secs for {len(X_test)} points) ={'=' * 23}")
    y_pred_rf = random_forest_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_rf)
    ratio_error = np.abs(np.array(y_pred_rf) - np.array(y_test)) / np.array(y_test)
    mare = np.mean(ratio_error)
    ratio = np.array(y_pred_rf) / np.array(y_test)
    mean_ratio = np.mean(ratio)
    print(f"MAE:        {mae:.2f}")
    print(f"MARE:       {mare:.2f}")
    print(f"Mean ratio: {mean_ratio:.2f}")
    print("---------------------------------------------------------------")
    print("Absolute ratio error percentiles:")
    print_quantiles(ratio_error)
    print("Ratio percentiles:")
    print_quantiles(ratio)


def print_quantiles(data, quantiles=None):
    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Assuming data is a 1D numpy array
    data_quantiles = np.quantile(data, quantiles)
    for q, val in zip(quantiles, data_quantiles):
        print(f"\t{int(q * 100)}th percentile: {val:.2f}")


def QRF(random_forest_model, X, X_train, y_train, quantiles=[0.9]):
    """
    Predict quantiles for X using a trained RandomForestRegressor.
    Efficient implementation using vectorized leaf mapping.
    From ChatGPT...
    """
    n_samples = X.shape[0]
    all_values = [[] for _ in range(n_samples)]  # collect leaf values for each sample

    for tree in random_forest_model.estimators_:
        # leaf indices for train and test
        train_leaves = tree.apply(X_train)
        test_leaves = tree.apply(X)

        # map each test sample to the training samples in the same leaf
        leaf_to_values = {}
        for leaf, y_val in zip(train_leaves, y_train):
            leaf_to_values.setdefault(leaf, []).append(y_val)

        # collect leaf values for each test sample
        for i, leaf in enumerate(test_leaves):
            all_values[i].extend(leaf_to_values[leaf])

    # compute requested quantiles
    quantile_preds = {
        q: np.array([np.quantile(v, q) for v in all_values]) for q in quantiles
    }

    # Just return the quantile predictions if only one is asked.
    if len(quantiles) == 1:
        return quantile_preds[quantiles[0]]
    else:
        return quantile_preds

def get_predictions(random_forest_model, X):
    model, X_Train, y_train = random_forest_model

    def batches(lst, size=10):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]

    for batch in batches(X, 1000):
        print(QRF(model, batch,  X_Train, y_train))



def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a RandomForestRegressor "
        "model for prompt length estimation."
    )

    # # Positional argument (required)
    # parser.add_argument("input_file", help="Path to the input file.")

    parser.add_argument(
        "--prompts",
        default="prompts.txt",
        help="Prompts file. Format: one prompt per line (default: prompts.txt).",
    )

    parser.add_argument(
        "--targets",
        default="targets/target.txt",
        help="Target estimation file. Format: one expected int value per line "
        "This file can have DSET_SIZE lines, which is fewer than the prompts "
        "file. If so, only DSET_SIZE lines are read from the prompts file "
        "(default: targets/target.txt).",
    )

    parser.add_argument(
        "--llm-model",
        default="/snfs1/llm-models/llama-3.2-3B-Instruct",
        help="Path to the model dir (default: "
        "/snfs1/llm-models/llama-3.2-3B-Instruct).",
    )

    parser.add_argument(
        "--embeddings",
        help="Path to the embeddings required for model training. If the file "
        "does not exist, embeddings are calculated from the prompts file. If "
        "it exists, it is loaded and trained. If the embeddings size (number "
        "of embeddings is different from the DSET_SIZE, the embeddings are "
        "calculated and saved to file. "
        "(default: embeddings/embeddings-<DSET_SIZE>.npy).",
    )

    parser.add_argument(
        "--random-forest-model",
        help="Path to the random forest model. After training, the model is "
        "saved in this path. If the file exists, the model is loaded. "
        "(default: models/random-forest-model-<DSET_SIZE>.pkl).",
    )

    parser.add_argument(
        "--force-training",
        action="store_true",
        help="Force the training of the model. Will overwrite the "
        "random forest model file.",
    )

    parser.add_argument(
        "--force-embedding",
        action="store_true",
        help="Force recalculation of embeddings. Will overwrite the embeddings file.",
    )

    parser.add_argument(
        "--filter-max-len",
        type=int,
        default=-1,
        help="Filter out all prompts which generated over MAX_LEN tokens. "
        "Doesn't filter by default.",
    )

    parser.add_argument(
        "--filter-min-len",
        type=int,
        default=-1,
        help="Filter out all prompts which generated less then MIN_LEN "
        "tokens. Doesn't filter by default.",
    )

    parser.add_argument(
        "--get-predictions-only",
        action="store_true",
        help="Just print the predictions for each prompt line. Disables --filter-*-len.",
    )


    # # Optional parameter (integer)
    # parser.add_argument(
    #     "-n",
    #     "--num-iterations",
    #     type=int,
    #     default=1,
    #     help="Number of iterations to run (default: 1).",
    # )

    return parser.parse_args()


if __name__ == "__main__":
    main()
