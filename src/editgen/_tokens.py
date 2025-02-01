from editgen._model import ModelProxy


def get_tokens(model: ModelProxy, prompts: list[str]):
    inputs = model.encode(prompts)

    tokens = []
    for index in range(len(prompts)):
        tokens.append([model.decode(item) for item in inputs["input_ids"][index]])

    return tokens


def get_replacement_indices(
    model: ModelProxy, prompts: list[str], word_a: str, word_b: str
) -> list[list[int]]:
    token_groups = get_tokens(model, prompts)

    if len(prompts[0].split()) != len(prompts[1].split()):
        raise NotImplementedError(f"Different prompt lengths ({prompts})")

    indices = []
    for word, tokens in zip([word_a, word_b], token_groups):
        prompt_indices, substring = [], []
        for i in range(len(tokens)):
            if word.startswith("".join([*substring, tokens[i]])):
                substring.append(tokens[i])
                prompt_indices.append(i)
        indices.append(prompt_indices)

    return indices


def get_reweight_word_indices(
    model: ModelProxy, prompts: list[str], word: str
) -> tuple[list[str], list[int]]:
    return get_replacement_indices(model, prompts, word, word)[0]


def get_refine_word_indices(
    model: ModelProxy, prompts: list[str]
) -> tuple[list[str], list[int]]:
    token_groups = get_tokens(model, prompts)

    indices = []
    for i, token_i in enumerate(token_groups[0]):
        for j, token_j in enumerate(token_groups[1]):
            if token_i.startswith("<") or token_j.startswith("<"):
                continue

            if token_i == token_j:
                indices.append((i, j))

    return list(zip(*indices))


def get_ignore_indices(
    model: ModelProxy,
    prompts: list[str],
) -> tuple[list[str], list[int]]:
    tokens_a, tokens_b = prompts[0].split(), prompts[1].split()
    if len(tokens_a) != len(tokens_b):
        raise NotImplementedError("Different prompt lengths")

    index = tokens_b.index("<IGNORE>")
    word = tokens_a[index]

    return [prompts[0], prompts[0]], get_replacement_indices(
        model,
        [prompts[0], prompts[0]],
        word,
        word,
    )[0]
