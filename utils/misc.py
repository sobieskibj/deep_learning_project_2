import copy
import itertools
from pathlib import Path


def make_configs(base_config, combinations):
    product_input = [p["values"] for p in combinations.values()]
    product = [p for p in itertools.product(*product_input)]
    configs = []
    print(f"Running {len(product)} configurations:")
    # for each combination
    for n, p in enumerate(product):
        config = copy.deepcopy(base_config)
        str_reprs = []
        # for each parameter in config
        for i, (param_name, parameter) in enumerate(combinations.items()):
            pointer = config
            for name in parameter["dict_path"][:-1]:
                # finish when pointing at second-last element from path
                pointer = pointer[name]
            # set desired value
            pointer[parameter["dict_path"][-1]] = p[i]
            str_reprs.append(f"{param_name}={p[i]}")
        id = "_".join(str_reprs)
        config["logger"]["name"] = id
        # config["logger"]["version"] = id
        print(f"{n}. {id}")
        configs.append(config)
    return configs


def get_last_ckpt_path(config):
    group = config["logger"]["group"]
    name = config["logger"]["name"]
    last_ckpt_path = list(
        (Path("./checkpoints").absolute() / group / name).glob("last*.ckpt")
    )
    if last_ckpt_path:
        return last_ckpt_path[0]
    else:
        return None


def mark_ckpt_as_finished(group, name):
    last_ckpt_path = list(
        (Path("./checkpoints").absolute() / group / name).glob("last*.ckpt")
    )[0]
    last_ckpt_path.rename(last_ckpt_path.with_stem(f"{last_ckpt_path.stem}_final"))
