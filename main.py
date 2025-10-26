from data import *
from train import train, test_model
from networks.static_models import *
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import wandb
import warnings
import argparse

warnings.filterwarnings("ignore")


def run_kfold(df_path, dataset, model_name, config, device):

    th = config["th"]
    df = pd.read_csv(df_path)
    sample_dataset = get_dataset_class(dataset)(df[0:10], th, train=True)
    num_nodes = sample_dataset.nrois
    label = sample_dataset.LABEL
    skf = model_selection.StratifiedKFold(n_splits=5)
    skf.get_n_splits(df, df[label])

    accs = []
    senss = []
    specs = []
    k = 0

    for train_index, test_index in skf.split(df, df[label]):
        k += 1
        train_val_df = df.iloc[train_index]
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15, stratify=train_val_df[label], random_state=0
        )
        test_df = df.iloc[test_index]

        train_loader = DataLoader(
            get_dataset_class(dataset)(train_df, th, train=True),
            batch_size=config["batch_size"],
            num_workers=3,
            pin_memory=False,
        )
        val_loader = DataLoader(
            get_dataset_class(dataset)(val_df, th, train=False),
            batch_size=config["batch_size"],
            num_workers=3,
            pin_memory=False,
        )
        test_loader = DataLoader(
            get_dataset_class(dataset)(test_df, th, train=False),
            batch_size=config["batch_size"],
            num_workers=3,
            pin_memory=False,
        )
        model = get_model_class(model_name)(num_nodes, **config)

        trained_model = train(model, train_loader, val_loader, config, device)
        acc, sens, spec = test_model(model, test_loader, device)

        accs.append(acc)
        senss.append(sens)
        specs.append(spec)
        print(
            "for fold {}, Acc = {:.3f}, Sens = {:.3f}, Spec  = {:.3f}".format(
                k, acc, sens, spec
            )
        )
        print("-" * 30)

    accs = np.array(accs)
    acc_mean = np.round(np.mean(accs), 3)
    acc_std = np.round(np.std(accs), 3)
    senss = np.array(senss)
    sens_mean = np.round(np.mean(senss), 3)
    sens_std = np.round(np.std(senss), 3)
    specs = np.array(specs)
    spec_mean = np.round(np.mean(specs), 3)
    spec_std = np.round(np.std(specs), 3)

    print(f"{model_name} on {dataset} dataset 5-fold results:")
    print(
        " Test Accuracy: mean = {:.3f} % ,std = {:.3f}".format(
            acc_mean * 100, acc_std * 100
        )
    )
    print(
        " Test Sens: mean = {:.3f} % ,std = {:.3f}".format(
            sens_mean * 100, sens_std * 100
        )
    )
    print(
        " Test Spec: mean = {:.3f} % ,std = {:.3f}".format(
            spec_mean * 100, spec_std * 100
        )
    )

    wandb.log(
        {
            "Test_acc": acc_mean,
            "Test_sens": sens_mean,
            "Test_spec": spec_mean,
            "std": [acc_std, sens_std, spec_std],
        }
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1", help="device")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Abide",
        help='Dataset name, valid options are ["Abide", "Mddrest", "UKBB_N"]',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        help='Static model name. valid options are ["GCN", "GIN", "GAT"]',
    )

    args = parser.parse_args()

    np.random.seed(0)
    device = torch.device(args.device)

    dataset = args.dataset
    model_name = args.model
    df_paths = {"Abide": "../Data/abide.csv"}
    df_path = df_paths[args.dataset]

    # default hyperparameters, you can search for optimal conifguration using wandb sweep feature
    hparams_defaults = {
        "f1": 128,
        "f2": 64,
        "layers": 3,
        "readout": "meanmax",
        "batch_size": 32,
        "lr": 3e-4,
        "ls": 0.1,
        "th": 0.5,
        "extra_fc": True,
        "num_heads": 1,
        "concat": False,
    }

    config = hparams_defaults
    wandb.init(config=hparams_defaults, project=model_name + "-" + dataset)
    config = wandb.config

    print(f"Running a {model_name} on the {dataset} dataset.")
    print("*" * 50)
    if dataset in ["Abide"]:
        run_kfold(df_path, dataset, model_name, config, device)

    print("-" * 30)
