from fdi_pln_2608_p5.modules.train import train_model


EXPERIMENTS = [
    {"name": "base", "d_model": 128, "n_heads": 4, "n_layers": 4, "dropout": 0.1},
    {"name": "small", "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.1},
    {"name": "deeper", "d_model": 128, "n_heads": 4, "n_layers": 6, "dropout": 0.1},
    {
        "name": "dropout_high",
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "dropout": 0.3,
    },
]


def run_experiments(resources_path="resources", epochs=3):
    for exp in EXPERIMENTS:
        print(f"\n=== Experimento: {exp['name']} ===")
        train_model(
            resources_path=resources_path,
            epochs=epochs,
            d_model=exp["d_model"],
            n_heads=exp["n_heads"],
            n_layers=exp["n_layers"],
            dropout=exp["dropout"],
            save_dir=f"checkpoints/{exp['name']}",
        )
