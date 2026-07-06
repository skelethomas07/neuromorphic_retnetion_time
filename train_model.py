from model_utils import train_and_save

if __name__ == "__main__":
    summary = train_and_save(
        data_path="data/retention_dataset.xlsx",
        model_dir="models",
        fast_demo=True,
    )
    print("Training finished.")
    for key in ["n_rows_after_filtering", "n_features", "r2_log_holdout", "rmse_log_holdout", "mae_log_holdout"]:
        print(f"{key}: {summary[key]}")
