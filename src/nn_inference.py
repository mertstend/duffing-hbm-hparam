import torch
import joblib


def evaluate_Duffing_nn_H3(NN_id, X):
    model_path = f'models/duffing_mlp_h3_{NN_id}.pt'
    NN_model = torch.load(model_path, weights_only=False)
    NN_model.eval()

    nn_input = torch.tensor(X, dtype=torch.float32)

    scaler = joblib.load(f'models/duffing_scaler_h3_{NN_id}.joblib')
    X_mean = torch.tensor(scaler['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(scaler['X_std'], dtype=torch.float32)
    y_mean = torch.tensor(scaler['y_mean'], dtype=torch.float32)
    y_std = torch.tensor(scaler['y_std'], dtype=torch.float32)
    nn_input_scaled = (nn_input - X_mean) / X_std

    with torch.no_grad():
        output_scaled = NN_model(nn_input_scaled)

    output = output_scaled * y_std + y_mean

    return output.detach().numpy()
