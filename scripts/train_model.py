import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib


MODE = 'inference'  # 'train' or 'inference'
SAVE = False  # whether to save the trained model

# load data
model_id = '2026-02-18_13-29-30'
data_id = '2026-02-18_14-04-47'
train_data = np.load('data/duffing_train_data_H3_N64_'+data_id+'.npz')
X_train = train_data['q_coeffs']  # input: [c1, s1, c3, s3]
y_train = train_data['fnl_coeffs']  # output: [c1, s1, c3, s3]
test_data = np.load('data/duffing_test_data_H3_N64_'+data_id+'.npz')
X_test = test_data['q_coeffs']
y_test = test_data['fnl_coeffs']
val_data = np.load('data/duffing_val_data_H3_N64_'+data_id+'.npz')
X_val = val_data['q_coeffs']
y_val = val_data['fnl_coeffs']

# convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

if MODE == 'train':
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        threshold=1e-4,
        threshold_mode='rel',
        min_lr=1e-8
    )

    n_epochs = 1000
    batch_size = 10

    train_data_losses = []
    validation_losses = []

    for epoch in range(n_epochs):
        epoch_data_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i+batch_size]
            ybatch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            y_pred = model(Xbatch)
            data_loss = loss_fn(y_pred, ybatch)
            loss = data_loss
            loss.backward()
            optimizer.step()

            epoch_data_loss += data_loss.item()
            n_batches += 1

        train_data_losses.append(epoch_data_loss / n_batches)

        model.eval()
        with torch.no_grad():
            y_validation_pred = model(X_val)
            validation_loss = loss_fn(y_validation_pred, y_val).item()
        model.train()
        validation_losses.append(validation_loss)

        scheduler.step(validation_loss)

        # shuffle training data to avoid systematic batch artefacts
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:4d} | "
            f"train loss = {epoch_data_loss / n_batches:.3e} |"
            f"val loss = {validation_loss:.3e} | "
            f"lr = {current_lr:.3e}")

    print("Finished training with adaptive Adam")
    print(f"Final train loss: {train_data_losses[-1]:.3e}")
    print(f"Final validation loss: {validation_losses[-1]:.3e}")

    if SAVE:
        # save current date to be able to load the model later
        save_date = np.datetime64('now').astype('str').replace(
            ':', '-').replace('T', '_')
        torch.save(model, 'models/duffing_mlp_h3_'+save_date+'.pt')
        joblib.dump({'train_losses': train_data_losses,
                     'validation_losses': validation_losses},
                    'models/duffing_losses_h3_'+save_date+'.joblib')
        print(f"Model and scaler saved with date id {save_date}")


if MODE == 'inference':
    model = torch.load('models/duffing_mlp_h3_'+model_id+'.pt',
                       weights_only=False)
    model.eval()

print(model)

# test a single sample
i = 1
X_sample = X_test[i:i+1]
model.eval()
with torch.no_grad():
    y_pred = model(X_sample)
print(f"Sample:\n{X_sample[0]}\n-> Prediction:\n{y_pred[0]}\n" +
      f"-> Ground truth:\n{y_test[i]}")

# evaluate the model on the test set
with torch.no_grad():
    y_pred = model(X_test)
mae = (y_pred - y_test).abs().mean(axis=0)
print(f"mean difference over test set: {np.round(np.asarray(mae), 4)}")
