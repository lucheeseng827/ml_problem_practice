use rust_torch::prelude::*;

fn main() {
    // Define the model and its layers
    let mut model = nn::Sequential::new();
    model.add(nn::Linear::new(3, 2));
    model.add(nn::Linear::new(2, 1));

    // Compile the model with the appropriate loss and optimizer
    let optimizer = Adam::new(model.parameters(), 0.001);

    // Train the model on the training set
    for epoch in 0..100 {
        for (x, y) in dataset {
            let y_pred = model.forward(&x);
            let loss = mse_loss(&y_pred, &y);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }
}
