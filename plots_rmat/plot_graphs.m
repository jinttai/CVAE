clc; clear;

cvae_training_data = readtable("v1.csv");
cvae_epoch = cvae_training_data.epoch;
cvae_loss = cvae_training_data.train_loss;


figure(1)
plot(cvae_epoch, cvae_loss);
title("Generative Model Training Result", 'FontSize', 18)
xlabel("Training epoch", 'FontSize', 14);
ylabel("Loss", 'FontSize', 14);
grid on

average = mean(cvae_loss(14950:end,1));