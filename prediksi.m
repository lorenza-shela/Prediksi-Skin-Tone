clc; clear; close all;

%% 1. Set Path ke Dataset
datasetPath = 'dataSkintone';

%% 2. Load Dataset Menggunakan imageDatastore
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% 3. Pisahkan Dataset Menjadi Training (80%) & Testing (20%)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

%% 4. Tentukan Ukuran Gambar untuk Input CNN
inputSize = [64 64 3]; % 64x64 pixel, 3 channel (RGB)

%% 5. Augmentasi Data (Meningkatkan Akurasi)
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...   % Lebih banyak rotasi
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

augmentedTrainData = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augmentedTestData = augmentedImageDatastore(inputSize, imdsTest);

%% 6. Definisikan Arsitektur CNN (Lebih Dalam & Lebih Banyak Filter)
layers = [
    imageInputLayer(inputSize, 'Name', 'input')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'batchnorm3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')

    dropoutLayer(0.5, 'Name', 'dropout1') % Mencegah overfitting

    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu4')

    dropoutLayer(0.5, 'Name', 'dropout2')

    fullyConnectedLayer(4, 'Name', 'fc2') % 4 kelas skin tone
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

%% 7. Tentukan Parameter Pelatihan (Hyperparameter Tuning)
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ... % Tambah epoch agar lebih stabil
    'MiniBatchSize', 64, ... % Batch lebih besar
    'InitialLearnRate', 0.0001, ... % Lebih kecil agar tidak fluktuatif
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedTestData, ...
    'ValidationFrequency', 5, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto'); % Bisa jalan di GPU/CPU

%% 8. Latih Model CNN
net = trainNetwork(augmentedTrainData, layers, options);

%% 9. Evaluasi Model
YPred = classify(net, augmentedTestData);
YTest = imdsTest.Labels;

% Hitung Akurasi
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Akurasi Model: %.2f%%\n', accuracy * 100);

% Plot Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix - Skin Tone Classification');

%% 10. Simpan Model untuk Digunakan di GUI
save('trainedSkinToneCNN.mat', 'net');

