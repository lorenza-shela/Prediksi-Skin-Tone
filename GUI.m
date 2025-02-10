function SkinToneClassificationGUI
    % Load Trained Model
    load('trainedSkinToneCNN.mat', 'net');
    
    % Create GUI Figure
    fig = uifigure('Name', 'Skin Tone Classifier by Lorenza Shela Tansyah', 'Position', [100 100 500 400]);
    
    % Judul
    uilabel(fig, 'Text', 'PREDIKSI SKIN TONE WAJAH', ...
        'FontSize', 16, 'FontWeight', 'bold', ...
        'Position', [180 450 300 30]);

    % UI Components
    btnUpload = uibutton(fig, 'Text', 'Upload Image', 'Position', [50 320 150 40], 'ButtonPushedFcn', @uploadImage);
    btnPredict = uibutton(fig, 'Text', 'Tampilkan Hasil', 'Position', [250 320 150 40], 'ButtonPushedFcn', @predictImage);
    ax = uiaxes(fig, 'Position', [100 80 300 200]);
    ax.XColor = 'none';
    ax.YColor = 'none';
    
    lblResult = uilabel(fig, 'Text', '', 'Position', [150 50 200 30], 'FontSize', 14, 'FontWeight', 'bold');
    
    % Variables
    img = [];
    
    % Upload Image Function
    function uploadImage(~, ~)
        [file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'});
        if isequal(file, 0)
            return;
        end
        img = imread(fullfile(path, file));
        imshow(img, 'Parent', ax);
    end
    
    % Predict Image Function
    function predictImage(~, ~)
        if isempty(img)
            uialert(fig, 'Silakan unggah gambar terlebih dahulu.', 'Peringatan');
            return;
        end
        
        % Resize Image
        imgResized = imresize(img, [64 64]);
        
        % Predict
        label = classify(net, imgResized);
        
        % Display Result
        lblResult.Text = sprintf('Prediksi: %s', string(label));
    end
end
