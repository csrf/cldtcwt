% Time how long it takes to run DTCWT.  All times are in ms.  Saves
% results to matlabDTCWTTimings.mat.

numRuns = 100;

numLevels = 0:10;

% Vary the number of levels calculated
tNumLevels = nan(length(numLevels),1);
X = zeros(720, 1280);

for n = 1:length(numLevels)

    tic;
    for m = 1:numRuns
        [~, Yh] = dtwavexfm2b(X, numLevels(n) + 1, ...
                              'near_sym_b_bp', 'qshift_b_bp', 2);
    end
    tNumLevels(n) = toc / numRuns * 1000;

end

imageSizes = int32(1.5.^(-10:0.5:1)' * [1280 720] / 4) * 4;

% Vary the size of the image
tSizes = nan(size(imageSizes, 1),1);
for n = 1:size(tSizes, 1)

    X = zeros(imageSizes(n,2), imageSizes(n,1));

    tic;
    for m = 1:numRuns
        [~, Yh] = dtwavexfm2b(X, 7, 'near_sym_b_bp', 'qshift_b_bp', 2);
    end
    tSizes(n) = toc / numRuns * 1000;

end

save('matlabDTCWTTimings.mat', ...
     'numLevels', 'tNumLevels',...
     'imageSizes', 'tSizes');




