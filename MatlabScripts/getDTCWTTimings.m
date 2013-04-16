% Time how long it takes to run DTCWT.  All times are in ms.  Saves
% results to DTCWTTimings.mat.

numLevels = 1:10;

% Vary the number of levels calculated
tNumLevels = nan(length(numLevels),1);
for n = 1:length(numLevels)
    tNumLevels(n) = runSpeedTestDTCWT([1280 720], numLevels(n));
end

imageSizes = int32(1.5.^(-10:0.5:1)' * [1280 720] / 4) * 4;

% Vary the size of the image
tSizes = nan(size(imageSizes, 1),1);
for n = 1:size(tSizes, 1)
    tSizes(n) = runSpeedTestDTCWT(imageSizes(n,:), 6);
end

save('DTCWTTimings.mat', ...
     'numLevels', 'tNumLevels',...
     'imageSizes', 'tSizes');




