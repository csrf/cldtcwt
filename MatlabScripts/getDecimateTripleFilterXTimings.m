% Time how long it takes to run DecimateFilterX with the
% symmetric extension kernel beforehand.  All times are in ms.  Saves
% results to decimateTripleFilterTimings.mat.

imageSizes = int32(1.5.^(-10:0.5:1)' * [1280 720] / 2) * 2;

% Vary the size of the image
tSizes = nan(size(imageSizes, 1),2);
for n = 1:size(tSizes, 1)

    tSizes(n,1) = ...
        3*runSpeedTestDecimateFilterX(imageSizes(n,:), 14, true);

    tSizes(n,2) = ...
        runSpeedTestDecimateTripleFilterX(imageSizes(n,:), 14, true);

end

save('decimateTripleFilterTimings.mat', ...
     'imageSizes', 'tSizes');




