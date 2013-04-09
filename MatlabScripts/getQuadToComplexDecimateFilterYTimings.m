% Time how long it takes to run QuadToComplexDecimateFilterY with the
% symmetric extension kernel beforehand.  All times are in ms.  Saves
% results to quadToComplexTimings.mat.

imageSizes = int32(1.5.^(-10:0.5:1)' * [1280/2 720] / 4) * 4;

% Vary the size of the image
tSizes = nan(size(imageSizes, 1),3);
for n = 1:size(tSizes, 1)

    tSizes(n,1) = ...
        runSpeedTestDecimateFilterY(imageSizes(n,:), 14, true);

    tSizes(n,2) = ...
        runSpeedTestQuadToComplex(double(imageSizes(n,:)) ./ [1 2], true);

    tSizes(n,3) = ...
        runSpeedTestQuadToComplexDecimateFilterY(imageSizes(n,:), 14, true);

end

save('quadToComplexTimings.mat', ...
     'imageSizes', 'tSizes');




