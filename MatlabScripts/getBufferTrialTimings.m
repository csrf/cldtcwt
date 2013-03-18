% Time how long it takes to run FilterX and FilterY (and both) with the
% symmetric extension kernel beforehand.  All times are in ms.  Saves
% results to filterTimings.mat.

filterLengths = 1:2:15;

% Vary the number of filter taps
tNumTaps = nan(length(filterLengths),3);
for n = 1:length(filterLengths)
    tNumTaps(n,:) = ...
        runSpeedTestBufferTrial([1280 720], filterLengths(n), true);
end

imageSizes = int(1.5.^(-10:0.5:1)' * [1280 720]);

% Vary the size of the image
tSizes = nan(size(imageSizes, 1),3);
for n = 1:size(tSizes, 1)
    tSizes(n,:) = ...
        runSpeedTestBufferTrial(imageSizes(n,:), 13, true);
end

save('filterTimings.mat', ...
     'filterLengths', 'tNumTaps',
     'imageSizes', 'tSizes');




