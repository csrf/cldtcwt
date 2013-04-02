% Time how long it takes to run FilterX and
% FilterY (and both) without the
% symmetric extension kernel beforehand.  All times are in ms.  Saves
% results to filterDecimateTapTimings.mat.

filterLengths = 2:2:16;

% Vary the number of filter taps
tNumTaps = nan(length(filterLengths),3);
for n = 1:length(filterLengths)

    tNumTaps(n,1) = ...
        runSpeedTestDecimateFilterX([1280 720], filterLengths(n), ...
                                    false);

    tNumTaps(n,2) = ...
        runSpeedTestDecimateFilterX([1280 720], filterLengths(n), ...
                                    false, 'FL');

    tNumTaps(n,3) = ...
        runSpeedTestDecimateFilterX([1280 720], filterLengths(n), ...
                                    false, 'RL');

    tNumTaps(n,4) = ...
        runSpeedTestDecimateFilterX([1280 720], filterLengths(n), ...
                                    false, 'FLRF');

    tNumTaps(n,5) = ...
        runSpeedTestDecimateFilterX([1280 720], filterLengths(n), ...
                                    false, 'RSLNT');

end

save('filterDecimateTapTimings.mat', ...
     'filterLengths', 'tNumTaps');




