function [dt] = runSpeedTestDecimateTripleFilterX(dims, filterLength, pad)
% [dt] = runSpeedTestDecimateTripleFilterX(dims, filterLength, pad)
% 
% Runs clDTCWT-SpeedTest-DecimateFilterX, and returns the results (in ms) as
% numbers.  dims is a two-element vector containing the size of the image.
% filterLength is the number of taps to use in the filter; pad is whether to 
% run the symmetric extension kernel before each call to the filtering kernel.
% Runs each kernel 10,000 times.

    % Call with dimensions, filter length and number of runs to test over
    [~, t] = system(['clDTCWT-SpeedTest-DecimateTripleFilterX ' ...
                     num2str(dims(1)) ' ' num2str(dims(2)) ' '...
                     num2str(filterLength) ' 10000 '...
                     num2str(pad)]);

    % For each line, read out its number
    dt = str2num(regexprep(t, '.* (.*) ms', '$1'));

end

