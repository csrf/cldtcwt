function [dt] = runSpeedTestQuadToComplexDecimateFilterY(dims, filterLength, pad)
% [dt] = runSpeedTestQuadToComplex(dims)
% 
% Runs clDTCWT-SpeedTest-QuadToComplex, and returns the results (in ms) as
% numbers.  dims is a two-element vector containing the size of the image.
% Runs each kernel 10,000 times.

    % Call with dimensions, filter length and number of runs to test over
    [~, t] = system(['clDTCWT-SpeedTest-QuadToComplex ' ...
                     num2str(dims(1)) ' ' num2str(dims(2)) ' 10000']);

    % For each line, read out its number
    dt = str2num(regexprep(t, '.* (.*) ms', '$1'));

end

