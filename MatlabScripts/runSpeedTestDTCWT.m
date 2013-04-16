function [dt] = runSpeedTestDTCWT(dims, numLevels)
% [dt] = runSpeedTestDTCWT(dims, numLevels)
% 
% Runs clDTCWT-SpeedTest-DTCWT, and returns the results (in ms) as
% numbers.  dims is a two-element vector containing the size of the image.
% numLevels is the number of levels to calculate.  Level 1 is not calculated.
% Runs each kernel 10,000 times.

    % Call with dimensions, filter length and number of runs to test over
    [~, t] = system(['clDTCWT-SpeedTest-DTCWT ' ...
                     num2str(dims(1)) ' ' num2str(dims(2)) ' '...
                     num2str(numLevels) ' 10000']);

    % Split into lines
    t = regexp(t, '.*', 'match', 'dotexceptnewline');

    % For each line, read out its number
    dt = str2num(regexprep(t{end}, '(.*)ms.*', '$1'));

end

