function [dt, numKeypoints] ...
              = runSpeedTestKeypoints(filename, startLevel, numLevels,...
                                      numIterations, maxNumKeypoints)
% [dt, numKeypoints] 
%    = runSpeedTestKeypoints(filename, startLevel, numLevels,
%                              numIterations, maxNumKeypoints)
% 
% Runs clDTCWT-SpeedTest-Keypoints, and returns the results (in ms) as
% numbers.  filename is the image to operate on.  startLevel is the level
% number to start producing outputs at; numLevels how many levels of output
% to produce; numIterations the number of executions of each stage to average
% over; maxNumKeypoints, the maximum number of descriptors to extract.
%
% Returns the times for the 4-sDTCWT, the keypoint detection and the keypoint
% extraction respectively.  numKeypoints is the number of descriptors
% actually extracted.
%
% Keypoint detection uses the product of subbands at right angles to each other.

    % MATLAB introduces something to the path which prevents the code from 
    % running.  This eliminates its contributions (temporarily) to avoid the
    % problem.
    l = getenv('LD_LIBRARY_PATH');
    setenv('LD_LIBRARY_PATH', '');

    % Call with dimensions, filter length and number of runs to test over
    [~, t] = system(['clDTCWT-SpeedTest-Keypoints ' ...
                     filename ' ' ...
                     num2str(startLevel) ' ' num2str(numLevels) ' ' ...
                     num2str(numIterations) ' ' num2str(maxNumKeypoints)]);

    setenv('LD_LIBRARY_PATH', l);

    % Split into lines
    t = regexp(t, '.*', 'match', 'dotexceptnewline');

    % For each line, read out its number
    dt = [];
    for n = 3:5
        dt(end+1) = str2num(regexprep(t{n}, '.* (.*)ms.*', '$1'));
    end

    numKeypoints = str2num(regexprep(t{2}, '.* (.*)', '$1'));

end

