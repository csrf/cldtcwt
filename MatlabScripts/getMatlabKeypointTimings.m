% Time how long it takes to run the 4-scale interleaved transform on test.bmp,
% detect the keypoints and describe them.  This is done for a range of 
% numbers of keypoints.

% All times are in ms.  Saves results to matlabKeypointTimings.mat.

image = 'test.bmp';
numIterations = 100;

img = imread(image);

maxNumDescriptors = [1 100:100:700];

tic;
for n = 1:numIterations
    % Calculate four trees, starting from Level 2
    Yh = computeFourTrees(img, 2);
end
tTransform = toc / numIterations * 1e3;

tic;
for n = 1:numIterations
    pos = detectKeypointsRightAngles(img, Yh);
end
tDetection = toc / numIterations * 1e3;

numDescriptors = nan(1,length(maxNumDescriptors));
tDescriptors = nan(1,length(maxNumDescriptors));
for m = 1:length(maxNumDescriptors)

    % Limit the number of descriptors extracted
    p = pos(1:min(maxNumDescriptors(m), size(pos,1)), :);

    tic;
    for n = 1:numIterations


        % Extract them
        X = get2Scale4treePMatDescriptors_fast(Yh, p, false, ...
                                               size(img,2), size(img,1));
    end
    
    tDescriptors(m) = toc / numIterations * 1e3;
    numDescriptors(m) = size(p,1);

end

save('matlabKeypointTimings.mat', ...
     'tTransform', 'tDetection',...
     'numDescriptors', 'tDescriptors');




