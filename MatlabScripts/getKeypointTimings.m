% Time how long it takes to run the 4-scale interleaved transform on test.bmp,
% detect the keypoints and describe them.  This is done for a range of 
% numbers of keypoints.

% All times are in ms.  Saves results to keypointTimings.mat.

image = 'test.bmp';
numIterations = 10000;
startLevel = 2;
numLevels = 3;

img = imread(image);

maxNumDescriptors = [1 100:100:700];

% Measure the transform and detection
t = runSpeedTestKeypoints(image, startLevel, numLevels, ...
                          numIterations, 1000);

tTransform = t(1);
tDetection = t(2);

% Measure the extraction
numDescriptors = nan(1,length(maxNumDescriptors));
tDescriptors = nan(1,length(maxNumDescriptors));
for m = 1:length(maxNumDescriptors)

    [t, numDescriptors(m)]...
         = runSpeedTestKeypoints(image, startLevel, numLevels, ...
                                 numIterations, maxNumDescriptors(m));
    tDescriptors(m) = t(3);

end

save('keypointTimings.mat', ...
     'tTransform', 'tDetection',...
     'numDescriptors', 'tDescriptors');




