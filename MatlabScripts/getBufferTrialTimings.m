
t = zeros(0,3);

for n = 1:2:15
    t((n+1)/2,:) = runSpeedTestBufferTrial([1280 720], n, true);
end

