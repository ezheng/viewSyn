function convertEPFLToMiddleBurry()
% need to change K matrix in line 21
fileListName = 'fountain.txt';
imagePath = 'C:\Users\ezheng\Dropbox\temp\fromLiang\images\quarter_size\';
motionPath = 'C:\Users\ezheng\Dropbox\temp\fromLiang\motion';

allImageNames = dir( fullfile(imagePath, '*.png'));
numOfImages = numel(allImageNames);

% get the rotation matrix and translation matrix
allCameraNames = dir(fullfile(motionPath, '*.camera'));
RC = zeros(4, 3, numOfImages);
for i = 1:numOfImages
    cameraPoseFile = fullfile(motionPath, allCameraNames(i).name);
    RC(:,:,i) = dlmread(cameraPoseFile);
end
    
% write the file
fid = fopen(fileListName, 'w');
fprintf(fid, '%i\n', numOfImages);
K = [2759.48/4.0, 0, 1520.69/4.0; 0, 2764.16/4, 1006.81/4; 0, 0, 1  ]; K = K' ; K = K(:);
for i = 1: numOfImages
%     fileName:
    fprintf(fid, '%s ', allImageNames(i).name); 
%     write K matrix
    for j = 1:9
        fprintf(fid, '%d ', K(j));
    end
    R = RC(1:3,:, i);
%     R = R';
    for j = 1:9
        fprintf(fid, '%d ', R(j));
    end
    T = -R' * RC(4,:,i)';
%   read and write 
    fprintf(fid, '%d %d %d\n', T(1), T(2), T(3));
    
end

fclose(fid);
