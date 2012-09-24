function convertEPFL2Middleburry()
outputFileName = 'fountain.txt';

workingDir = 'C:\Enliang\data\epfl\fountain';
poseDir = fullfile(workingDir, 'camera');
imageDir = fullfile(workingDir, 'image');
allImgs = dir(fullfile(imageDir, '*.png' ));
allCameraPose = dir(fullfile(poseDir, '*.png.camera'));
assert(numel(allImgs) == numel(allCameraPose));
% ------------------------------------------

numOfImages = numel(allImgs);

fid = fopen(outputFileName, 'w');
assert(fid>0);

fprintf(fid, '%i\n', numOfImages);
for i = 1:numOfImages
    imageName = fullfile(imageDir, allImgs(i).name);
    cameraPoseName = fullfile(poseDir, allCameraPose(i).name);
    [K, R, T] = readCameraPoses(cameraPoseName);
    
    fprintf(fid, '%s ', imageName); 
%     write K matrix
    KK = K';
    for j = 1:9
        fprintf(fid, '%6f ', KK(j));
    end
%     write R matrix
    RR = R';
    for j = 1:9
        fprintf(fid, '%.12f ', RR(j));
    end
%     write T
    fprintf(fid, '%.12f %.12f %.12f\n', T(1), T(2), T(3));

end
fclose(fid);

end

function [K,R,T] = readCameraPoses(fileName)
    data = dlmRead(fileName);
    
    K = data(1:3,:);
    R = data(5:7,:);
    R = R';
    T = -R * data(8,:)';
    
end




