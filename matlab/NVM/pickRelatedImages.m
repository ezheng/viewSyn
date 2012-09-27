function pickRelatedImages( output_NVMFileName, input_NVMfileName, refCamId)
% Input: nvm file
% Output: nvm file, with related cameras

% input_NVMfileName = 'randObject.nvm';
% refCamId = 10; % this id starts from 1
% output_NVMFileName = 'randObject_picked.nvm';
% -------------------------------------------------------------------------
[camera, points3D] = readNVM(input_NVMfileName);

numOfAllCameras = numel(camera);
numOf3DPoints = numel(points3D);
votes = zeros(numOfAllCameras, 1);
d = 0;
for i = 1:numOf3DPoints
    measures = points3D(i).measure; %        imageIdx, featureIdx, x, y
    measures = measures + 1;
    if( ~isempty( find(measures(:,1)==refCamId)))
        votes( measures(:,1)) = votes( measures(:,1)) + 1;
         d = d + 1;
    end    
end

[~, idx] = sort(votes, 1, 'descend');
writePickedCamNVM(camera(idx), output_NVMFileName);

end


function writePickedCamNVM( cameras, output_NVMFileName)
    fid = fopen(output_NVMFileName, 'w');
    assert(fid > 0);
    fprintf(fid, 'NVM_V3\n\n');
    fprintf(fid, '%i\n', numel(cameras));
    
    for i = 1:numel(cameras)
        fprintf(fid, '%s %.8f ', cameras(i).name, cameras(i).focalLength);
        fprintf(fid, '%.12f %.12f %.12f %.12f ', cameras(i).quarternion(1), ...
            cameras(i).quarternion(2), cameras(i).quarternion(3), cameras(i).quarternion(4));
        fprintf(fid, '%.12f %.12f %.12f ', cameras(i).pos(1), cameras(i).pos(2), cameras(i).pos(3));
        fprintf(fid, '%.12f %f\n',  cameras(i).distortion(1),  cameras(i).distortion(2));        
    end
    
    fprintf(fid, '\n');
    fclose(fid);
end



