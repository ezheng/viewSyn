function main()

% nvmfilename = 'undistorted_Box_highRes_new2.nvm';
% [cameras, points3d ] = readNVM(nvmfilename);
% 
% numofimages = numel(cameras);
% 
% % for i = 1:numofimages
% %     x(i) = cameras(i).pos(1);
% %     y(i) = cameras(i).pos(2);
% %     z(i) = cameras(i).pos(3);
% % end
% for i = 1:numel(points3d)
%     x(i) = points3d(i).pos(1);
%     y(i) = points3d(i).pos(2);
%     z(i) = points3d(i).pos(3);
% end
% 
% plot3(x,y,z, 'b*');
% xlabel('x'), ylabel('y'), zlabel('z');
% hold on;
% plot3(x,y,z, 'r*');
% axis equal; grid on;
% hold off;
% 
% % 
% 
% save data.mat;

 load data.mat

imageWidth = 2448;
imageHeight = 2048;

% project the color first. Then project the depth.
numOf3DPoints = numel(points3d);
numOfImages = numel(cameras);

% read all images
imageWidth = 2448; imageHeight = 2048;
images = zeros(imageHeight, imageWidth, 3, numOfImages);

numOfImages = 2;
for i = 1:numOfImages
    images(:,:,:, i) = imread(cameras(i).name{1});
    fprintf(1, 'reading image %d\n', i);
end

projectedImage = nan(imageHeight, imageWidth, 1);
projectedImage1 = nan(imageHeight, imageWidth, 1);
savingPath = '.\figure';
numOf2DPoints = 0;

addpath .\export_fig
count = 1;
for k = 2:numOfImages
    for j = 1:numOf3DPoints    % 3D points
        pos = points3d(j).pos(1:3);
        measure = points3d(j).measure;
        numOfMeasures = size(measure, 1);
        
        for i = 1: numOfMeasures
            imageIndex = measure(i, 1)+1;
            if (imageIndex == k)
                
                
                rotation = quaternion2matrix(cameras(imageIndex).quarternion);
                translation = cameras(imageIndex).pos;
%                 focalLength = cameras(imageIndex).focalLength;
                %             K = [focalLength, cameras(imageIndex).distortion(1) , imageWidth /2 ; 0 focalLength, imageHeight/2.0; 0 0 1 ];
                 fx = 2396.64459; fy = 2393.60290; cx = 1253.19835; cy = 984.31266;
%                 fx = 2396.64459, fy = 2396.64459, cx = imageWidth/2, cy = imageHeight/2;
                K = [fx, 0 , cx ; 0 fy, cy; 0 0 1 ];
                R = rotation(1:3,1:3);
                P = K * [R, -R * translation'];
                x = P * [pos'; 1];
                x = x / x(3);
                           x(1) - imageWidth/2
                           (x(2) - imageHeight/2 ) * fx/fy
                dist = norm(pos - translation);
                
                %           projectedImage(round(imageHeight - measure(i,4) - imageHeight * 0.5), round(measure(i,3) + imageWidth * 0.5), :) = points3d(j).rgb/255;
                %             projectedImage( round(measure(i,4) + imageHeight * 0.5), round(measure(i,3) + imageWidth * 0.5), :) = points3d(j).rgb/255;
                posX = round(measure(i,3) + imageWidth * 0.5);
                posY = round(measure(i,4) + imageHeight * 0.5);
                projectedImage1( posY-3:posY+3 , posX -3:posX +3 , :) = dist;
                projectedImage( posY, posX, :) = dist;
                numOf2DPoints = numOf2DPoints + 1;
                %             x(1) - imageWidth/2
                %             x(2) - imageHeight/2
            end
        end
    end
    fprintf(1, 'numOf2DPoints: %d\n', numOf2DPoints);
    h = figure(1);
    imshow(projectedImage1);
%     saveas(h, ['sparsePoints_',num2str(k),'.bmp']);
    export_fig(fullfile(savingPath,[num2str(count),'.eps']));
    count = count + 1;
    h = figure(2);
    imshow(imresize(images(:,:,:,k)/255,0.25));
%     saveas(h, ['originalImage_', num2str(k), '.bmp']);
    export_fig(fullfile(savingPath, [num2str(count), '.eps']));
    count = count + 1;
    tic;
    output = bilateralFilter( projectedImage, rgb2gray(images(:,:,:,k)/255));
    toc
    h = figure(3);
    imagesc(output);
    axis equal; colorbar;
%     saveas(h, ['depthImage_.',num2str(k),'.bmp']);
    export_fig(fullfile(savingPath, [num2str(count),'.eps']));
    count = count + 1;
end


% K = [2370.9296875, 0 , 2448 /2 ; 0, 2370.9296875, 2048/2.0; 0 0 1 ];
% quat = [0.862849047833 0.385440592559 -0.290581093148 -0.149966840322];
% rotation = quaternion2matrix(quat);
% R = rotation(1:3,1:3);
% translation = [1.98945008166 -0.0115062563055 2.15161034409];
% P = K * [R, -R * translation'];
% pos = [10.7882448226 11.693774716 6.5714832538];
% x = P * [pos'; 1];
% x = x / x(3);
% x(1) - 2448/2
% x(2) - 2048/2

    