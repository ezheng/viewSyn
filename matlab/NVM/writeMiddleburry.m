function writeMiddleburry(cameras, outputFileName)


fid = fopen(outputFileName, 'w');
fprintf(fid, '%i\n', numel(cameras));

for i = 1:numel(cameras)
    fprintf(fid, '%s ', cameras(i).name);
%     writing K matrix
    K = [ cameras(i).focalLength, 0 , cameras(i).width /2 ; 0, cameras(i).focalLength, cameras(i).height/2.0; 0 0 1 ];
    K = K';
    for j = 1:9
        fprintf(fid, '%.6f ', K(j));
    end
        
%     writing R matrix    
    R = quaternion2matrix(cameras(i).quarternion);
    RR = R(1:3,1:3)';
    for j = 1:9
        fprintf(fid, '%.12f ', RR(j));
    end
    
%   writing T vector    
    C = cameras(i).pos';
    T = -R(1:3,1:3)*C;
    fprintf(fid, '%.12f %.12f %.12f\n', T(1), T(2), T(3));
    
end
fclose(fid);





