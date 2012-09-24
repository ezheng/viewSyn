function image = read_middleBurry(fileName)

% fileName = 'C:\Enliang\matlab\patchBased\fountain\fountain - Copy.txt'; 

imageIdx = 0;
fid = fopen(fileName, 'r');
assert(fid ~= 0);
tline = fgets(fid); % read the number of image
tline = fgets(fid);
while ischar(tline)
    imageIdx = imageIdx + 1;      
    words=regexp(tline,' +','split');
    
    image(imageIdx).imageName = words{1};
    K = str2double( words(2:10)); K = reshape(K, 3,3); K = K'; 
    image(imageIdx).K = K;
    R = str2double( words(11:19)); R = reshape(R, 3,3); R = R';
    image(imageIdx).R = R;    
    image(imageIdx).T = str2double( words(20:22))';
    image(imageIdx).C = -image(imageIdx).R' * image(imageIdx).T;
    tline = fgets(fid);
end
fclose(fid);

