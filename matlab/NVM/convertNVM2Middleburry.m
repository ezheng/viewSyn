function convertNVM2Middleburry()
clc;

nvmFileName = 'C:\Enliang\data\randObject\randObject.nvm';
pcdbFileName = 'C:\Enliang\data\randObject\pcdb.txt';
outputFileName = 'randObject.txt' ;

% ------------------------------------------------------------------

[folderHierarchy, fileExt, filePath] = parsePCDB(pcdbFileName);
if( strcmp(fileExt, '$'))
    fileExt = [];
end

[pathstr, name, ~] = fileparts(nvmFileName);
if ~exist(fullfile(pathstr, [name, '.mat']), 'file')
    
    [camera, points3d] = readNVM(nvmFileName);
    % change the 'name' in 'camera' to full file path
    for i = 1:numel(camera)
        %     i
        if mod(i, 100) == 0
            fprintf( '%f%% percent is finished\n', i/numel(camera) *100 );
        end
        
        filename = camera(i).name;
        
        if(~isempty(folderHierarchy))
            folderHierarchy_str = [filename(1:folderHierarchy(1)),'/'];
            base = folderHierarchy(1);
            for j = 2: numel(folderHierarchy)
                folderHierarchy_str = [folderHierarchy_str,filename(base+1: (base + folderHierarchy(j))),'/'];
                base = base + folderHierarchy(j);
            end
        else 
            folderHierarchy_str = [];
        end
        camera(i).name = fullfile(filePath, folderHierarchy_str, [filename,'.',fileExt]);
        img = imread(camera(i).name);
        camera(i).height = size(img, 1);
        camera(i).width = size(img, 2);
        camera(i).depth = size(img, 3);        
        %     imageinfo = imfinfo(camera(i).name);
        %     camera(i).height = imageinfo.height;
    end
    save(fullfile(pathstr, [name, '.mat']));
else
    load(fullfile(pathstr, [name, '.mat']));
end

outputFileName = fullfile(pathstr, outputFileName);
writeMiddleburry(camera, outputFileName);

