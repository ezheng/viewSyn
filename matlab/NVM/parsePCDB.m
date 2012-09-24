function [folderHierarchy, fileExt, filePath] = parsePCDB(pcdbFileName)

fid = fopen(pcdbFileName);

tline = fgetl(fid);
while ischar(tline)
    
%     parse this line
      words=regexp(tline,' +','split'); %numOfWords = numel(words);
      if strcmp(words{1},'image') && strcmp(words{2}, 'DIR')
            fileExt = words{3};
            numOffolderHierarchy = str2double(words{4});
            folderHierarchy = str2double(words(5:5 + numOffolderHierarchy - 1));
      end
      
      if strcmp(words{1},'image') && strcmp(words{2}, 'DEF')
          filePath = words{5};
          
      end
       
    tline = fgetl(fid);
end

fclose(fid);
